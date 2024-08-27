//
// Created by huangyuyang on 6/1/23.
//

#include "utils.h"

#include "llama.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    std::vector <float> GetInterLeavePowerOf2(int n) {
        float start = powf(2, -powf(2, -(log2f(n) - 3)));
        float ratio = start;
        std::vector <float> ret;
        for (int i = 0; i < n; i++) {
            ret.push_back(start * powf(ratio, i));
        }
        return ret;
    }
    std::vector <float> GetInterleave(int n) {
        int base = 1;
        while (base < n) {
            base <<= 1;
        }
        if (base == n) {
            return GetInterLeavePowerOf2(n);
        } else {
            std::vector <float> ret = GetInterLeavePowerOf2(base / 2);
            std::vector <float> part2 = GetInterLeavePowerOf2(base);
            for (int i = 0; i < n - base / 2; i++) {
                ret.push_back(part2[i * 2]);
            }
            return ret;
        }
    }

    LlamaModel::LlamaModel() {
        this->model_struct = "llama";
        this->model_type = "llama";

        // 默认使用 llama3 的提示词和instruction
        this->pre_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>";
        this->user_role="<|start_header_id|>user<|end_header_id|>\n";
        this->bot_role="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
        this->history_sep="<|eot_id|>\n";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.mlp.down_proj.weight", "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",  "model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.gateup_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight"
        };
    }

    void LlamaModel::InitParams() {
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        head_dim = embed_dim / num_attention_heads;
        rotary_dim = head_dim;
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            std::string type = this->weight.dicts["rope_scaling.type"];
            if (type == "linear")
               rope_type = RoPEType::LINEAR_SCALE;
            else if (type == "dynamic")
               rope_type = RoPEType::DYMAMIC_NTK;
        }
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor, std::max(max_positions, 16384));
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));
    }

    std::pair<std::vector<float>, std::vector<float>> LlamaModel::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }
        float scale = rope_type == RoPEType::LINEAR_SCALE ? factor : 1.0;
        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i / scale * invFreq[j]);
                cos[i][j] = ::cos((float)i / scale * invFreq[j]);
            }
        }
        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }
    //llama.cpp
    //推理函数，实际上是batch=1的特殊情况
    int LlamaModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        //调用ForwardBatch的vector返回值的第一个
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    //真正做推理的函数
    //返回一个vector数组，长度为batch大小，batch的预测的下一个token id的结果
    //batch: batch大小
    //inputIds: 用户输入的数据通过一系列处理后的token id, 是一个维度为{1, seqLen}的Data类
    //attentionMask： 输出时不能看到未来的信息，是一个维度{seqLen, seqLen}的矩阵，矩阵对角线以上为1，遮挡后面信息
    //posistionIds：还没有经过处理的位置编码，维度为{1, seqLen}的向量，值从0开始递增
    //pastKeyValues： 保存历史信息的KV Cache，每个Transofmer的block都有自己的KV Cache
    //===============================上面这些变量是Transofmer计算过程中用到的=============================================
    //===============================下面这些变量是Transfomer计算结束后预测token甬道的======================================
    //generationConfig: 需要超参数配置的时候会用到，top_k, top_p temperature等
    //lastTokens: 在预测时需要进行repeat_penalty会用到
    //retLogits: 当config中设置了需要打印retLogits时，会把预测的logits存入这个数组
    std::vector <int> LlamaModel::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        //在计算Q，K，V三个矩阵的时候，需要做三次矩阵乘法
        //输入X:[seqLen, h]分别乘以W_Q, W_K, W_V三个[h, h]矩阵
        //得到输出Q，K，V，三个[seqLen, h]矩阵
        //通过Cat算子合并QKV三个矩阵，可以做一次乘法运算得到三个结果
        //合并后的mergeQKV:[h, 3h]
        //X * mergeQKV = [seqLen, h] * [h, 3h] = [seqLen, 3h]
        //这个结果包含了QKV的所有结果，之后可以通过Split算子分割得到各自的QKV
        //这里的3h只是一种指代，方便理解，具体数值要看模型设计
        if (!mergeQKV) {
            bool canMerge = true;
            for (int i = 0; i < block_cnt; i++) {
                //qwen2没有这个权重，有的llama模型可以直接提供合并后的权重，权重名的W_pack已经说名是合并后的
                std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
                //Q，K,V各自的权重名字
                std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
                std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
                std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
                std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
                std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
                std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
                //我们新加入的mergeQKV的权重名字
                std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                //qwen2没有提供合并后的权重，提供了就没有需要我们做的
                //其实这里的命名很有迷惑性，weight.weight，第一个weight是WeightMap,第二个weight才是模型权重，
                //WeightMap除了权重还有tokenizer等其他东西
                if (weight.weight.find(qkvWeightName) != weight.weight.end() || weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    mergeQKV = true;
                    break;
                } else {
                    //找出weight和bias的数据
                    Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                    Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                    Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();

                    Data &q = weight.weight[qWeightName];
                    Data &k = weight.weight[kWeightName];
                    Data &v = weight.weight[vWeightName];
                    
                    //INT4G之前做量化的时候提过，如果原来的列维度/分组数如果不能整除需要padding
                    //由于有padding，就不能随便合并在一起，不同量化信息可能会出现在同一行
                    if ((q.dataType == DataType::INT4_GROUP && q.dims[1] % q.groupCnt != 0) || 
                        (k.dataType == DataType::INT4_GROUP && k.dims[1] % k.groupCnt != 0) ||
                        (v.dataType == DataType::INT4_GROUP && v.dims[1] % v.groupCnt != 0)) {
                        canMerge = false;
                        break;
                    }
                    //这里把bias合并在一起，得到一个[3h]的bias向量
                    if (weight.weight.find(qBiasName) != weight.weight.end()) {
                        Data middle;
                        Cat(qBias, kBias, -1, middle);
                        Cat(middle, vBias, -1, weight.weight[mergeQkvBiasName]);
                        weight.weight[mergeQkvBiasName].name = mergeQkvBiasName;
                    } else {
                        weight.weight[mergeQkvBiasName] = Data();
                    }
                    //合并weight,可以看到dims[0] = 3h, dims[1] = h
                    //至于为什么是[3h, h]，因为fastllm的Linear计算是会转置权重的，实际计算的时候还是看作[h, 3h]
                    weight.weight[mergeQkvWeightName] = Data(q.dataType, {q.dims[0] + k.dims[0] + v.dims[0], q.dims[1]});
                    Data &mergeQKV = weight.weight[mergeQkvWeightName];
                    //分配合并后的空间，并且使用memcpy拷贝数据
                    mergeQKV.name = mergeQkvWeightName;
                    mergeQKV.Allocate();
                    memcpy(mergeQKV.cpuData, q.cpuData, q.GetBytes());
                    memcpy(mergeQKV.cpuData + q.GetBytes(), k.cpuData, k.GetBytes());
                    memcpy(mergeQKV.cpuData + q.GetBytes() + k.GetBytes(), v.cpuData, v.GetBytes());
                    //INT4G量化的相关数据也要保存
                    mergeQKV.group = q.group;
                    mergeQKV.groupCnt = q.groupCnt;
                    mergeQKV.perChannelAxis = q.perChannelAxis;
                    mergeQKV.perChannelsConfigs = AppendVector(q.perChannelsConfigs, AppendVector(k.perChannelsConfigs, v.perChannelsConfigs));
                    mergeQKV.zeros = AppendVector(q.zeros, AppendVector(k.zeros, v.zeros));
                    mergeQKV.scales = AppendVector(q.scales, AppendVector(k.scales, v.scales));
                    mergeQKV.mins = AppendVector(q.mins, AppendVector(k.mins, v.mins));
                    //清除原来的Q，K，V以及他们的bias
                    weight.weight.erase(qWeightName);
                    weight.weight.erase(kWeightName);
                    weight.weight.erase(vWeightName);
                    weight.weight.erase(qBiasName);
                    weight.weight.erase(kBiasName);
                    weight.weight.erase(vBiasName);
                }
            }
            //处理完所有之后可以使用mergeQKV
            this->mergeQKV = canMerge;
        }
        //swiglu是一个llama常用的激活函数
        //如果能理解mergeQKV的做法，mergeSwiglu也一样
        //说到底就是为了减少矩阵乘法次数，因为所有算子里乘法是计算最耗费时间
        //swiglu公式大致是SwiGLU(x, W, V) = Swish(xW) \dot xV
        //x和W，V都需要做乘法计算
        //同理我们可以把W和V合并到一起减少计算次数
        if (!mergeSwiglu) {
            bool canMerge = true;
            for (int i = 0; i < block_cnt; i++) {
                //需要合并的变量名
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
                //合并后的变量名
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                //有些模型会直接提供
                if (weight.weight.find(swigluWeightName) != weight.weight.end()) {
                    mergeQKV = true;
                    break;
                }
                Data &w1 = weight.weight[w1WeightName], &w3 = weight.weight[w3WeightName];
                //同理，int4g量化如果做过padding就不可以合并
                if ((w1.dataType == DataType::INT4_GROUP && w1.dims[1] % w1.groupCnt != 0) || 
                    (w3.dataType == DataType::INT4_GROUP && w3.dims[1] % w3.groupCnt != 0)) {
                    canMerge = false;
                    break;
                }
                //合并后的维度[2h, h]
                weight.weight[swigluWeightName] = Data(w1.dataType, {w1.dims[0] + w3.dims[0], w1.dims[1]});
                Data &swiglu = weight.weight[swigluWeightName];
                swiglu.name = swigluWeightName;
                //分配，拷贝，释放原始空间
                swiglu.Allocate();
                memcpy(swiglu.cpuData, w1.cpuData, w1.GetBytes());
                memcpy(swiglu.cpuData + w1.GetBytes(), w3.cpuData, w3.GetBytes());
                    
                swiglu.perChannelAxis = w1.perChannelAxis;
                swiglu.group = w1.group;
                swiglu.groupCnt = w1.groupCnt;
                swiglu.perChannelsConfigs = AppendVector(w1.perChannelsConfigs, w3.perChannelsConfigs);
                swiglu.zeros = AppendVector(w1.zeros, w3.zeros);
                swiglu.scales = AppendVector(w1.scales, w3.scales);
                swiglu.mins = AppendVector(w1.mins, w3.mins);

                weight.weight.erase(w1WeightName);
                weight.weight.erase(w3WeightName);
            }
            //可以使用mergeSwiglu
            this->mergeSwiglu = canMerge;            
        }
        
        //qwen2没有用到linear bias的东西，没了解相关内容
        Data alibiData;
        if (this->weight.dicts["use_alibi"] == "1") {
            std::vector<float> alibi = GetInterleave(num_attention_heads);
            alibiData.CopyFrom(Data(DataType::FLOAT32, {(int) alibi.size()}, alibi));
        }

        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        //sinData是公用的，不是每个block分开
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        //Embedding层将输入的token序列找到对应的特征向量
        //InputIds        -> hiddenStates
        //[batch, seqLen] -> [batch, seqLen, h], h是特征维度
        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType); //basellm默认都是Float32，Embedding一般都是不做量化的

        int seqlen = hiddenStates.dims[1];
        //transformer开始, block_cnt就是block数量
        //其实这一块的代码理解非常简单，大部分的代码都是输入->输出，这个输出作为下一个算子的输入
        for (int i = 0; i < block_cnt; i++) {
            //分配device执行，实际上没有任何操作直接return了，这个应该是多卡部署才用到
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            //RMS做标准化，维度不变 
            //hiddenStates:[batch, seqLen, h] -> attenInput:[batch, seqLen, h]
            //hiddenStates将在最后的残差层再用于计算，目前只需要关注attenInput的操作
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            //得到QKV矩阵， bsz变量就是batch，seqLen变量就是seqLen, 这些变量不必关注，我会写出所有维度变化
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);
                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
            } else {
                //使用了mergeQkv，用合并的矩阵计算，然后通过Split分割
                //attenInput 变成了qkv
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    //attenInput * weight = qkv
                    //[batch, seqLen, h] * [mergeQKV, h]^T = [batch, seqLen, mergeQKV]
                    //Linear计算中的weight是转置的，这样才符合矩阵乘法计算的规则
                    Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                    //找出原来Q，K，V各自的大小
                    //qwen2引入了Group Attention Query导致Q，K，V的维度不一样
                    //简单来说就是减少了K，V维度，降低计算复杂度
                    //qwen2-7B，q有1536，k和v都是256，但是为了方便理解，我们还是统一用h表达维度
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);
                    //通过Split得到q,k,v
                    Split(qkv, -1, 0, qdim, q);                     //[batch, seqLen, h]
                    Split(qkv, -1, qdim, qdim + per, k);            //[batch, seqLen, h]
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);  //[batch, seqLen, h]
                } else {
                    Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                    Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                    Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                    Linear(attenInput, weight[qWeightName], qBias, q);
                    Linear(attenInput, weight[kWeightName], kBias, k);
                    Linear(attenInput, weight[vWeightName], vBias, v);
                }
            }

            //分成的多头维度，qwen2-7B每个头的维度是128
            std::vector <int> qkvSize = {bsz, seqlen, -1, head_dim};
            q.Reshape(qkvSize);     //[batch, seqLen, headCount, 128]
            k.Reshape(qkvSize);     //[batch, seqLen, headCount, 128]
            v.Reshape(qkvSize);     //[batch, seqLen, headCount, 128]

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(DataDevice::CUDA);
                pastValue.ToDevice(DataDevice::CUDA);
            }
            //位置信息编码
            //如果有历史信息就使用过去的序列信息，否则就使用当前的seqlen
            int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqlen : seqlen;
            //仅在第一个block执行，并且目标序列长度超过最大位置，并且使用动态NTK类型的RoPE才执行
            if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                //NTK Rope具体计算并且存入sinData中
                float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                float newbase = rope_base * scale;
                std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
            }

            //qwen2不使用linear bias, 使用RotatePosition进行位置编码
            if (alibiData.dims.size() == 0) {
                fastllm::LlamaRotatePosition2D(q, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            }


            //对换1，2的维度，因为之后的QK之类的计算要根据seqLen计算了 
            //[batch, seqLen, headCount, 128] -> [batch, headCount, seqLen, 128]
            PermuteSelf(q, {0, 2, 1, 3});   //[batch, headCount, seqLen, 128]
            PermuteSelf(k, {0, 2, 1, 3});   //[batch, headCount, seqLen, 128]
            PermuteSelf(v, {0, 2, 1, 3});   //[batch, headCount, seqLen, 128]

            //合并batch和headCount, 很难给这个计算结果取一个通俗的名字，就叫它N
            //至于为什么要合并，我个人觉得还是因为qwen2的group attention query模块
            //Q和KV维度并不一致，之后的计算可能会合并在一起会比较方便
            //[batch, headCount, seqLen, 128] -> [N, seqLen, 128]
            qkvSize = {-1, seqlen, head_dim};
            q.Reshape(qkvSize);     //[N, seqLen, 128]
            k.Reshape(qkvSize);     //[N, seqLen, 128]
            v.Reshape(qkvSize);     //[N, seqLen, 128]

            //KV Cache管理
            //pastKey和pastValue负责存储历史结果，目前着重于推理过程
            //KV Cache的东西会在之后小节介绍，
            //毕竟是存储历史数据，把pastKey看作K，pastValue看作V也无妨
            int unitLen = 64;
#ifdef USE_CUDA
            unitLen = 128;
#endif
            //KV Cache核心就是划一大块空间存储之前计算的K和V
            //代码中就是pastKeyh和pastValue
            //这一块空间大小由unitLen决定，在seqLen维度进行扩展
            //常见的Data维度是[batch, seqLen, h]
            //KV Cache在seqLenw维度扩展，预分配[batch, unitLen, h]

            //expansion属性就是给KV Cache准备的，
            //普通的Data类，就是数据占多少空间，Data就要多少空间
            //但是KV Cache是预先要分配一大块空间，所以就要靠expansion
            //pastKey.dims代表实际占用，pastKey.expansionDims代表分配大小

            //这里的while循环其实就是保证pastKey能存下所有数据
            //pastKey.dims[1]也就是pastKey的seqLen + k.dims[1]就是k的seqLen < pastKey的扩容维度
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || 
                    k.dims[1] > pastKey.expansionDims[1])) || 
                    (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                //要扩容pastKey
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    //没有分配过KV Cache的扩容
                    //可以看到这里主要修改的是第1维度
                    //找到能放下的unitLen倍数
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    //运行中的扩容
                    //在原来的基础上修改第1维度
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            //pastValue也一样
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                   || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }


            //将k矩阵cat到pastKey中
            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);
            //随着推理一直进行，不断有新的kv cat到pastKey中，这样就有了历史信息

            //得到Q，K，V后正式开始Attention计算
            //qwen2没有alibiData，直接用Attention算子
            //pastKey,pastValue是KV Cache中的内容，它包含所有历史问答的K，V值，但是逻辑上依旧和k相同[N, seqlen, 128]
            //attentionMask用于遮挡序列中看不到的信息
            //qkv就是输出结果
            //q.dims[0]/pastKey.dims[0]，qwen2用来计算group attention query的组数量
            //sqrt(head_dim)，注意力计算的系数】
            //1,Attention类型
            // if (alibiData.dims.size() == 0) {
            //     Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
            // } else
            {
                //Attention等价操作,我们为了表达简洁，忽略系数
                //Q * K^T = attenWeights
                //[N， seqLen, 128] * [N, seqLen, 128]^T = [N, seqLen, seqLen]
                //这个[seqLen, seqLen]就代表历史不同序列的注意力分数，attention score
                MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
                //增加一个维度，变成[1, N, seqLen, seqLen], 估计是算子需要4个维度？
                attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                if (alibiData.dims.size() != 0) {
                    attenWeights.Reshape({-1, num_attention_heads, attenWeights.dims[2], attenWeights.dims[3]});
                    AlibiMask(attenWeights, alibiData, -10000);
                    attenWeights.Reshape({1, -1, attenWeights.dims[2], attenWeights.dims[3]});
                } else if (attentionMask.dims.size() != 0) {
                    //遮挡后续信息，-10000的所用是在计算Softmax的时候e^x 会是一个很小的值，不会信息泄露
                    AttentionMask(attenWeights, attentionMask, -10000);
                }
                Softmax(attenWeights, attenWeights, -1); //Softmax, 将attention score计算成概率分布, 维度不变
                //softmaxRes * V = qkv
                //[1, N, seqLen, seqLen] * [N, seqLen, 128] = [1, N, seqLen, 128]
                MatMul(attenWeights, pastValue, qkv, 1.f, attenWeights.dims[1] / pastValue.dims[0]); //将结果乘以V得到输出
                //qkv删除之前加的维度，便会[N, seqLen, 128]
                qkv.Reshape({qkv.dims[1], qkv.dims[2], qkv.dims[3]});
            }
            
            //交换回0，1维度，[seqLen, N, 128]
            PermuteSelf(qkv, {1, 0, 2});
            //这两步其实我一直没理解在干吗
            //我们之前一直是[batch, seqLen, h]这样
            //它先转成[seqLen, batch, h] 再交换seqLen和batch, 有点多此一举
            //直接qkv.Reshape({bsz, seqlen ,-1})一步到位，我测试下来也ok
            //至此qkv：[batch, seqLen, h], 这个h就相当于是多头注意力的结果concate在一起
            qkv.Reshape({seqlen, bsz, -1});
            PermuteSelf(qkv, {1, 0, 2});
            //多头注意力的W_O乘法qkv
            //qkv * W_O = attenInput
            //[batch, seqLen, h] * [h, h] = [batch. seqLen, h]
            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(qkv, weight[oWeightName], oBias, attenInput);
            //在最一开始的hiddenStates在这里参与了残差层的相加
            //所有之前的变量都已经在也用不到，我们就只会用到这个hiddenStates
            //attenInput + hiddenStates = hiddenStates
            //[batch, seqLen, h] + [batch, seqLen, h] = [batch, seqLen ,h]
            AddTo(hiddenStates, attenInput);
            //Attention计算结束，mlp做激活
            //更新后的hiddenStates做RMS标准化，维度不变
            //hiddenStates -> attenInput
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            //我们使用了mergeSwiglu减少Linear次数
            if (this->mergeSwiglu) {
                //我们之前起的名字
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                // if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                //     LinearEx(attenInput, weight[swigluWeightName], Data(), q, LinearExType::ExSwiglu);
                // } 
                // else 
                {
                    //这里输出v，完全是变量复用，不用创建新的变量消耗空间，并没有特定含义
                    //v这里没有拆分，因为Swiglu算子支持直接计算合并矩阵
                    //attenInput * W = v
                    //[batch, seqLen, h] * [mergeSwiglu, h]^T = [batch, seqLen, mergeSwiglu]
                    Linear(attenInput, weight[swigluWeightName], Data(), v);
                    //Swiglu计算, 维度不变, q还是变量复用
                    //q : [batch, seqLen, mergeSwiglu]
                    Swiglu(v, q);
                }
            } else {
                if (CanRunLinearEx(LinearExType::ExSilu)) {
                    LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), q, LinearExType::ExSilu);
                } else {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), q);
                    Silu(q, q);
                }
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), v);
                MulTo(q, v);
            }
            //q做mlp的最后一次映射，其实就是把mergeSwiglu的维度变回到h
            //q * W = k
            //[batch, seqLen, mergeSwiglu] * [h, mergeSwglu]^T = [batch, seqLen, h]
            Linear(q, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), k);
            //将结果更新到hiddenStates, hiddenStates保持[batch, seqLen, h]
            AddTo(hiddenStates, k);
        }
        //运行完所有的block
        Data logits, topk;
        Data tempHiddenStates;
        Data *lastHiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;
            RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
            ToDataType(logits, DataType::FLOAT32);
            if (generationConfig.output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    memcpy((float*)(*retLogits)[b]->data(), 
                        ((float*)logits.cpuData) + ((b + 1) * logits.dims[1] - 1) * size, 
                        size * logits.unitSize);
                }
            }

            if (generationConfig.IsSimpleGreedy()) {
                TopK(logits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    int base = b;
                    lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    int base = b * logits.dims[1] + logits.dims[1] - 1;
                    lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
                }
            }
        }
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;

        return lastRet;
    }

    std::vector <int> LlamaModel::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        int seqLen = inputIds.dims[1];
        Data alibiData;
        if (this->weight.dicts["use_alibi"] == "1") {
            std::vector<float> alibi = GetInterleave(num_attention_heads);
            alibiData.CopyFrom(Data(DataType::FLOAT32, {(int) alibi.size()}, alibi));
        }

        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, curAttenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        std::vector <Data> curContextLayer;
        curContextLayer.resize(batch);
        std::vector <Data> curKs, curVs, curQs;
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);
        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);
        Data allPositionIds;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            allPositionIds.CopyFrom(*(Data*)positionIds[0]);
            allPositionIds.Expansion({1, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(allPositionIds, *(Data*)positionIds[i], 1);
            }
        }

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[qkvWeightName], Data(), qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);
                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
            } else {
                if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);

                    Split(qkv, -1, 0, qdim, q);
                    Split(qkv, -1, qdim, qdim + per, k);
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);
                } else {
                    Data qBias = (weight.weight.find(qBiasName) != weight.weight.end()) ? weight[qBiasName] : Data();
                    Data kBias = (weight.weight.find(kBiasName) != weight.weight.end()) ? weight[kBiasName] : Data();
                    Data vBias = (weight.weight.find(vBiasName) != weight.weight.end()) ? weight[vBiasName] : Data();
                    Linear(attenInput, weight[qWeightName], qBias, q);
                    Linear(attenInput, weight[kWeightName], kBias, k);
                    Linear(attenInput, weight[vWeightName], vBias, v);
                }
            }

            q.Reshape({q.dims[0], q.dims[1], -1, head_dim});
            k.Reshape({k.dims[0], k.dims[1], -1, head_dim});
            v.Reshape({v.dims[0], v.dims[1], -1, head_dim});
            int cacheOuter = k.dims[2], cacheInner = k.dims[3];
            int targetSeqLength = 0;
            for (int b = 0; b < batch; b++) {
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    if (GetKVCacheInCPU()) {
                        pastKey.lockInCPU = true;
                        pastValue.lockInCPU = true;
                    } else {
                        pastKey.ToDevice(DataDevice::CUDA);
                        pastValue.ToDevice(DataDevice::CUDA);
                    }
                    targetSeqLength = std::max(targetSeqLength, (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b]);
            }

            if (targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                    float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                    float newbase = rope_base * scale;
                    std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                    sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                    cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
            }

            for (int b = 0; b < batch; b++) {
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                int curLen = seqLens[b];
                
                int unitLen = 64;
#ifdef USE_CUDA
                unitLen = 128;
#endif
                while ((pastKey.dims.size() == 0 &&
                        (pastKey.expansionDims.size() == 0 || curLen > pastKey.expansionDims[1]))
                       || (pastKey.dims.size() > 0 && pastKey.dims[1] + curLen > pastKey.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector<int> {cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 &&
                        (pastValue.expansionDims.size() == 0 || curLen > pastValue.expansionDims[1]))
                       || (pastValue.dims.size() > 0 && pastValue.dims[1] + curLen > pastValue.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }

            if (alibiData.dims.size() == 0) {
                fastllm::LlamaRotatePosition2D(q, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            }

            Data attenOutput = Data(this->dataType);
            int total = 0;

            if (false) {
                
            } else {
                if (all1 && batch > 1) {
                    q.Reshape({-1, q.dims[2], q.dims[3]});
                    k.Reshape({-1, k.dims[2], k.dims[3]});
                    v.Reshape({-1, v.dims[2], v.dims[3]});

                    std::vector <int> qdims = {q.dims[1], 1, q.dims[2]};
                    std::vector <uint64_t> qstrides = {(uint64_t)q.dims[2], (uint64_t)q.dims[2], 1};
                    std::vector <int> kdims = {k.dims[1], 1, k.dims[2]};
                    std::vector <uint64_t> kstrides = {(uint64_t)k.dims[2], (uint64_t)k.dims[2], 1};
                    std::vector <int> vdims = {v.dims[1], 1, v.dims[2]};
                    std::vector <uint64_t> vstrides = {(uint64_t)v.dims[2], (uint64_t)v.dims[2], 1};
                    for (int b = 0; b < batch; b++) {
                        curQs[b].dims = qdims;
                        curQs[b].strides = qstrides;
                        curQs[b].FakeFrom(q, b * q.strides[0] * q.unitSize);
                        curKs[b].dims = kdims;
                        curKs[b].strides = kstrides;
                        curKs[b].FakeFrom(k, b * k.strides[0] * k.unitSize);
                        curVs[b].dims = vdims;
                        curVs[b].strides = vstrides;
                        curVs[b].FakeFrom(v, b * v.strides[0] * v.unitSize);
                    }

                    total = batch;
                } else {
                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});

                    std::vector<int> qkvSize = {-1, seqlen, head_dim};
                    q.Reshape(qkvSize);
                    k.Reshape(qkvSize);
                    v.Reshape(qkvSize);

                    for (int b = 0; b < batch; b++) {
                        Split(k, 1, total, total + seqLens[b], curKs[b]);
                        Split(v, 1, total, total + seqLens[b], curVs[b]);
                        Split(q, 1, total, total + seqLens[b], curQs[b]);
                        total += seqLens[b];
                    }
                }

                for (int b = 0; b < batch; b++) {
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    pointersK[b] = (&curKs[b]);
                    pointersV[b] = (&curVs[b]);
                }
                CatDirectBatch(keys, pointersK, 1);
                CatDirectBatch(values, pointersV, 1);
            }

            if (alibiData.dims.size() == 0 && all1 && batch > 1) {
                attenOutput.ToDevice(q.dataDevice);
                attenOutput.Resize({1, batch, embed_dim});
                attenOutput.Allocate();
                for (int b = 0; b < batch; b++) {
                    qs[b] = (&curQs[b]);
                    keys[b] = (pastKeyValues[b * block_cnt + i].first);
                    values[b] = (pastKeyValues[b * block_cnt + i].second);
                    masks[b] = attentionMask[b];
                    curContextLayer[b].FakeFrom(attenOutput, b * embed_dim * attenOutput.unitSize);
                    contexts[b] = (&curContextLayer[b]);
                }
                AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], 1.0 / scale_attn, 1);
            } else {
                if (alibiData.dims.size() == 0) {
                    attenOutput.ToDevice(curQs[0].dataDevice);
                    attenOutput.Resize({1, total, embed_dim});
                    attenOutput.Allocate();
                    int curLen = 0;
                    for (int b = 0; b < batch; b++) {
                        auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                        Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                        curAttenOutput.FakeFrom(attenOutput, curLen * embed_dim * attenOutput.unitSize);
                        curLen += seqLens[b];

                        // 1.2 Attention
                        if (attentionMask[b] == nullptr) {
                            Attention(q, pastKey, pastValue, Data(), curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                        } else {
                            Attention(q, pastKey, pastValue, *attentionMask[b], curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                        }
                        PermuteSelf(curAttenOutput, {1, 0, 2});
                    }
                } else {
                    for (int b = 0; b < batch; b++) {
                        auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                        Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;

                        // 1.2 Attention
                        // 1.2.0 q * k^T
                        if (alibiData.dims.size() == 0) {
                            if (attentionMask[b] == nullptr) {
                                Attention(q, pastKey, pastValue, Data(), curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                            } else {
                                Attention(q, pastKey, pastValue, *attentionMask[b], curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                            }
                        } else {
                            MatMulTransB(q, pastKey, attenWeights, 1.0 / sqrt(head_dim), q.dims[0] / pastKey.dims[0]);
                            attenWeights.Reshape({1, attenWeights.dims[0], attenWeights.dims[1], attenWeights.dims[2]});
                            if (alibiData.dims.size() != 0) {
                                AlibiMask(attenWeights, alibiData, -10000);
                            } else if (attentionMask[b] != nullptr) {
                                AttentionMask(attenWeights, *attentionMask[b], -10000);
                            }

                            Softmax(attenWeights, attenWeights, -1);
                            MatMul(attenWeights, pastValue, curAttenOutput, 1.f, attenWeights.dims[1] / pastValue.dims[0]);
                            curAttenOutput.Reshape({curAttenOutput.dims[1], curAttenOutput.dims[2], curAttenOutput.dims[3]});
                        }

                        PermuteSelf(curAttenOutput, {1, 0, 2});
                        curAttenOutput.Reshape({bsz, seqLens[b], -1});                    
                        if (attenOutput.dims.size() == 0) {
                            std::vector <int> dims = curAttenOutput.dims;
                            dims[1] = total;
                            attenOutput.Expansion(dims);
                            attenOutput.ToDevice(q.dataDevice);
                        }
                        CatDirect(attenOutput, curAttenOutput, 1);
                    }
                }
            }

            Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
            Linear(attenOutput, weight[oWeightName], oBias, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            if (this->mergeSwiglu) {
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                    LinearEx(attenInput, weight[swigluWeightName], Data(), w1, LinearExType::ExSwiglu);
                } else {
                    Linear(attenInput, weight[swigluWeightName], Data(), w3);
                    Swiglu(w3, w1);
                }
            } else {
                if (CanRunLinearEx(LinearExType::ExSilu)) {
                    LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1, LinearExType::ExSilu);
                } else {
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                }
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);
            }

            Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], Data(), w2);
            AddTo(hiddenStates, w2);
        }

        Data logits;
        std::vector <Data> curLogits;
        curLogits.resize(batch);

        if (batch > 1 && !all1) {
            int total = 0;
            std::vector <Data> lastTokens;
            std::vector <Data*> lastTokenPointers;
            lastTokens.resize(seqLens.size());
            for (int b = 0; b < seqLens.size(); b++) {
                Split(hiddenStates, 1, total + seqLens[b] - 1, total + seqLens[b], lastTokens[b]);
                total += seqLens[b];
                lastTokenPointers.push_back(&lastTokens[b]);
            }
            CatBatch(lastTokenPointers, 1, hiddenStates);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
        Linear(hiddenStates, weight["lm_head.weight"], Data(), logits);
        ToDataType(logits, DataType::FLOAT32);
        std::vector <int> lastRet;
        int total = 0;

        bool allSimple = true, needLogits = false;
        int maxTopK = 1;
        for (int b = 0; b < batch; b++) {
            if (!generationConfigs[b].IsSimpleGreedy()) {
                allSimple = false;
                break;
            }
        }
        for (int b = 0; b < batch; b++) {
            needLogits |= generationConfigs[b].output_logits;
            maxTopK = std::max(maxTopK, generationConfigs[b].top_k);
        }

        if (batch > 1 && allSimple) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            float *topkData = (float*)topk.cpuData;
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (topkData[0] + 1e-3));
                topkData += topk.Count(2);
            }
        } else if (batch > 1 && maxTopK <= 50 && !needLogits) {
            int maxTokenSetSize = 0;
            for (int b = 0; b < batch; b++) {
                maxTokenSetSize = std::max(maxTokenSetSize, (int)lastTokens.units[b].tokenSet.size());
            }
            std::vector <float> penaltyData = std::vector <float> (batch * maxTokenSetSize, -100.0f);
            std::vector <float> penaltyScaleData = std::vector <float> (batch, 1.0f);
            for (int b = 0; b < batch; b++) {
                int curId = 0;
                for (int i : lastTokens.units[b].tokenSet) {
                    penaltyData[b * maxTokenSetSize + curId] = i;
                    curId++;
                }
                penaltyScaleData[b] = generationConfigs[b].repeat_penalty;
            }
            Data penalty, penaltyScale;
            penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
            penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
            RepeatPenalty(logits, penalty, penaltyScale);
            Data topk;
            TopK(logits, topk, maxTopK);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                lastRet.push_back(LLMSamplingOnly(topk, b, generationConfigs[b]));
            }
        } else {
            for (int b = 0; b < batch; b++) {
                pointersK[b] = (&curLogits[b]);
            }
            SplitBatch(logits, 1, batch, pointersK);

            for (int b = 0; b < batch; b++) {
                Data &curLogit = curLogits[b];
                if (generationConfigs[b].output_logits && retLogits != nullptr && (*retLogits)[b] != nullptr) {
                    curLogit.ToDevice(DataDevice::CPU);
                    (*retLogits)[b]->resize(curLogit.Count(0));
                    memcpy((float*)(*retLogits)[b]->data(), (float*)curLogit.cpuData, curLogit.GetBytes());
                }
                if (generationConfigs[b].IsSimpleGreedy()) {
                    Data topk;
                    TopK(curLogit, topk, 1);
                    topk.ToDevice(DataDevice::CPU);
                    lastRet.push_back((int) (((float *) topk.cpuData)[0] + 1e-3));
                } else {
                    lastRet.push_back(LLMSampling(curLogit, 0, generationConfigs[b], lastTokens.units[b]));
                }
            }
        }
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;
        return lastRet;
    }

    bool LlamaModel::NeedAttentionMask(int qlen, int klen) {
        if (this->weight.dicts["use_alibi"] != "1" && 
            ((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void LlamaModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
                    vpids[i * maxLen + base + j] = j;
                }

                std::fill(vmask.data() + i * maxLen * maxLen,
                        vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                            vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    std::string LlamaModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string LlamaModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    //llama.cpp
    void LlamaModel::WarmUp() {
        printf("Warmup...\n");
        //inputIds就是1，所有参数都可以硬编码写
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});
        //初始化KV Cache, WarmUp结束也就没了
        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            this->weight["lm_head.weight"] = Data();
            this->weight["lm_head.weight"].CopyFrom(this->weight["model.embed_tokens.weight"]);
        }
        //做一次推理预热显卡
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
    }
}
