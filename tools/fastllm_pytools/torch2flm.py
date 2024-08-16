import struct
import builtins, os, json
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers.decoders import ByteLevel

#将量化后的数据导出到文件中
def writeString(fo, s):
    bytes = s.encode()
    #'i'代表以int方式写入
    fo.write(struct.pack('i', len(bytes)))
    fo.write(bytes)

#写入键值对，用于模型信息导出
def writeKeyValue(fo, key, value):
    writeString(fo, key)
    writeString(fo, value)

#fastllm支持的量化类型
fastllm_data_type_dict = {
    "int4g": 9,
    "int4": 8,
    "int8": 3,
    "float16": 7,
    "float32": 0,
}

#fastllm的权重类型，主要分为linear和embedding两大类，
#QuantizedLienar现在没用到，应该以后会添加支持
fastllm_weight_type_dict = {
    "linear": 1,
    "embedding": 2,
    "QuantizedLinear": 111
}

#这个可能是作者用来测试的例子
v = np.random.randint(-127, 127, [10, 20]);
temp = v;
c_max = np.expand_dims(np.abs(v).max(axis = -1), -1)
c_scale = c_max / 127.0
v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)

#将数据v以int8量化(1 Byte)类型导出到文件fo中，假设v的维度为[h, h]
def write_int8(fo, v):
    #找到数据中的最大绝对值，并且保证值在(0.1, 1e100)之间, cmax:[h, 1]
    c_max = np.expand_dims(np.abs(v).max(axis = -1), -1).clip(0.1, 1e100)
    #得到映射比例, c_scale:[h, 1]
    c_scale = c_max / 127.0
    #将v映射到（1，255）之间并且看作uint8，v:[h, h]
    v = (v / c_scale + 128.5).clip(1, 255).astype(np.uint8)
    #元数据，在读取时知道是int8量化
    fo.write(struct.pack('i', 3))
    fo.write(struct.pack('i', 0))
    #将cmax信息逐行以float形式(4 Bytes)写入文件，-c_max看作最小值，这种可能是作者认为有效的量化方式
    #int8并没有记录zero的值，这算是一种对称量化
    for i in range(c_max.shape[0]):
        fo.write(struct.pack('f', -c_max[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    #写入量化后的数据v
    fo.write(v.data)

#将数据v以int4g量化(4 bits)类型导出到文件fo中，假设v的维度为[h, h]
def write_int4g(fo, v, groupCnt = -1):
    # 默认每128个元素为一组
    if (groupCnt == -1):
        groupCnt = 128;
    #k是行维度，m是列维度
    k = v.shape[0]
    m = v.shape[1]
    #将列维度分组成group个组，每个组包含groupCnt（128）个元素
    group = (m - 1) // groupCnt + 1
    #看groupCnt能否整除h大小维度，如果不能整除就要pad补零
    pad = group * groupCnt - m
    if (pad > 0):
        v = np.concatenate((v, np.zeros([k, pad])), axis = 1)
    #数据v resize大小，每128列，v在不需要pad情况下的size:[h*h/128,128]
    v.resize(k * group, groupCnt)
    #每一行128个最小值和最大值获取，这里不像int8算绝对值了
    c_min = np.expand_dims(v.min(axis = -1), -1) #[h*h/128, 1]
    c_max = np.expand_dims(v.max(axis = -1), -1) #[h*h/128, 1]
    c_scale = (c_max - c_min) / 15.0             #[h*h/128, 1]
    #计算量化后的0的位置，并且确保0的值在（0，15）之间
    #fastllm使用的int4g会重新计算zero值，算是一种非对称量化
    c_zero = np.round(0.0 - c_min / c_scale)     #[h*h/128, 1]
    c_zero = c_zero.clip(0, 15)                  #[h*h/128, 1]
    #根据0的位置重新计算c_min的大小
    c_min = -c_scale * c_zero                    #[h*h/128, 1]   
    #对数据v进行量化到4bit大小，即(0, 15)之间
    v = (v - c_min) / c_scale
    # v+0.5为了让0.5-1的值能成为1，让0变少一点
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8) #[h*h/128,128]
    #如果pad过，则恢复v的为[h, group * groupCount]，并且把pad的0给去除了
    if (pad > 0):
        v.resize(k, group * groupCnt)
        v = v[:, :-pad].copy(order = 'C')
    #由于是4bit，可以通过 *16或<<4来把两个4bit组合成一个字节保存
    v = v[:, 0::2] * 16 + v[:, 1::2]
    #写入量化类型为int4g，并且写入分组数量和每组的个数
    fo.write(struct.pack('i', 9))
    fo.write(struct.pack('i', 0))
    fo.write(struct.pack('i', group))
    fo.write(struct.pack('i', groupCnt))
    #写入用于恢复的c_min和c_max数据
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    #写入量化后的数据v
    fo.write(v)

#int4量化就是int4g去除group相关的操作, 设v维度:[h,h]
def write_int4(fo, v):
    #找最小值和最大值，并且重新定义0的位置
    c_min = np.expand_dims(v.min(axis = -1), -1)    #[h, 1]
    c_max = np.expand_dims(v.max(axis = -1), -1)    #[h, 1]
    c_scale = (c_max - c_min) / 15.0                #[h, 1]
    c_zero = np.round(0.0 - c_min / c_scale)        #[h, 1]
    c_zero = c_zero.clip(0, 15)                     #[h, 1]
    c_min = -c_scale * c_zero
    #量化数据v，并且把两个4bit的组合在一起
    v = (v - c_min) / c_scale
    v = (v + 0.5).astype(np.int8).clip(0, 15).astype(np.uint8)
    v = v[:, 0::2] * 16 + v[:, 1::2]
    #量化类型为8，int4
    fo.write(struct.pack('i', 8))
    fo.write(struct.pack('i', 0))
    #写入数据
    for i in range(c_min.shape[0]):
        fo.write(struct.pack('f', c_min[i][0]));
        fo.write(struct.pack('f', c_max[i][0]));
    fo.write(v.data)

#导出量化模型的最主要的函数
def tofile(exportPath,
           model,
           tokenizer = None,
           pre_prompt = None,
           user_role = None,
           bot_role = None,
           history_sep = None,
           eos_id = None,
           dtype = "float16"):
    int4g_groupcnt = -1
    if (dtype.startswith("int4g") and len(dtype) > 5):
        try:
            int4g_groupcnt = int(dtype[5:])
            dtype = "int4g";
        except:
            print("dtype should be like \"int4g256\"")
            exit(0)
    if (dtype not in fastllm_data_type_dict):
        print("dtype should be one of ", list(fastllm_data_type_dict.keys()))
        exit(0)

    # 0.1 model info
    modelInfo = model.config.__dict__
    #如果有generation_config就更新配置
    if model.generation_config is not None:
        modelInfo.update(model.generation_config.__dict__)
    if ("model_type" not in modelInfo):
        print("unknown model_type.")
        exit(0)

    fo = open(exportPath, "wb")

    # 0. version id
    #用于区分fastllm的开发版本
    fo.write(struct.pack('i', 2))
    #将函数参数填入到模型信息中
    if (pre_prompt is not None):
        modelInfo["pre_prompt"] = pre_prompt
    if (user_role is not None):
        modelInfo["user_role"] = user_role
    if (bot_role is not None):
        modelInfo["bot_role"] = bot_role
    if (history_sep):
        modelInfo["history_sep"] = history_sep
    if (modelInfo["model_type"] == "baichuan"):
        if (hasattr(model, "model") and hasattr(model.model, "get_alibi_mask")):
            # Baichuan / Baichuan2 13B
            modelInfo["use_alibi"] = "1"
        modelInfo["pre_prompt"] = ""
        if (modelInfo["vocab_size"] == 125696):
            # Baichuan 2代
            modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + ">") if hasattr(model.generation_config, "user_token_id") else "";
        else:
            # Baichuan-13B-chat
            modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.user_token_id) + "> ") if hasattr(model.generation_config, "user_token_id") else "";
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(model.generation_config.assistant_token_id) + ">") if hasattr(model.generation_config, "assistant_token_id") else "";
        modelInfo["history_sep"] = ""
    #如果模型是qwen一代，做一些设置
    if (modelInfo["model_type"] == "qwen"):
        if modelInfo["chat_format"] == "chatml":
            modelInfo["im_end_id"] = tokenizer.im_end_id
            modelInfo["im_start_id"] = tokenizer.im_start_id
    elif (modelInfo["model_type"] == "qwen2"):
        #模型是qwen2则硬编码eos_token_id
        modelInfo["eos_token_id"] = "151645"
    elif (modelInfo["model_type"] == "internlm"):
        modelInfo["eos_token_id"] = "103028"
        if "rotary" in modelInfo:
            rope_scaling = modelInfo.pop("rotary")
            if isinstance(rope_scaling, builtins.dict):
                modelInfo["rope_scaling.type"] = rope_scaling["type"]
                modelInfo["rope_theta"] = rope_scaling["base"]
    elif (modelInfo["model_type"] == "internlm2"):
        modelInfo["eos_token_id"] = "92542"
    if (modelInfo["model_type"] == "chatglm" and hasattr(tokenizer, "build_chat_input")):
        # chatglm3
        modelInfo["pre_prompt"] = "";
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.get_command("<|user|>")) + "> \n");
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.get_command("<|assistant|>")) + ">");
        modelInfo["history_sep"] = "";
    if (modelInfo["model_type"] == "chatglm" and hasattr(tokenizer, "name") and tokenizer.name == "GLM4Tokenizer"):
        # glm-4-chat
        modelInfo["pre_prompt"] = "";
        modelInfo["user_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.convert_tokens_to_ids("<|user|>")) + ">\n");
        modelInfo["bot_role"] = ("<FLM_FIX_TOKEN_" + str(tokenizer.convert_tokens_to_ids("<|assistant|>")) + ">");
        modelInfo["history_sep"] = "";
        modelInfo["tokenizer_class"] = tokenizer.name;
    if "rope_scaling" in modelInfo and isinstance(modelInfo["rope_scaling"], builtins.dict):
        rope_scaling = modelInfo.pop("rope_scaling")
        modelInfo["rope_scaling.type"] = rope_scaling["type"]
        modelInfo["rope_scaling.factor"] = rope_scaling["factor"]
    if eos_id:
        modelInfo["eos_token_id"] = str(eos_id)

    #qwen2有一个merges.txt表示会使用ReBERTA这种byte level的位置编码，会存放在merge字典中
    merges = {}
    #如果有tokenizer，不同模型基本都会有自己的分词器
    if tokenizer:
        modelInfo["tokenizer_use_score"] = "1" # 分词带分数
        #是否有特殊tokens，qwen2有<|im_start|>， <|im_end|>之类的,设置modelInfo有特殊token
        if len(tokenizer.all_special_tokens) > 0:
            #token_set有点迷，感觉作用不大
            token_set = set()
            #对比token是不是出现在promt的中，代表我们将会用特殊tokens
            for token in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token, tokenizer.pad_token]:
                for prompt in [pre_prompt, user_role, bot_role, history_sep]:
                    if prompt and str(token) in prompt:
                        modelInfo["tokenizer_has_special_tokens"] = "1"
                token_set.add(str(token))
            if len(tokenizer.all_special_tokens) > len(token_set):
                modelInfo["tokenizer_has_special_tokens"] = "1"
        #qwen2不用sp_model
        #判断tokenizer是否是sp_model，会用sentencepiece进行分词
        if hasattr(tokenizer, "sp_model") or (hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "sp_model")):
            try:
                import sentencepiece.sentencepiece_model_pb2 as model_pb2
                with open(tokenizer.vocab_file, "rb") as f:
                    sp_model_data = f.read()
                    sp_model_proto = model_pb2.ModelProto.FromString(sp_model_data)
                    modelInfo["tokenizer_add_dummy_prefix"] = sp_model_proto.normalizer_spec.add_dummy_prefix
                    if sp_model_proto.normalizer_spec.remove_extra_whitespaces:
                        modelInfo["tokenizer_remove_extra_whitespaces"] = True
            except:
                pass
        elif isinstance(tokenizer, PreTrainedTokenizerFast):
            #qwen2实际使用PreTrainedTokenizer
            modelInfo["tokenizer_add_dummy_prefix"] = False
            #读取vocab文件
            tokenizer_file_name = tokenizer.vocab_file if (hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file) else tokenizer.vocab_files_names['tokenizer_file']
            tokenizer_file = os.path.join(tokenizer.name_or_path, tokenizer_file_name)
            if os.path.exists(tokenizer_file):
                with open(tokenizer_file, "r", encoding='utf-8') as f:
                    #使用json解析vocab文件
                    tokenizer_data = json.load(f)
                    #qwen2没有normalizer
                    #判断是否用normalizer做清理和处理
                    if "normalizer" in tokenizer_data and tokenizer_data["normalizer"] and "normalizers" in tokenizer_data["normalizer"]:
                        for normalizer in tokenizer_data["normalizer"]["normalizers"]:
                            if normalizer["type"] == "Prepend" and \
                                    (normalizer["prepend"] == '▁' or normalizer["prepend"] == ' '):
                                modelInfo["tokenizer_add_dummy_prefix"] = True
                    #qwen2有merges，之后会用merges计算位置分数
                    if "merges" in tokenizer_data["model"]:
                        bpe_merges = tokenizer_data["model"]["merges"]
                        #去除空格
                        bpe_merges = [pair.replace(" ", "") for pair in bpe_merges]
                        #给bpe_merges的元素生成对应id值，从-1开始一直减一
                        merges = builtins.dict(zip(bpe_merges, range(0, -len(bpe_merges), -1)))
            #判断tokenizer的decoder是不是ByteLevel的
            if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer._tokenizer, "decoder") \
                    and isinstance(tokenizer._tokenizer.decoder, ByteLevel):
                modelInfo["tokenizer_byte_as_char"] = True
            else:
            if hasattr(tokenizer, "byte_encoder") and hasattr(tokenizer, "byte_decoder"):
                modelInfo["tokenizer_byte_as_char"] = True
            if not hasattr(tokenizer, "add_prefix_space") or not getattr(tokenizer, "add_prefix_space", True):
                modelInfo["tokenizer_add_dummy_prefix"] = False

    #qwen2没有微调
    if hasattr(model, "peft_config"):
        adapter_size = len(model.peft_config)
        modelInfo["peft_size"] = adapter_size
    #将modelInfo现在的长度写入，在推理时可以知道modelInfo读取大小
    fo.write(struct.pack('i', len(modelInfo)))
    #写入modelInfo信息
    for it in sorted(modelInfo.keys()):
        writeKeyValue(fo, str(it), str(modelInfo[it]))

    if hasattr(model, "peft_config"):
        for adapter_name in model.peft_config.keys():
            adapter_dict = model.peft_config[adapter_name].__dict__
            writeString(fo, adapter_name)
            fo.write(struct.pack('i', len(adapter_dict)))
            for it in adapter_dict.keys():
                writeKeyValue(fo, str(it), str(adapter_dict[it]))

    #权重名称字典
    weight_type_dict = {}
    # 1. vocab
    if (tokenizer):
        if (hasattr(tokenizer, "tokenizer")):
            if (str(type(tokenizer.tokenizer)).find("Encoding") == -1):
                tokenizer = tokenizer.tokenizer
        if (hasattr(tokenizer, "sp_model")):
            piece_size = tokenizer.sp_model.piece_size()
            fo.write(struct.pack('i', piece_size))
            for i in range(piece_size):
                s = tokenizer.sp_model.id_to_piece(i).encode()
                fo.write(struct.pack('i', len(s)))
                for c in s:
                    fo.write(struct.pack('i', c))
                fo.write(struct.pack('i', i))
                fo.write(struct.pack('f', float(tokenizer.sp_model.get_score(i))))
        else:
            #处理merge信息，将之前merge字典组合成元组，并且按照id值排序
            if hasattr(tokenizer, "bpe_ranks"):
                merges = {("".join(bpe_tokens), token_index) for bpe_tokens, token_index in sorted(tokenizer.bpe_ranks.items(), key=lambda kv: kv[1])}
            #得到词表文件并写入大小，vocab是一个{单词：token_id}字典
            vocab = tokenizer.get_vocab()
            fo.write(struct.pack('i', len(vocab)))
            for v in vocab.keys():
                #根据merges信息给当前单词打分
                score = merges[v] if v in merges else 1.0
                if (isinstance(v, str)):
                    s = v.encode()
                else:
                    s = v
                #将当前单词大小写入flm中
                fo.write(struct.pack('i', len(s)))
                #将每个字符写入flm中
                for c in s:
                    fo.write(struct.pack('i', c))
                #将token id和score写入flm中
                fo.write(struct.pack('i', vocab[v]))
                fo.write(struct.pack('f', score))
            #如果有特殊token则写到flm文件中，基本所有llm模型都会有
        if ("tokenizer_has_special_tokens" in modelInfo):
            all_special_tokens = tokenizer.all_special_tokens
            if hasattr(tokenizer, "added_tokens_decoder"):
                for i in tokenizer.added_tokens_decoder:
                    all_special_tokens.append(str(tokenizer.added_tokens_decoder[i]))
            fo.write(struct.pack('i', len(all_special_tokens)))
            for special_token in all_special_tokens:
                writeString(fo, special_token)
    else:
        fo.write(struct.pack('i', 0))

    module_dict = {}
    for key, m in model.named_modules():
        if (isinstance(m, torch.nn.Linear)):
            weight_type_dict[key + ".weight"] = "linear"
            module_dict[key + ".weight"] = m
        if (isinstance(m, torch.nn.Embedding)):
            weight_type_dict[key + ".weight"] = "embedding"

    #读取模型权重， dict是一个{weightname, tensor}的字典
    model = model.cpu();
    dict = model.state_dict()
    #写入dict的长度
    fo.write(struct.pack('i', len(dict)))
    tot = 0
    for key in dict:
        #原始数据是float32
        ori_np_data_type = np.float32
        cur_weight_type = 0
        #判断fastllm能否支持量化
        if (key in weight_type_dict and weight_type_dict[key] in fastllm_weight_type_dict):
            cur_weight_type = fastllm_weight_type_dict[weight_type_dict[key]]
        
        #如果是linear数据，就按照dtype进行量化
        to_data_type = 0
        if (cur_weight_type == 1):
            to_data_type = fastllm_data_type_dict[dtype]
            if (to_data_type == 7):
                #如果要转float16，则不需要额外计算，直接转就行
                ori_np_data_type = np.float16
        #转成ori_np_data_type：float32或float16
        if (dict[key].dtype == torch.bfloat16):
            #bfloat16需要额外half()一下
            cur = dict[key].half().numpy().astype(ori_np_data_type)
        else:
            cur = dict[key].numpy().astype(ori_np_data_type)
        
        weight_name = key
        #写入权重名称
        writeString(fo, weight_name)
        #写入权重的shape占用字节数
        fo.write(struct.pack('i', len(cur.shape)))
        #写入具体的权重shape数字
        for i in cur.shape:
            fo.write(struct.pack('i', i))
        #按照不同的量化策略写入对应信息
        if (to_data_type == 3):
            write_int8(fo, cur)
        elif (to_data_type == 8):
            write_int4(fo, cur)
        elif (to_data_type == 9):
            write_int4g(fo, cur, groupCnt = int4g_groupcnt)
        else:
            fo.write(struct.pack('i', to_data_type))
            fo.write(cur.data)
        tot += 1
        print("output (", tot, "/", len(dict), end = " )\r")
    print("\nfinish.")
    fo.close()
