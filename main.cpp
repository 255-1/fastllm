#include "model.h"
#ifdef _WIN32
#include <stdlib.h>
#endif

std::map <std::string, fastllm::DataType> dataTypeDict = {
    {"float32", fastllm::DataType::FLOAT32},
    {"half", fastllm::DataType::FLOAT16},
    {"float16", fastllm::DataType::FLOAT16},
    {"int8", fastllm::DataType::INT8},
    {"int4", fastllm::DataType::INT4_NOZERO},
    {"int4z", fastllm::DataType::INT4},
    {"int4g", fastllm::DataType::INT4_GROUP}
};

struct RunConfig {
    std::string path = "chatglm-6b-int4.bin"; // 模型文件路径
    std::string systemPrompt = "";
    std::set <std::string> eosToken;
    int threads = 4; // 使用的线程数
    bool lowMemMode = false; // 是否使用低内存模式

    fastllm::DataType dtype = fastllm::DataType::FLOAT16;
    fastllm::DataType atype = fastllm::DataType::FLOAT32;
    int groupCnt = -1;
};

//模型运行参数，平时主要用到的是-p,top_p,top_k,--temperature和--repeat_penalty
//其他的比如dtype, atype之类的都不是离线flm需要用的
void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-t|--threads> <args>:        使用的线程数量" << std::endl;
    std::cout << "<-l|--low>:                   使用低内存模式" << std::endl;
    std::cout << "<--system> <args>:            设置系统提示词(system prompt)" << std::endl;
    std::cout << "<--eos_token> <args>:         设置eos token" << std::endl;
    std::cout << "<--dtype> <args>:             设置权重类型(读取hf文件时生效)" << std::endl;
    std::cout << "<--atype> <args>:             设置推理使用的数据类型(float32/float16)" << std::endl;
    std::cout << "<--top_p> <args>:             采样参数top_p" << std::endl;
    std::cout << "<--top_k> <args>:             采样参数top_k" << std::endl;
    std::cout << "<--temperature> <args>:       采样参数温度，越高结果越不固定" << std::endl;
    std::cout << "<--repeat_penalty> <args>:    采样参数重复惩罚" << std::endl;
}

void ParseArgs(int argc, char **argv, RunConfig &config, fastllm::GenerationConfig &generationConfig) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.path = sargv[++i];
        } else if (sargv[i] == "-t" || sargv[i] == "--threads") {
            config.threads = atoi(sargv[++i].c_str());
        } else if (sargv[i] == "-l" || sargv[i] == "--low") {
            config.lowMemMode = true;
        } else if (sargv[i] == "-m" || sargv[i] == "--model") {
            i++;
        } else if (sargv[i] == "--top_p") {
            generationConfig.top_p = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--top_k") {
            generationConfig.top_k = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--temperature") {
            generationConfig.temperature = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--repeat_penalty") {
            generationConfig.repeat_penalty = atof(sargv[++i].c_str());
        } else if (sargv[i] == "--system") {
            config.systemPrompt = sargv[++i];
        } else if (sargv[i] == "--eos_token") {
            config.eosToken.insert(sargv[++i]);
        } else if (sargv[i] == "--dtype") {
            std::string dtypeStr = sargv[++i];
            if (dtypeStr.size() > 5 && dtypeStr.substr(0, 5) == "int4g") {
                config.groupCnt = atoi(dtypeStr.substr(5).c_str());
                dtypeStr = dtypeStr.substr(0, 5);
            }
            fastllm::AssertInFastLLM(dataTypeDict.find(dtypeStr) != dataTypeDict.end(),
                                    "Unsupport data type: " + dtypeStr);
            config.dtype = dataTypeDict[dtypeStr];
        } else if (sargv[i] == "--atype") {
            std::string atypeStr = sargv[++i];
            fastllm::AssertInFastLLM(dataTypeDict.find(atypeStr) != dataTypeDict.end(),
                                    "Unsupport act type: " + atypeStr);
            config.atype = dataTypeDict[atypeStr];
        } else {
            Usage();
            exit(-1);
        }
    }
}

int main(int argc, char **argv) {
    //解析运行参数
    RunConfig config;
    fastllm::GenerationConfig generationConfig;
    ParseArgs(argc, argv, config, generationConfig);
    //打印PC支持哪些SIMD指令
    fastllm::PrintInstructionInfo();
    //设定线程状态，在一些情况下会使用多线程
    //使用cpu的算子会将矩阵进行切分，分配到不同的线程计算
    //在线量化模型
    fastllm::SetThreads(config.threads);
    //低内存状态，使用mmap共享代替直接分配
    fastllm::SetLowMemMode(config.lowMemMode);
    if (!fastllm::FileExists(config.path)) {
        printf(u8"模型文件 %s 不存在！\n", config.path.c_str());
        exit(0);
    }
    //支持两种读取模型，一种是读取转换好的flm离线模型。另一种是读取huggingface文件夹，目前只支持safetensor格式
    //从huggingface读取流程一样，它会用到多线程处理数据，相比离线方法能提供更多的转换种类如下，但是不会保存模型
    // enum DataType {
    // FLOAT32 = 0, BFLOAT16 = 1, INT16 = 2, INT8 = 3, INT4 = 4, INT2 = 5, BIT = 6, FLOAT16 = 7,
    // INT4_NOZERO = 8, INT4_GROUP = 9, INT32PARAM = 100,
    // DATA_AUTO_NONE = 99999, DATA_AUTO_LINEAR, DATA_AUTO_EMBEDDING, DATA_AUTO_CONV
    // };
    //我们主要讲解从已经转换好的qwen2 int4g的flm模型读取的过程，即CreateLLMModelFromFile这个函数，流程都大差不差
    bool isHFDir = fastllm::FileExists(config.path + "/config.json") || fastllm::FileExists(config.path + "config.json");
    auto model = !isHFDir ? fastllm::CreateLLMModelFromFile(config.path) : fastllm::CreateLLMModelFromHF(config.path, config.dtype, config.groupCnt);
    //下面这些设置基本上都是为了huggingface文件夹读取做的，atype，eos_token，prompt在之前的torch2file函数中都完成了
    if (config.atype != fastllm::DataType::FLOAT32) {
        model->SetDataType(config.atype);
    }
    model->SetSaveHistoryChat(true);
    //eosToken的设置是离线做不到的，torch2file中识别出模型后是硬编码，并且不支持多个eos_token_id
    //qwen系列的generation_config.json中的eos_token_id有两个，但是都是硬编码成第一个，而在线转换中可以插入多个eosToken
    for (auto &it : config.eosToken) {
        generationConfig.stop_token_ids.insert(model->weight.tokenizer.GetTokenId(it));
    }
    std::string systemConfig = config.systemPrompt;
    fastllm::ChatMessages messages = {{"system", systemConfig}};
    //获取模型类型，并开始和用户交互
    static std::string modelType = model->model_type;
    printf(u8"欢迎使用 %s 模型. 输入内容对话，reset清空历史记录，stop退出程序.\n", model->model_type.c_str());
    //读取用户输入，然后使用model->Response函数进行回答，直到输入stop结束交互
    while (true) {
        printf(u8"用户: ");
        std::string input;
        std::getline(std::cin, input);
        if (input == "reset") {
            messages = {{"system", config.systemPrompt}};
            continue;
        }
        if (input == "stop") {
            break;
        }
        messages.push_back(std::make_pair("user", input));
        //回复
        std::string ret = model->Response(model->ApplyChatTemplate(messages), [](int index, const char* content) {
            if (index == 0) {
                printf("%s:%s", modelType.c_str(), content);
                fflush(stdout);
            }
            if (index > 0) {
                printf("%s", content);
                fflush(stdout);
            }
            if (index == -1) {
                printf("\n");
            }
        }, generationConfig);
        messages.push_back(std::make_pair("assistant", ret));
    }

    return 0;
}