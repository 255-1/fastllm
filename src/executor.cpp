//
// Created by huangyuyang on 6/13/23.
//

#include "utils.h"

#include "executor.h"

#include "devices/cpu/cpudevice.h"

#ifdef USE_CUDA
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/multicudadevice.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#ifdef USE_TFACC
#include "devices/tfacc/tfaccdevice.h"
#endif

namespace fastllm {
    Executor::Executor() {
        this->devices.clear();
#ifdef USE_CUDA
        this->devices.push_back((BaseDevice*) new CudaDevice());
        this->devices.push_back((BaseDevice*) new MultiCudaDevice());
#endif
#ifdef USE_TFACC
        this->devices.push_back((BaseDevice*) new TfaccDevice());
#endif
        this->devices.push_back((BaseDevice*) new CpuDevice());
    }

    Executor::~Executor() {
        for (int i = 0; i < devices.size(); i++) {
            delete devices[i];
        }
    }

    void Executor::ClearDevices() {
        this->devices.clear();
    }

    void Executor::AddDevice(fastllm::BaseDevice *device) {
        this->devices.push_back(device);
    }

    void Executor::SetFirstDevice(const std::string &device) {
        auto temp = this->devices;
        this->devices.clear();
        for (int i = 0; i < temp.size(); i++) {
            if (StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
                this->devices.back()->deviceIds = ParseDeviceIds(device, temp[i]->deviceType);
            }
        }
        for (int i = 0; i < temp.size(); i++) {
            if (!StartWith(device, temp[i]->deviceType)) {
                this->devices.push_back(temp[i]);
            }
        }
    }

    std::vector <int> Executor::GetDeviceIds(const std::string &device) {
        for (int i = 0; i < devices.size(); i++) {
            if (StartWith(devices[i]->deviceType, device)) {
                return devices[i]->deviceIds;
            }
        }
        return {0};
    }

    bool Executor::CanRunOnFirstDevice(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams) {     
        return this->devices[0]->CanRun(opType, datas, floatParams, intParams);
    }

    //执行器
    //opType：算子名字
    //datas：一般包含input, weight, output三个东西
    //floatParams: 额外的float类型变量
    //intParams: 额外的int类型变量
    //datas的类型DataDict为了不Copy Data数据是一个{string, Data*}类型
    void Executor::Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams) {
         //用于记录每个算子运行耗时
        auto st = std::chrono::system_clock::now();
        bool lockInCPU = false;
        //遍历datas参数，确保这些数据是否要锁在CPU
        for (auto &it: datas) {
            if (intParams.find(it.first + "___batch") != intParams.end()) {
                int batch = intParams.find(it.first + "___batch")->second;
                for (int i = 0; i < batch; i++) {
                    lockInCPU |= (((Data**)it.second)[i] && ((Data**)it.second)[i]->lockInCPU);
                }
            } else {
                lockInCPU |= (it.second && it.second->lockInCPU);
            }
        }
        //遍历可用设备，一般是1个CPU + 1个GPU
        for (auto device: devices) {
            //如果需要锁在CPU中，就跳过GPU
            if (lockInCPU && device->deviceType != "cpu") {
                continue;
            }
            //cpudevice.cpp, cudadevice.cpp分别负责cpu和cuda算子实现
            //每个算子都有三个函数，分别是
            //CanRun: 当前设备可不可以跑这个算子
            //Reshape: 用于判断输入的变量维度是否符合要求，并且给output Data分配空间
            //Run：具体的算子实现
            if (device->CanRun(opType, datas, floatParams, intParams)) {
#ifdef USE_CUDA
                if (device->deviceType == "cuda" && device->deviceIds.size() > 0) {
                    FastllmCudaSetDevice(device->deviceIds[0]);
                }
                if (device->deviceType == "multicuda" && device->deviceIds.size() > 0) {
                    FastllmMultiCudaSetDevice(device->deviceIds);
                }
#endif
                //如果当强device CanRun就把数据转移到对应的device上
                for (auto &it: datas) {
                    if (intParams.find(it.first + "___batch") != intParams.end()) {
                        int batch = intParams.find(it.first + "___batch")->second;
                        for (int i = 0; i < batch; i++) {
                            if (((Data**)it.second)[i]) {
                                ((Data**)it.second)[i]->ToDevice((void *) device);
                            }
                        }
                    } else {
                        if (it.second) {
                            it.second->ToDevice((void *) device);
                        }
                    }
                }
                //Reshape: 用于判断输入的变量维度是否符合要求，并且推导output型状
                device->Reshape(opType, datas, floatParams, intParams);
                //运行算子
                device->Run(opType, datas, floatParams, intParams);
                break;
            }
        }
        //记录算子的运行时间
        float spend = GetSpan(st, std::chrono::system_clock::now());
        profiler[opType] += spend;
    }

    void Executor::ClearProfiler() {
        profiler.clear();
    }

    void Executor::PrintProfiler() {
        float sum = 0.0;
        for (auto &it : profiler) {
            printf("%s spend %f\n", it.first.c_str(), it.second);
            sum += it.second;
        }
        printf("total spend %f\n", sum);
    }
}