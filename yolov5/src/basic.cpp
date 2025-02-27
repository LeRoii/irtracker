#include "basic.h"

basic::basic(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight)
    :g_deviceId_(0), g_imageDataBuf_(nullptr)
{
    g_imageDataSize_ = RGBU8_IMAGE_SIZE(modelWidth, modelHeight);

    omModelPath = modelPath;
}

basic::~basic()
{
}


Result basic::Init_atlas()
{
    // ACL init
    const char *aclConfigPath = nullptr;
    aclError ret = aclInit(aclConfigPath);
    if (ret) {
        std::cout<<"acl get run mode failed"<<std::endl;
        return FAILED;
    }
    
    // open device
    ret = aclrtSetDevice(g_deviceId_);
    if (ret) {
        std::cout<<"Acl open device failed"<<std::endl;
        return FAILED;
    }

    // create context
    ret = aclrtCreateContext(&g_context_, g_deviceId_);
    if (ret) {
        std::cout<<"acl create context failed"<<std::endl;
        return FAILED;
    }

    // create stream
    ret = aclrtCreateStream(&g_stream_);
    if (ret) {
        std::cout<<"acl create stream failed"<<std::endl;
        return FAILED;
    }

    // Gets whether the current application is running on host or Device
    ret = aclrtGetRunMode(&g_runMode_);
    if (ret) {
        std::cout<<"acl get run mode failed"<<std::endl;
        return FAILED;
    }

    ret = aclmdlLoadFromFile(omModelPath, &g_modelId_);
    if(ret){
        std::cout<<"loadding model error, model name: " << omModelPath << std::endl;
        return FAILED;
    }

    g_modelDesc_ = aclmdlCreateDesc();
    if (g_modelDesc_ == nullptr) {
        std::cout<<"create model description failed"<<std::endl;
        return FAILED;
    }

    ret = aclmdlGetDesc(g_modelDesc_, g_modelId_);
    if (ret) {
        std::cout<<"get model description failed"<<std::endl;
        return FAILED;
    }

    return SUCCESS;
}

//创造输入数据
Result basic::CreateInput() {

    aclError aclRet = aclrtMalloc(&g_imageDataBuf_, g_imageDataSize_, ACL_MEM_MALLOC_HUGE_FIRST);

    g_input_ = aclmdlCreateDataset();
    if (g_input_ == nullptr) {
        std::cout<<"can't create dataset, create input failed"<<std::endl;
        return FAILED;
    }
        
    aclDataBuffer* inputData = aclCreateDataBuffer(g_imageDataBuf_, g_imageDataSize_);
    if (inputData == nullptr) {
        std::cout<<"can't create data buffer, create input failed"<<std::endl;
        return FAILED;
    }

    aclError ret = aclmdlAddDatasetBuffer(g_input_, inputData);
    if (inputData == nullptr) {
        std::cout<<"can't add data buffer, create input failed"<<std::endl;
        aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }

    return SUCCESS;
}

//创造输出数据
Result basic::CreateOutput() {

    if (g_modelDesc_ == nullptr) {
        std::cout<<"no model description, create ouput failed"<<std::endl;
        return FAILED;
    }


    g_output_ = aclmdlCreateDataset();
    if (g_output_ == nullptr) {
        std::cout<<"can't create dataset, create output failed"<<std::endl;
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(g_modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t buffer_size = aclmdlGetOutputSizeByIndex(g_modelDesc_, i);

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret) {
            std::cout<<"can't malloc buffer, create output failed"<<std::endl;
            return FAILED;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, buffer_size);
        if (ret) {
            std::cout<<"can't create data buffer, create output failed"<<std::endl;
            aclrtFree(outputBuffer);
            return FAILED;
        }

        ret = aclmdlAddDatasetBuffer(g_output_, outputData);
        if (ret) {
            std::cout<<"can't add data buffer, create output failed"<<std::endl;
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }

    return SUCCESS;
}


void* basic::GetInferenceOutputItem(uint32_t& itemDataSize,  uint32_t idx)
{
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_output_, idx);
    if (dataBuffer == nullptr) {
        std::cout<<"Get dataset buffer from model failed "<<std::endl;
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        std::cout<<"Model inference output failed" <<std::endl;
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        std::cout<<"Model inference output is 0" <<std::endl;
        return nullptr;
    }

    void* data = nullptr;
    if (g_runMode_ == ACL_HOST) {
        data = CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr) {
            std::cout<<"Copy inference output to host failed"<<std::endl;
            return nullptr;
        }
    } else {
        data = dataBufferDev;
    }

    itemDataSize = bufferSize;
    return data;
}

void* basic::CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize)
{
    // uint8_t* buffer = new uint8_t[dataSize];
    // if (buffer == nullptr) {
    //     std::cout<<"New malloc memory failed"<<std::endl;
    //     return nullptr;
    // }

    // aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    // if (aclRet) {
    //     std::cout<<"Copy device data to local failed" << std::endl;
    //     delete[](buffer);
    //     return nullptr;
    // }

    // return (void*)buffer;

    void* buffer = nullptr;
    aclError ret = aclrtMallocHost(&buffer, dataSize);
    if (ret != ACL_SUCCESS) {
	cout << "aclrtMallocHost failed, result code is " << ret << endl;
    }
    
    aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        std::cout<<"Copy device data to local failed"<<std::endl;;
        //delete[](buffer);
	    (void)aclrtFreeHost(buffer);
        return nullptr;
    }

    return (void*)buffer;
}

Result basic::inference() {
    aclError ret = aclmdlExecute(g_modelId_, g_input_, g_output_);
    if(ret) {
        std::cout<<"Model inference failed"<<std::endl;
        return FAILED;
    }

    return SUCCESS;
}

void basic::DestroyResource() {

    aclrtFree(g_imageDataBuf_);
    aclError ret = aclmdlUnload(g_modelId_);
    // if (CLOCK_REALTIME_ALARM) {
    //     std::cout<<"unload model failed"<<std::endl;
    // }

    DestroyDesc();
    DestroyInput();
    DestroyOutput();

    ret = aclrtResetDevice(g_deviceId_);
    if (ret) {
        std::cout<<"reset device failed"<<std::endl;
    }

    ret = aclFinalize();
    if (ret) {
        std::cout<<"finalize acl failed"<<std::endl;
    }
    std::cout<<"end to finalize acl"<<std::endl;
    
}   


void basic::DestroyDesc()
{
    if (g_modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(g_modelDesc_);
        g_modelDesc_ = nullptr;
    }
}

void basic::DestroyInput()
{
    if (g_input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(g_input_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_input_, i);
        aclDestroyDataBuffer(dataBuffer);
    }
    aclmdlDestroyDataset(g_input_);
    g_input_ = nullptr;
}

void basic::DestroyOutput()
{
    if (g_output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(g_output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(g_output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(g_output_);
    g_output_ = nullptr;
}

