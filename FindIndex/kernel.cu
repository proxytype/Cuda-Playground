
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thread>
#include <stdio.h>
#include <vector>
#include <Windows.h>
#include <Wbemidl.h>
#include <comutil.h>
#include <random>

#pragma comment(lib, "wbemuuid.lib")

const int MAX_ELEMENTS = 141000000;
const int MAX_CUDA_PER_BLOCK = 128;

const int CPU_MAX_THREADS = 4;

int* d_arr;

int numOfElements = MAX_ELEMENTS;
int numCPUThreads = CPU_MAX_THREADS;
int numCudaPerBlock = MAX_CUDA_PER_BLOCK;
int selectedIndex = -1;

int* createArray() {
    std::cout << "  Creating array with " << numOfElements << " elements." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    int* values = new int[numOfElements];

    for (int i = 0; i < numOfElements; i++)
    {
        values[i] = i;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "  Complete: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "__________________________________________________________" << std::endl;

    return values;
}

void cudaCopyArray(int* values) {

    std::cout << "  Copy array to Device...." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    cudaMalloc((void**)&d_arr, numOfElements * sizeof(int));
    cudaMemcpy(d_arr, values, numOfElements * sizeof(int), cudaMemcpyHostToDevice);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "  Complete time: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "__________________________________________________________" << std::endl;
}

void parallelSearch(int* arr, int size, int target, int& result, int threadId) {
    int start = threadId * (size / numCPUThreads);
    int end = (threadId + 1) * (size / numCPUThreads);

    for (int i = start; i < end; i++) {
        if (arr[i] == target) {
            result = i;
            return;
        }
    }
}

__global__ void findValueKernel(int* arr, int targetIndex, int maxElements, int* result) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (*result != -1) {
        return;  // If result has been found, no need to continue searching.
    }

    for (int i = threadId; i < maxElements; i += blockDim.x * gridDim.x) {
        if (arr[i] == targetIndex) {
            result[0] = arr[i];
        }
    }
}

void executeGPU(int randomIndex) {

    int result = -1;

    auto startTime = std::chrono::high_resolution_clock::now();

    int* d_result;
    cudaMalloc((void**)&d_result, sizeof(int));
    cudaMemcpy(d_result, &result, sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel with multiple threads and blocks
    int blocksPerGrid = (numOfElements + numCudaPerBlock - 1) / numCudaPerBlock;
    int threadsPerBlock = numCudaPerBlock;

    findValueKernel << <blocksPerGrid, threadsPerBlock >> > (d_arr, randomIndex, numOfElements, d_result);

    // Copy the result from the device to host
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_arr);
    cudaFree(d_result);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    if (result != -1) {
        std::cout << "  Value found at random index: " << result << std::endl;
    }

    std::cout << "  Complete time: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "__________________________________________________________" << std::endl;
}

void executeCPU(int* values, int randomIndex) {

    int targetValue = values[randomIndex];

    std::vector<std::thread> threads(numCPUThreads);
    std::vector<int> results(numCPUThreads, -1);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numCPUThreads; i++) {
        threads[i] = std::thread(parallelSearch, values, numOfElements, targetValue, std::ref(results[i]), i);
    }

    for (int i = 0; i < numCPUThreads; i++) {
        threads[i].join();
        if (results[i] != -1) {
            std::cout << "  Value found at index: " << results[i] << std::endl;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "  Complete time: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "__________________________________________________________" << std::endl;
}

void printHost() {
    HRESULT hres;

    // Initialize COM library
    hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hres)) {
        std::cerr << "  Failed to initialize COM library. Error code: 0x" << std::hex << hres << std::dec << std::endl;
        return;
    }

    // Initialize COM security
    hres = CoInitializeSecurity(
        nullptr, -1, nullptr, nullptr, RPC_C_AUTHN_LEVEL_DEFAULT,
        RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE, nullptr
    );
    if (FAILED(hres)) {
        std::cerr << "  Failed to initialize COM security. Error code: 0x" << std::hex << hres << std::dec << std::endl;
        CoUninitialize();
        return;
    }

    // Initialize WMI
    IWbemLocator* pLoc = nullptr;
    hres = CoCreateInstance(CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);
    if (FAILED(hres)) {
        std::cerr << "  Failed to create IWbemLocator. Error code: 0x" << std::hex << hres << std::dec << std::endl;
        CoUninitialize();
        return;
    }

    IWbemServices* pSvc = nullptr;
    hres = pLoc->ConnectServer(
        _bstr_t(L"ROOT\\CIMV2"), nullptr, nullptr, nullptr, 0, nullptr, nullptr, &pSvc
    );
    if (FAILED(hres)) {
        std::cerr << "  Failed to connect to WMI. Error code: 0x" << std::hex << hres << std::dec << std::endl;
        pLoc->Release();
        CoUninitialize();
        return;
    }

    // Set security levels on the proxy
    hres = CoSetProxyBlanket(
        pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr, RPC_C_AUTHN_LEVEL_CALL,
        RPC_C_IMP_LEVEL_IMPERSONATE, nullptr, EOAC_NONE
    );
    if (FAILED(hres)) {
        std::cerr << "  Failed to set security levels on WMI proxy. Error code: 0x" << std::hex << hres << std::dec << std::endl;
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return;
    }

    // Query CPU information
    IEnumWbemClassObject* pEnumerator = nullptr;
    hres = pSvc->ExecQuery(
        bstr_t("WQL"), bstr_t("SELECT * FROM Win32_Processor"), WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &pEnumerator
    );
    if (FAILED(hres)) {
        std::cerr << "  Failed to execute WMI query. Error code: 0x" << std::hex << hres << std::dec << std::endl;
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        return;
    }

    IWbemClassObject* pclsObj = nullptr;
    ULONG uReturn = 0;

    while (pEnumerator) {
        hres = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
        if (0 == uReturn) {
            break;
        }

        VARIANT vtModel;
        VARIANT vtCores;

        hres = pclsObj->Get(L"Name", 0, &vtModel, 0, 0);
        if (SUCCEEDED(hres)) {
            _bstr_t cpuModel(vtModel.bstrVal, false); // Use _bstr_t to convert to string
            std::wcout << L"    CPU Model: " << cpuModel << std::endl;
            VariantClear(&vtModel);
        }

        hres = pclsObj->Get(L"NumberOfCores", 0, &vtCores, 0, 0);
        if (SUCCEEDED(hres)) {
            std::wcout << L"    Number of CPU Cores: " << vtCores.uintVal << std::endl;
            VariantClear(&vtCores);
        }

        pclsObj->Release();
    }

    // Cleanup
    pSvc->Release();
    pLoc->Release();
    pEnumerator->Release();
    CoUninitialize();

    std::cout << "" << std::endl;
}

void printDevice() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "  No CUDA-compatible GPU devices found." << std::endl;
    }

    std::cout << "  Found " << deviceCount << " CUDA-compatible GPU device(s):" << std::endl;

    for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceIndex);

        std::cout << "  Device " << deviceIndex << ": " << deviceProp.name << std::endl;
        std::cout << "      Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "      Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "      Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;

        // Calculate CUDA cores per multiprocessor based on compute capability
        int coresPerMultiprocessor;
        switch (deviceProp.major) {
        case 2:  // Fermi architecture
            coresPerMultiprocessor = 32;
            break;
        case 3:  // Kepler architecture
            coresPerMultiprocessor = 192;
            break;
        case 5:  // Maxwell architecture
            coresPerMultiprocessor = 128;
            break;
        case 6:  // Pascal architecture
            coresPerMultiprocessor = 64;
            break;
        case 7:  // Volta architecture and newer
            coresPerMultiprocessor = 64;
            break;
        default:
            coresPerMultiprocessor = -1;  // Unknown architecture
            break;
        }
        if (coresPerMultiprocessor != -1) {
            std::cout << "      CUDA Cores per Multiprocessor: " << coresPerMultiprocessor << std::endl;
        }
        else {
            std::cout << "      CUDA Cores per Multiprocessor: Unknown" << std::endl;
        }

        std::cout << "  Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
    }
    std::cout << "" << std::endl;
}

void printTitle() {
    std::cout << "  ARRAY COMPARiSON - CPU VS GPU" << std::endl;
    std::cout << "   -h for more options" << std::endl;
    std::cout << "      www.rudenetworks.com | 0.23 beta" << std::endl;
    std::cout << "__________________________________________________________" << std::endl;
}

void printMenu() {
    
    std::cout << "  -t <CPU threads> (default: 4)" << std::endl;
    std::cout << "  -c <Cuda threads per block> (default: 128)" << std::endl;
    std::cout << "  -m <Number of Elements to create> (default: 141000000)" << std::endl;
    std::cout << "  -r <Index to search> (default: Random)" << std::endl;
    std::cout << "  -h (display menu)" << std::endl;
}

int main(int argc, char* argv[]) {

    printTitle();

    for (int i = 1; i < argc; i++) {
        if (std::strncmp(argv[i], "-r", 2) == 0) {
            if (i + 1 < argc) {
                selectedIndex = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-t", 2) == 0) {
            if (i + 1 < argc) {
                numCPUThreads = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-c", 2) == 0) {
            if (i + 1 < argc) {
                numCudaPerBlock = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-m", 2) == 0) {
            if (i + 1 < argc) {
                numOfElements = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-h", 2) == 0) {
            printMenu();
            return;
        }
    }

    if (selectedIndex == -1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distribution(1, numOfElements);
        selectedIndex = distribution(gen);
    }
    
    int* values = createArray();
    cudaCopyArray(values);

    std::cout << "  Selected index: " << selectedIndex << std::endl;
    std::cout << "__________________________________________________________" << std::endl;
    std::cout << "***** Execute CPU *****" << std::endl;
    std::cout << "" << std::endl;
    printHost();
    executeCPU(values, selectedIndex);
    std::cout << "***** Execute GPU *****" << std::endl;
    std::cout << "" << std::endl;
    printDevice();
    executeGPU(selectedIndex);

    delete[] values;

    exit(0);
}


