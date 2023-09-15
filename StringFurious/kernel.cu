
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <execution>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <Windows.h>
#include <Wbemidl.h>
#include <comutil.h>

#pragma comment(lib, "wbemuuid.lib")

const int DEFAULT_VECTOR_STRING_LENGTH = 1000; // Maximum length of a random string
const int DEAFULT_NUM_STRINGS = 100000;     // Number of random strings to generate
const int DEAFULT_NUM_THREADS = 4;
const int DEAFULT_CUDA_PER_BLOCK = 128;

int stringLength = DEFAULT_VECTOR_STRING_LENGTH;
int numOfStrings = DEAFULT_NUM_STRINGS;
int numOfThreads = DEAFULT_NUM_THREADS;
int numOfCudaBlocks = DEAFULT_CUDA_PER_BLOCK;

__device__ int customStrncmp(const char* s1, const char* s2, int n) {
    for (int i = 0; i < n; ++i) {
        if (s1[i] != s2[i]) {
            return s1[i] - s2[i];
        }
    }
    return 0;
}

__global__ void countStringsWithSubstringKernel(const char* strings, const char* substring, int* results, int numStrings, int substringLength, int _stringlength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numStrings) {
        const char* str = strings + tid * _stringlength;
        if (customStrncmp(str, substring, substringLength) == 0) {
            atomicAdd(results, 1);
        }
    }
}

void allocateCUDAMemory(const std::vector<std::string>& strings, const std::string& substring, char*& d_strings, char*& d_substring, int*& d_results) {

    cudaMalloc((void**)&d_strings, numOfStrings * stringLength);
    cudaMalloc((void**)&d_substring, substring.size() + 1);
    cudaMalloc((void**)&d_results, sizeof(int));

    cudaMemcpy(d_substring, substring.c_str(), substring.size() + 1, cudaMemcpyHostToDevice);

    for (int i = 0; i < numOfStrings; ++i) {
        cudaMemcpy(d_strings + i * stringLength, strings[i].c_str(), stringLength, cudaMemcpyHostToDevice);
    }
}

void freeCUDAMemory(char* d_strings, char* d_substring, int* d_results) {
    cudaFree(d_strings);
    cudaFree(d_substring);
    cudaFree(d_results);
}

std::string generateRandomString(int length) {
    static const char charset[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const int charsetSize = sizeof(charset) - 1;

    std::string result;
    result.reserve(length);

    for (int i = 0; i < length; ++i) {
        result += charset[rand() % charsetSize];
    }

    return result;
}

// Function to generate an array of random strings
std::vector<std::string> generateRandomStringArray(int numStrings) {
    std::vector<std::string> randomStrings;
    randomStrings.reserve(numStrings);

    for (int i = 0; i < numStrings; ++i) {
        //int randomLength = rand() % MAX_STRING_LENGTH + 1;

        int randomLength = stringLength;
        std::string randomStr = generateRandomString(randomLength);
        randomStrings.push_back(randomStr);
    }

    return randomStrings;
}

int countStringsWithSubstring(const std::vector<std::string>& strings, const std::string& substring, int start, int end) {
    int count = 0;

    for (int i = start; i < end; ++i) {
        if (strings[i].find(substring) == 0) {
            count++;
        }
    }

    return count;
}

int parallelCount(const std::vector<std::string>& strings, const std::string& substring) {

    int count = 0;
    std::vector<std::thread> threads;
    std::vector<int> threadResults(numOfThreads, 0);

    for (int i = 0; i < numOfThreads; ++i) {
        int start = i * (numOfStrings / numOfThreads);
        int end = (i == numOfThreads - 1) ? numOfStrings : (i + 1) * (numOfStrings / numOfThreads);

        threads.emplace_back([&, start, end, i] {
            int threadCount = countStringsWithSubstring(strings, substring, start, end);
        threadResults[i] = threadCount;
            });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    for (int threadCount : threadResults) {
        count += threadCount;
    }

    return count;
}


// Function to count the number of strings starting with a specific substring (CUDA)
int countStringsWithSubstringCUDA(const std::vector<std::string>& strings, const std::string& substring, char*& d_strings, char*& d_substring, int*& d_results) {
    int countCUDA = 0;

    // Kernel launch
    int threadsPerBlock = numOfCudaBlocks;
    int blocksPerGrid = (numOfStrings + threadsPerBlock - 1) / threadsPerBlock;

    countStringsWithSubstringKernel << <blocksPerGrid, threadsPerBlock >> > (d_strings, d_substring, d_results, numOfStrings, substring.size(), stringLength);
    cudaDeviceSynchronize();
    // Copy the result back to the CPU
    cudaMemcpy(&countCUDA, d_results, sizeof(int), cudaMemcpyDeviceToHost);

    return countCUDA;
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
    std::cout << "  STRiNG FURiOus - CPU VS GPU" << std::endl;
    std::cout << "   -h for more options" << std::endl;
    std::cout << "      www.rudenetworks.com | 0.9 beta" << std::endl;
    std::cout << "_____________________________________________________________" << std::endl;
}

void printMenu() {

    std::cout << "  -t <CPU threads> (default: 4)" << std::endl;
    std::cout << "  -c <Cuda threads per block> (default: 128)" << std::endl;
    std::cout << "  -s <Number of strings to create> (default: 10000)" << std::endl;
    std::cout << "  -l <string length> (default: 100)" << std::endl;
    std::cout << "  -m <substring to search>" << std::endl;
    std::cout << "  -h (display menu)" << std::endl;
}



int main(int argc, char* argv[]) {

    printTitle();
    std::string substringToFind = "ab";

    for (int i = 1; i < argc; i++) {

        if (std::strncmp(argv[i], "-t", 2) == 0) {
            if (i + 1 < argc) {
                numOfThreads = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-c", 2) == 0) {
            if (i + 1 < argc) {
                numOfCudaBlocks = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-s", 2) == 0) {
            if (i + 1 < argc) {
                numOfStrings = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-m", 2) == 0) {
            substringToFind = argv[i + 1];
            return;
        }

        if (std::strncmp(argv[i], "-l", 2) == 0) {
            stringLength = std::atoi(argv[i + 1]);
            return;
        }


        if (std::strncmp(argv[i], "-h", 2) == 0) {
            printMenu();
            return;
        }
    }


    srand(static_cast<unsigned int>(time(nullptr)));

    std::cout << "  - Generate random string array:" << numOfStrings << " items" << std::endl;
    auto startMeasure = std::chrono::high_resolution_clock::now();
    std::vector<std::string> randomStringArray = generateRandomStringArray(numOfStrings);
    auto endMeasure = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationMeasure = endMeasure - startMeasure;
    std::cout << "  - Execution Complete: " << durationMeasure.count() << " seconds" << std::endl;
    std::cout << "_____________________________________________________________" << std::endl;

    char* d_strings;
    char* d_substring;
    int* d_results;

    startMeasure = std::chrono::high_resolution_clock::now();
    allocateCUDAMemory(randomStringArray, substringToFind, d_strings, d_substring, d_results);
    endMeasure = std::chrono::high_resolution_clock::now();
    durationMeasure = endMeasure - startMeasure;
    std::cout << "  - GPU Memory Allocation Time: " << durationMeasure.count() << " seconds" << std::endl;
    std::cout << "_____________________________________________________________" << std::endl;

    std::cout << "  - Execute CPU Parallel (Threads:" << numOfThreads << "): " << std::endl;
    printHost();
    // Measure the execution time of CPU function
    startMeasure = std::chrono::high_resolution_clock::now();
    int countCPU = parallelCount(randomStringArray, substringToFind);
    endMeasure = std::chrono::high_resolution_clock::now();
    durationMeasure = endMeasure - startMeasure;
    
    std::cout << "  - Execution Complete:" << durationMeasure.count() << " seconds" << std::endl;
    std::cout << "  - Count of strings starting with '" << substringToFind << "' (CPU): " << countCPU << std::endl;
    std::cout << "_____________________________________________________________" << std::endl;

    std::cout << "  - Execute CUDA Device (Threads Per Block:" << numOfCudaBlocks << "): " << std::endl;
    printDevice();

    
    // Count the number of strings with the specified substring using CUDA
    startMeasure = std::chrono::high_resolution_clock::now();
    int countCUDA = countStringsWithSubstringCUDA(randomStringArray, substringToFind, d_strings, d_substring, d_results);
    endMeasure = std::chrono::high_resolution_clock::now();
    durationMeasure = endMeasure - startMeasure;

    std::cout << "  - Execution Complete: " << durationMeasure.count() << " seconds" << std::endl;
    std::cout << "  - Count of strings starting with '" << substringToFind << "' (CUDA): " << countCUDA << std::endl;

    // Free memory allocated on the GPU
    freeCUDAMemory(d_strings, d_substring, d_results);

    return 0;
}