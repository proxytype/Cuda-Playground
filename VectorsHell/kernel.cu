#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <stdio.h>
#include <random>
#include <thread>
#include <chrono> // Added for timing
#include <Windows.h>
#include <Wbemidl.h>
#include <comutil.h>
#include <iostream>

#pragma comment(lib, "wbemuuid.lib")

const int MAX_THREADS = 4;
const int MAX_CUDA_PER_BLOCK = 128;
const int MAX_VECTORS = 10000;
const int MAX_ELEMENTS_IN_VECTOR = 10000;

int numVectors = MAX_VECTORS;
int maxElements = MAX_ELEMENTS_IN_VECTOR;
int numThreads = MAX_THREADS;
int numCudaPerBlock = MAX_CUDA_PER_BLOCK;

__global__ void multiplyVectorElementsGPU(int* d_vec, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		d_vec[idx] *= 10;
		d_vec[idx] += 10;
		d_vec[idx] -= 5;
		d_vec[idx] /= 2;
	}
}

std::vector<int>* createArrayOfVectors(int numVectors, int maxElements) {
    std::cout << "  Creating array of vectors" << std::endl;
    std::cout << "  - Total Elements:" << numVectors * maxElements << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

	std::vector<int>* arrayOfVectors = new std::vector<int>[numVectors];

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> distribution(10, 100);

	for (int i = 0; i < numVectors; ++i) {
		for (int j = 0; j < maxElements; ++j) {
			arrayOfVectors[i].push_back(distribution(gen));
		}
	}

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "  Complete: " << elapsed.count() << " milliseconds" << std::endl;
    std::cout << "__________________________________________________________" << std::endl;
	return arrayOfVectors;
}

void allocateAndCopyAllVectorsToDevice(int*** d_vectors, std::vector<int>* arrayOfVectors, int numVectors, int maxElements) {
	*d_vectors = new int* [numVectors];

	for (int i = 0; i < numVectors; ++i) {
		int totalNumElements = maxElements;

		cudaMalloc((void**)&((*d_vectors)[i]), totalNumElements * sizeof(int));
		cudaMemcpy((*d_vectors)[i], arrayOfVectors[i].data(), totalNumElements * sizeof(int), cudaMemcpyHostToDevice);
	}
}

void executeGPU(std::vector<int>* arrayOfVectors, int numVectors, int maxElements) {
	int** d_vectors;
	allocateAndCopyAllVectorsToDevice(&d_vectors, arrayOfVectors, numVectors, maxElements);

	int threadsPerBlock = numCudaPerBlock;
	int totalNumElements = numVectors * maxElements;
	int blocksPerGrid = (totalNumElements + threadsPerBlock - 1) / threadsPerBlock;

	auto start = std::chrono::high_resolution_clock::now();

	multiplyVectorElementsGPU << <blocksPerGrid, threadsPerBlock >> > (d_vectors[0], totalNumElements);

	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;

	printf("  GPU Execution Time: %.4f seconds\n", elapsed.count());

	for (int i = 0; i < numVectors; ++i) {
		cudaMemcpy(arrayOfVectors[i].data(), d_vectors[i], maxElements * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(d_vectors[i]);
	}

	delete[] d_vectors;
    std::cout << "__________________________________________________________" << std::endl;
}


void multiplyVectorElements(std::vector<int>& vec, int start, int end) {

	for (int i = start; i < end; ++i) {
		vec[i] *= 10;
		vec[i] += 10;
		vec[i] -= 5;
		vec[i] /= 2;
	}
}

void executeCPU(std::vector<int>* arrayOfVectors, int numVectors, int numThreads) {
	std::vector<std::thread> threads;

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < numVectors; ++i) {
		int elementsPerVector = arrayOfVectors[i].size();
		int elementsPerThread = elementsPerVector / numThreads;

		for (int j = 0; j < numThreads; ++j) {
			int start = j * elementsPerThread;
			int end = (j == numThreads - 1) ? elementsPerVector : (j + 1) * elementsPerThread;

			threads.emplace_back(multiplyVectorElements, std::ref(arrayOfVectors[i]), start, end);
		}
	}

	for (std::thread& t : threads) {
		t.join();
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;

	printf("  CPU Execution Time: %.4f seconds\n", elapsed.count());
    std::cout << "__________________________________________________________" << std::endl;
}

void printHost() {
    HRESULT hres;

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
            _bstr_t cpuModel(vtModel.bstrVal, false);
            std::wcout << L"  CPU Model: " << cpuModel << std::endl;
            VariantClear(&vtModel);
        }

        hres = pclsObj->Get(L"NumberOfCores", 0, &vtCores, 0, 0);
        if (SUCCEEDED(hres)) {
            std::wcout << L"  Number of CPU Cores: " << vtCores.uintVal << std::endl;
            VariantClear(&vtCores);
        }

        pclsObj->Release();
    }

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
    std::cout << "  VECTORS HeLL - CPU VS GPU" << std::endl;
    std::cout << "   -h for more options" << std::endl;
    std::cout << "      www.rudenetworks.com | 0.11 beta" << std::endl;
    std::cout << "__________________________________________________________" << std::endl;
}

void printMenu() {

    std::cout << "  -t <CPU threads> (default: 4)" << std::endl;
    std::cout << "  -c <Cuda threads per block> (default: 128)" << std::endl;
    std::cout << "  -v <Number of vector to create> (default: 10000)" << std::endl;
    std::cout << "  -e <Number of elements in vector> (default: 10000)" << std::endl;
    std::cout << "  -h (display menu)" << std::endl;
}

int main(int argc, char* argv[]) {

    printTitle();

    for (int i = 1; i < argc; i++) {

        if (std::strncmp(argv[i], "-t", 2) == 0) {
            if (i + 1 < argc) {
                numThreads = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-c", 2) == 0) {
            if (i + 1 < argc) {
                numCudaPerBlock = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-v", 2) == 0) {
            if (i + 1 < argc) {
                numVectors = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-m", 2) == 0) {
            if (i + 1 < argc) {
                maxElements = std::atoi(argv[i + 1]);
            }
        }

        if (std::strncmp(argv[i], "-h", 2) == 0) {
            printMenu();
            return;
        }
    }

    std::cout << "  Vectors:" << numVectors << " Elements:" << maxElements << std::endl;
    std::cout << "  Threads:" << numThreads << " Cuda Block:" << numCudaPerBlock << std::endl;

    std::cout << "__________________________________________________________" << std::endl;

	std::vector<int>* arrayOfVectors = createArrayOfVectors(numVectors, maxElements);

    std::cout << "***** Execute CPU *****" << std::endl;
    std::cout << "" << std::endl;
    printHost();
	executeCPU(arrayOfVectors, numVectors, numThreads);

    std::cout << "***** Execute GPU *****" << std::endl;
    std::cout << "" << std::endl;
    printDevice();
	executeGPU(arrayOfVectors, numVectors, maxElements);

	delete[] arrayOfVectors;

	return 0;
}
