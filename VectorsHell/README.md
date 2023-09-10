# VECTORS HeLL - CPU VS GPU

This is a C++ code that demonstrates the performance difference between CPU and GPU for performing vector element-wise operations. It creates an array of vectors, each containing a specified number of elements, and then applies a simple arithmetic operation to each element in parallel using both CPU and GPU. The code also provides an option to configure the number of threads, CUDA threads per block, the number of vectors, and the number of elements in each vector through command-line arguments.

## Prerequisites

Before running this code, make sure you have the following libraries and tools installed:

- CUDA: This code uses NVIDIA CUDA for GPU processing. Ensure you have a compatible GPU and the CUDA toolkit installed.

## Code Structure

The code consists of the following main parts:

1. **Initialization and Configuration**
   - The code starts by initializing various parameters such as the maximum number of threads, CUDA threads per block, the maximum number of vectors, and the maximum elements in a vector.
   - It also defines a CUDA kernel function `multiplyVectorElementsGPU` to perform the element-wise operations on the GPU.

2. **Data Preparation**
   - The `createArrayOfVectors` function generates an array of vectors with random integer elements within a specified range.

3. **GPU Execution**
   - The `executeGPU` function allocates memory on the GPU, copies data from the CPU to the GPU, launches the CUDA kernel for parallel execution, and then copies the results back to the CPU.

4. **CPU Execution**
   - The `executeCPU` function divides the vector elements into multiple threads for parallel execution on the CPU.

5. **System Information Retrieval**
   - The `printHost` function retrieves information about the host CPU using Windows Management Instrumentation (WMI).
   - The `printDevice` function retrieves information about CUDA-compatible GPUs using the CUDA API.

6. **Command-Line Argument Parsing**
   - The `main` function parses command-line arguments to customize the number of threads, CUDA threads per block, number of vectors, and number of elements in each vector.

## Usage

To run the code, you can use the following command-line arguments:

- `-t <CPU threads>`: Set the number of CPU threads (default: 4).
- `-c <Cuda threads per block>`: Set the number of CUDA threads per block (default: 128).
- `-v <Number of vector to create>`: Set the number of vectors to create (default: 10000).
- `-e <Number of elements in vector>`: Set the number of elements in each vector (default: 10000).
- `-h`: Display the menu with available options.

For example, to run the code with 8 CPU threads, 256 CUDA threads per block, 5000 vectors, and 5000 elements in each vector, you can use the following command:

```
./VectorsHell -t 8 -c 256 -v 5000 -e 5000
```

The code will then execute the element-wise operations on both the CPU and GPU, displaying the execution times and system information.

**Note:** Ensure that you have the required environment set up, including a compatible CUDA GPU, before running the code.
