Certainly, here's another version of the README.md file for your code:

---

# String Furious - CPU vs GPU

![GitHub stars](https://img.shields.io/github/stars/yourusername/your-repo.svg?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/your-repo.svg?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/your-repo.svg?style=social)
![GitHub followers](https://img.shields.io/github/followers/yourusername.svg?style=social)

## Overview

String Furious is a C++ program designed to compare the performance of CPU and GPU in counting the number of strings in an array that start with a specified substring. The program generates a random array of strings and provides options to perform the counting operation using CPU threads or CUDA threads on a GPU. It measures the execution time for both CPU and GPU operations and reports the results.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Options](#options)
- [Building](#building)
- [License](#license)

## Introduction

String Furious is a versatile tool for evaluating the computational power of CPUs and GPUs when dealing with string operations. By generating random strings and searching for a specific substring, it allows users to assess the performance gap between CPU and GPU processing.

## Prerequisites

Before using String Furious, ensure you have the following prerequisites installed:

- C++ compiler with C++11 support or later.
- NVIDIA GPU with CUDA support (for GPU acceleration).
- CUDA Toolkit installed (for GPU acceleration).
- CMake for building the project.

## Usage

Follow these steps to utilize String Furious:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Build the project using CMake:

   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

3. Execute the String Furious program with your preferred options:

   ```bash
   ./StringFurious [options]
   ```

## Options

String Furious offers several command-line options for customization:

- `-t <CPU threads>`: Set the number of CPU threads to use (default: 4).
- `-c <Cuda threads per block>`: Specify the number of CUDA threads per block (default: 128).
- `-s <Number of strings to create>`: Define the number of random strings to generate (default: 10000).
- `-l <string length>`: Adjust the maximum length of a random string (default: 100).
- `-m <substring to search>`: Specify the substring to search for in the generated strings.
- `-h`: Display the menu with available options.

## Building

String Furious can be built using CMake. Ensure you have CMake installed and follow the usage instructions above to build the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## About

String Furious is developed by Your Name. Visit our website at [www.rudenetworks.com](https://www.rudenetworks.com).

If you encounter any issues or have questions, please [create an issue](https://github.com/yourusername/your-repo/issues) on GitHub.

Thank you for using String Furious!

---
