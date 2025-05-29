#include <iostream>
#include <functional>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>
#include <exception>
#include <memory>
#include <chrono>
#include <studentlib.h>

void __terminate_gracefully(const std::string &msg) noexcept {
    std::cout << -1 << std::endl;
    std::cerr << msg << std::endl;
    exit(EXIT_SUCCESS);
}

int main(int argc, char *argv[]) {
    #if defined(__cpp_lib_filesystem)
        // Header available
    #else
        __terminate_gracefully("<filesystem> header is not supported by this compiler");
    #endif
    
    try {
        // Parse arguments
        if (argc < 3) __terminate_gracefully("Usage: ./tester.out <N> <B> <optional:seed>");    
        std::random_device rd;
        std::mt19937 rng(argc > 3 ? std::atoi(argv[3]) : rd());
        std::int32_t N = std::atoi(argv[1]), B = std::atoi(argv[2]);
        
        // Util func
        std::function<int(void)> generateRandomValue = [&]() {
            std::uniform_int_distribution<int> distribution(0, B-1);
            return distribution(rng);
        };
        
        // Create input data file
        std::string input_path = std::filesystem::temp_directory_path() / ("input-" + std::to_string(N) + ".dat");
        std::cout << "[1/4] Looking for input file" << std::endl;
        
        auto gen_input = [&](std::string &file_path, int size) {
            auto data = std::make_unique<int[]>(size);
            if (std::filesystem::exists(file_path)) {
                std::cout << "\t- Input file: " << file_path << " found, using existing input file" << std::endl;
                std::ifstream in_fs(file_path, std::ios::binary);
                in_fs.read(reinterpret_cast<char*>(data.get()), sizeof(int) * size);
            }
            else {
                std::cout << "\t- Input file not found. Creating new test data: " << file_path << std::endl;
                std::ofstream out_fs(file_path, std::ios::binary);
                for (std::int32_t i = 0; i < size; i++) data[i] = generateRandomValue();
                out_fs.write(reinterpret_cast<char*>(data.get()), sizeof(int) * size);
            }
            return std::move(data);
        };
        
        auto input_data = gen_input(input_path, N);
        
        // Create solution file
        std::string sol_path = std::filesystem::temp_directory_path() / ("sol-" + std::to_string(N) +"-" + std::to_string(B) + ".dat");
        std::cout << "[2/4] Looking for verification file " << sol_path << std::endl;
        
        if (std::filesystem::exists(sol_path)) {
            std::cout << "[3/4] Verification file found, using existing verification data" << std::endl;
        }
        else {
            std::cout << "[3/4] Verification file not found. Creating new verification data" << std::endl;
            auto histogram = std::make_unique<int[]>(B);
            for (int i = 0; i < B; i++) histogram[i] = 0;
            
            for (int i = 0; i < N; i++) {
                histogram[input_data[i]]++;
            }
            
            std::ofstream sol_fs(sol_path, std::ios::binary);
            sol_fs.write(reinterpret_cast<char*>(histogram.get()), sizeof(int) * B);
        }
        
        input_data.reset();
        
        std::cout << "[4/4] Running student solution" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        const std::string student_sol_path = solution::compute(input_path, N, B);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        
        // Verify solution
        std::ifstream sol_fs(sol_path, std::ios::binary);
        std::ifstream student_fs(student_sol_path, std::ios::binary);
        
        auto sol_hist = std::make_unique<int[]>(B);
        auto student_hist = std::make_unique<int[]>(B);
        
        sol_fs.read(reinterpret_cast<char*>(sol_hist.get()), sizeof(int) * B);
        student_fs.read(reinterpret_cast<char*>(student_hist.get()), sizeof(int) * B);
        
        for (int i = 0; i < B; i++) {
            if (sol_hist[i] != student_hist[i]) {
                __terminate_gracefully("Histogram mismatch at bin " + std::to_string(i) + 
                                      ": expected " + std::to_string(sol_hist[i]) + 
                                      ", got " + std::to_string(student_hist[i]));
            }
        }
        
        std::filesystem::remove(student_sol_path);
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    
    } catch(const std::exception &e) {
        __terminate_gracefully(e.what());
    }
    return EXIT_SUCCESS;
}