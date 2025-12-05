g++ gemm_tf32.cpp -std=c++17 -I/usr/local/musa/include -L /usr/local/musa/lib/ -fopenmp -lmudnn -lmusart -o gemm_tf32 -O2
