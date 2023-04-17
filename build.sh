#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$PATH:$CUDA_HOME/bin
export CUDA_LIBRARY_PATH=$CUDA_HOME/lib64

nvcc -std=c++14 -O3 -Xcompiler -static-libstdc++,-O3 ip_sockets.cpp tcp_sockets.cpp dprintf.cpp jsonxx.cc Dynexchip.cpp kernel.cu -o dynexsolve -lcurl libCrypto.a

#strip dynexsolve 2>/dev/null
