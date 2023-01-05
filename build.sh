#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export PATH=$PATH:$CUDA_HOME/bin
export CUDA_LIBRARY_PATH=$CUDA_HOME/lib64

ln -sf  `g++ -print-file-name=libstdc++.a`

nvcc -std=c++11 -O3 ip_sockets.cpp tcp_sockets.cpp dprintf.cpp jsonxx.cc Dynexchip.cpp kernel.cu -o dynexsolve -lcurl libCrypto.a libstdc++.a

strip dynexsolve
