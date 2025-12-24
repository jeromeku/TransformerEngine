#pragma once

#include "cute/tensor.hpp"

#define PRINT_CUTE(x)        \
    do {                     \
        printf("%s:\t", #x); \
        cute::print(x);      \
        printf("\n");        \
    } while (0)

constexpr uint32_t DELAY = 500000000;  // .5s

__device__ void thread_sleep(int num_iters = 100, int num_cycles = DELAY){
    for (int i = 0; i < num_iters; i++) __nanosleep(num_cycles);
}
