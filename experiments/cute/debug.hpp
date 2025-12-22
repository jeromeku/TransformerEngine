#pragma once

#include "cute/tensor.hpp"

#define PRINT_CUTE(x)        \
    do {                     \
        printf("%s:\t", #x); \
        cute::print(x);      \
        printf("\n");        \
    } while (0)

