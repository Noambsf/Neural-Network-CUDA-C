/*
  Example to compile with the DEBUG key to activate error checking :
    nvcc error_sol.cu -D DEBUG

  Returns "unspecified launch failure in error2_sol.cu at line 13" at runtime
*/

#include <stdio.h>
#include "error.h"

__global__ void foo(int *ptr)
{
  *ptr = 7;
}

int main(void)
{
  foo<<<1,1>>>(0);
  CHECK_ERROR(cudaGetLastError());
  CHECK_ERROR(cudaDeviceSynchronize());
  return 0;
}