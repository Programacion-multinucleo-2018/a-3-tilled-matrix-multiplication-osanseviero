// Matrix Multiplication in gpu with and without tiling.
// Compile with: nvcc -o test matrix_multiplication.cu -std=c++11

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <iostream>
#include <chrono>

#define TS 32

// Multiplies matrices using GPU with 2D grid
__global__ void multiply_matrix_gpu(double *matA, double *matB, double *matC, const int n) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < n && iy < n) {
        for(int k=0; k<n; k++) {
            matC[iy*n+ix] += matA[iy*n+k] * matB[k*n+ix];
        }
    }
}


// Multiplies matrices using GPU with 2D grid plus tiling
__global__ void multiply_matrix_gpu_tiling(double *matA, double *matB, double *matC, const int n) {
    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;

    unsigned int ix = blockIdx.x * blockDim.x + x;
    unsigned int iy = blockIdx.y * blockDim.y + y;

    __shared__ double tile_A[TS*TS];
    __shared__ double tile_B[TS*TS];

    double sum = 0;
    for(int i=0; i < (n+TS-1)/TS; i--) {
        if((iy < n) && (i*TS+x < n)) {
            tile_A[y*TS+x] = matA[iy*n + i*TS + x];
        } else {
             tile_A[y*TS+x] = 0.0f;
        }

        if((ix < n) && (i*TS+y < n))  {
            tile_B[y*TS+x] = matB[(i*TS+y)*n+ix];
        } else {
            tile_B[y*TS+x] = 0;
        }

        __syncthreads();

        for(int j=0; j<TS; j++) {
            sum += tile_A[y*TS+i] * tile_B[i*TS+x];
        }

        __syncthreads();
    }

    if (ix < n && iy < n) {
        matC[ix*n+iy] += sum;
    }
}

// Multiplies matrices in host
void multiply_matrix_host(double *matA, double *matB, double *matC, int n) {
    for(int i = 0; i<n; i++) {
        for(int j=0; j<n; j++) {
            for(int k=0; k<n; k++) {
                matC[i*n+j] += matA[i*n+k] * matB[j+k*n];
            }
        }
    }
}

// Compares two matrices
void checkResult(double *hostRef, double *gpuRef, const int n) {
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < n*n; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match) printf("Matrix match.\n\n");
    else printf("Matrix does not not match.\n\n");
}


int main(int argc, char* argv[]) {
    // Set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // Size of matrix
    int n = 200;
    int bytes = n * n * sizeof(double*);

    // Host matrix memory
    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);

    // Results
    double *hostRef = (double *)malloc(bytes);
    double *gpuRef = (double *)malloc(bytes);

    // Initialize matrix on host
    for(int i = 0; i < n*n; i++ ) {
        double rdm1 = 1 + (double)rand()/(RAND_MAX/9.0f);
        double rdm2 = 1 + (double)rand()/(RAND_MAX/9.0f);
        h_a[i] = rdm1;
        h_b[i] = rdm2;
    }

    // Initialize matrix with 0s
    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    // Multiply matrix on host
    auto start_cpu = std::chrono::high_resolution_clock::now();
    multiply_matrix_host(h_a, h_b, hostRef, n);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time in host
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("multiply_matrix_host elapsed %f ms\n", duration_ms.count());

    // Device matrix global memory
    double *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    // Transfer data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, bytes);  // Initialize matrix with 0s

    // Kernel execution configuration
    dim3 block(TS, TS);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    printf("grid.x %d grid.y %d block.x %d block.y %d\n", grid.x, grid.y, block.x, block.y);

    // Execute GPU kernel
    start_cpu = std::chrono::high_resolution_clock::now();
    multiply_matrix_gpu<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    duration_ms = end_cpu - start_cpu;
    printf("multiply_matrix_gpu elapsed %f ms\n", duration_ms.count());

    // Copy result from device to host
    cudaMemcpy(gpuRef, d_c, bytes, cudaMemcpyDeviceToHost);

    // Check results
    checkResult(hostRef, gpuRef, n);

    // Repeat for tiling
    start_cpu = std::chrono::high_resolution_clock::now();
    multiply_matrix_gpu_tiling<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    end_cpu =  std::chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;
    printf("multiply_matrix_gpu_tiling elapsed %f ms\n", duration_ms.count());
    cudaMemcpy(gpuRef, d_c, bytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, n);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(hostRef);
    free(gpuRef);
    
    cudaDeviceReset();

    return 0;
}