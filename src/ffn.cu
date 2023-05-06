#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <tuple>
#include <cublas.h>

using namespace std;

#define HX 4
#define WX 4
#define HW 4
#define WW 4
#define HY 4
#define WY 4

__host__ void OutputHWSpecs()
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0); // use GPU0
    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;
}

__host__ float* initHostMatrix(int num_elements, bool is_random, float value)
{
    float* matrix = (float*) malloc(sizeof(float)* num_elements);
    for(int i=0; i<num_elements; i++)
    {
        matrix[i] = is_random ? rand()/(float)rand() : value;
    }
    return matrix;
}

__host__ float* initDeviceFromHostMatrix(float* h_matrix, int num_elements)
{
    float* d_matrix;
    cudaMalloc((void**)&d_matrix, num_elements);
    cudaMemcpy(d_matrix, h_matrix, num_elements, cudaMemcpyFromHostToDevice);

    return d_matrix;
}

__global__ float* addBias(float* x, float* bias, int height, int width)
{
    int thread_id = threadIdx.x + blockDim.x*blockIdx.x;
    int bias_id = thread_id % width;
    x[thread_id] = x[thread_id] + bias[bias_id];
}

__host__ float* addBiasLauncher(float* x, float* bias, int height, int width)
{
    int num_elements = height*width;
    int blocks = threads / 1024 + 1;
    int threads = min(1024, num_elements);
    addBias<<<blocks, threads>>>(x, bias, height, width);
}
__host__ void printFloatArray(float *x, int num_elements)
{
    for (int i = 0; i < num_elements; i++)
        std::cout << x[i] << " ";
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    OutputHWSpecs();

    bool is_random=false;
    float* h_x = initHostMatrix(HX*WX, is_random, 1.0);
    float* d_x = initDeviceFromHostMatrix(h_x, HX*WX);
    float* h_w = initHostMatrix(HW*WW, is_random, 1.0);
    float* d_w = initDeviceFromHostMatrix(h_w, HW*WW);
    float* h_b = initHostMatrix(HW, is_random, 1.0);
    float* d_b = initDeviceFromHostMatrix(h_b, HW);
    float* h_y = initHostMatrix(HX*HW, is_random, 1.0);
    float* d_y = initDeviceFromHostMatrix(h_y, HX*HW);
    
    // create cublas handle
    cublasHanlde_t cublas_handle;
    cublasCreate(&cublas_handle);
    // dot product input x by weights w
    cublasSgemm(
        'n',
        't',
        HX, WW, HW,
        d_x, HX,
        d_w, WW,
        d_y, HX,
    );

    // add bias to the product
    addBiasLauncher(d_y, d_b, HX, HW);

    //create cudnn handle
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    // destroy cublas handle
    cublasDestroy(cublas_handle);

    // destroy cudnn handle
    cudnnDestroy(handle_);

    // free device matrices
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_y);
    return 0;
}