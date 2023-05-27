#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <tuple>
#include <cublas_v2.h>

using namespace std;

#define HX 5 //4 //
#define WX 4 //4 //
#define HW 4 //4 //
#define WW 2 //4 //
#define HY HX //4 //
#define WY WW //4 //

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


__host__ cudnnTensorDescriptor_t createTensorDescriptor(int h, int w)
{
    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 1, c = 1; //, h = 1, w = 10;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);

    return x_desc;
}

__host__ float *runCuDnnActivationForward(cudnnHandle_t handle_, cudnnTensorDescriptor_t x_desc, float *x)
{
    // create activation function descriptor
    float alpha[1] = {1.0};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x);

    return x;
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
    cudaMalloc((void**)&d_matrix, sizeof(float)*num_elements);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float)*num_elements, cudaMemcpyHostToDevice);

    return d_matrix;
}

__global__ void addBias(float* x, float* bias, int height, int width)
{
    int thread_id = threadIdx.x + blockDim.x*blockIdx.x;
    int bias_id = thread_id % width;
    x[thread_id] = x[thread_id] + bias[bias_id];
}

__host__ void addBiasLauncher(float* x, float* bias, int height, int width)
{
    int num_elements = height*width;
    int blocks = num_elements / 1024 + 1;
    int threads = min(1024, num_elements);
    addBias<<<blocks, threads>>>(x, bias, height, width);
}
__host__ void printDeviceFloatArray(float *x, int h, int w, std::string prefix)
{
    float* h_x = (float*) malloc(sizeof(float)*h*w);

    cudaMemcpy(h_x, x, sizeof(float)*h*w, cudaMemcpyDeviceToHost);
    std::cout << prefix << " \n";

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            std::cout << h_x[i*w + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    free(h_x);
}

int main(int argc, char **argv)
{
    OutputHWSpecs();

    bool is_random=true;
    float* h_x = initHostMatrix(HX*WX, is_random, 1.0);
    float* d_x = initDeviceFromHostMatrix(h_x, HX*WX);

    float* h_w = initHostMatrix(HW*WW, is_random, 1.0);
    float* d_w = initDeviceFromHostMatrix(h_w, HW*WW);

    float* h_b = initHostMatrix(HW, is_random, -3.5);
    float* d_b = initDeviceFromHostMatrix(h_b, HW);

    float* h_y = initHostMatrix(HY*WY, false, 0.0);
    float* d_y = initDeviceFromHostMatrix(h_y, HY*WY);

    // prints
    printDeviceFloatArray(d_x, HX, WX, "x init: ");
    printDeviceFloatArray(d_w, HW, WW, "w init: ");
    printDeviceFloatArray(d_y, HY, WY, "y init: ");

    float alpha[1] = {1.0};
    float beta[1] = {0.0};


    // create cublas handle
    cublasHandle_t cublas_handle;
    cublasCreate_v2(&cublas_handle);
    cublasStatus_t stat;

    // dot product input x by weights w
    stat = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        //WX, HW, WW,
        WW, HX, HW,
        alpha,
        d_w, WW,
        d_x, WX,
        beta,
        d_y, WW
    );

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("gemm failed");
        return EXIT_FAILURE;
    }

    printDeviceFloatArray(d_y, HY, WY, "output after GEMM: ");

    // add bias to the product
    addBiasLauncher(d_y, d_b, HY, WY);

    printDeviceFloatArray(d_y, HY, WY, "output after adding bias: ");

    //create cudnn handle
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);

    auto y_desc = createTensorDescriptor(HY, WY);
    runCuDnnActivationForward(cudnn_handle, y_desc, d_y);
    
    printDeviceFloatArray(d_y, HY, WY, "output after activation: ");

    // destroy cublas handle
    cublasDestroy(cublas_handle);

    // destroy cudnn handle
    cudnnDestroy(cudnn_handle);

    // free host & device memory
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_y);
    free(h_x);
    free(h_w);
    free(h_b);
    free(h_y);
    return 0;
}
