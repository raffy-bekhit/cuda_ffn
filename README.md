# cuda_ffn
A simple project that implements a fully connected layer as follows:
* dot product of x (input), w (weights) using cublas single precision gemm
* adding bias using a cuda kernel
* do sigmoid activation using cuDNN

Matrices are randomly initialized.

#Building & Running

```bash run.sh```
