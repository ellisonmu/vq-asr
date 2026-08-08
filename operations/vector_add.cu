// c[i] = A[i] + B[i]

# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>

__global__ void addKernel(int *c, const int *a, const int *b, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = a[i] + b[i];
}

int main() {
    const int size = 1000000, bytes = size * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);
    
    for (int i = 0; i < size; i++) { // generate two example vectors
        h_a[i] = i; 
        h_b[i] = i * 2;
    }

    int *dev_a, *dev_b, *dev_c;
    //allocate memory on gpu, storing the address in dev_{a,b,c}
    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

    //copy data stored at host memory to device memory
    cudaMemcpy(dev_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_b, bytes, cudaMemcpyHostToDevice); 

    int threadsPerBlock = 256, blocksPerGrid = (size + 255) / 256;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, size);

    //copy data stoed at device memory to host memory
    cudaMemcpy(h_c, dev_c, bytes, cudaMemcpyDeviceToHost);

    printf("Vector addition of %d elements\n", size);
    printf("Verification: c[0]=%d, c[999999]=%d\n", h_c[0], h_c[999999]);

    //free device memory
    cudaFree(dev_a); 
    cudaFree(dev_b); 
    cudaFree(dev_c);

    //free host memeory
    free(h_a); 
    free(h_b); 
    free(h_c);
    return 0;
}