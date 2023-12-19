
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
__constant__ float d_filter[3][3];

inline __device__ uchar3 operator*(uchar3 a, float b) {
    uchar3 c;
    c.x = a.x * b;
    c.y = a.y * b;
    c.z = a.z * b;

    return c;
}

inline __device__ uchar3 operator+(uchar3 a, uchar3 b) {
    uchar3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

__global__ void ConvKernel(uchar3* d_image, size_t pitch, size_t width, size_t height, uchar3* out_image,
    size_t out_pitch) {
    int i = (blockIdx.y * blockDim.y + threadIdx.y) * pitch;
    int j = (blockIdx.x * blockDim.x + threadIdx.x) * sizeof(uchar3);
    if (i >= pitch * (height-1) || i == 0) {
        return;
    }
    if (j >= (width-1) * sizeof(uchar3) || j == 0) {
        return;
    }

    auto d_image_ptr = reinterpret_cast<uchar1*>(d_image);
    auto d_image_pos = reinterpret_cast<uchar3*>(d_image_ptr + i + j);
    auto d_image_pre = reinterpret_cast<uchar3*>(d_image_ptr + i - pitch + j);
    auto d_image_post = reinterpret_cast<uchar3*>(d_image_ptr + i + pitch + j);
    auto a11 = d_image_pos[0];
    auto a12 = d_image_pos[1];
    auto a10 = (d_image_pos-1)[0];
    auto a00 = (d_image_pre - 1)[0];
    auto a01 = d_image_pre[0];
    auto a02 = d_image_pre[1];
    auto a20 = (d_image_post - 1)[0];
    auto a21 = d_image_post[0];
    auto a22 = d_image_post[1];

    auto res = a00 * d_filter[0][0] + a01 * d_filter[0][1] + a02 * d_filter[0][2] +
        a10 * d_filter[1][0] + a11 * d_filter[1][1] + a12 * d_filter[1][2] +
        a20 * d_filter[2][0] + a21 * d_filter[2][1] + a22 * d_filter[2][2];

    int i_out = (blockIdx.y * blockDim.y + threadIdx.y) * out_pitch;
    auto out_image_ptr = reinterpret_cast<uchar1*>(out_image);
    auto out_image_pos = reinterpret_cast<uchar3*>(out_image_ptr + i_out + j);
    out_image_pos[0] = res;
}

int main()
{
    float h_filter[3][3];
    uchar3 h_image[1000][1000];
    uchar3* d_image;
    size_t pitch;
    cudaError_t cuda_memcpy_to_symbol_status = cudaMemcpyToSymbol(d_filter, h_filter, 9 * sizeof(float));
    if (cuda_memcpy_to_symbol_status != cudaSuccess) {
        printf("cudaMemcpyToSymbol failed\n");
        return 1;
    }
    cudaError_t cuda_pitch_status = cudaMallocPitch(&d_image, &pitch, 1000 * sizeof(uchar3), 1000);
    if (cuda_pitch_status != cudaSuccess) {
        printf("cudaMallocPitch failed\n");
        return 1;
    }

    cudaError_t cuda_memcpy2d_status = cudaMemcpy2D(d_image, pitch, h_image, 
        1000 * sizeof(uchar3), 1000 * sizeof(uchar3), 1000, cudaMemcpyHostToDevice);

    if (cuda_memcpy2d_status != cudaSuccess) {
        printf("cudaMemcpy2d failed\n");
        return 1;
    }

    dim3 blockDim;
    blockDim.x = 32;
    blockDim.y = 1024 / 32;
    blockDim.z = 1;

    dim3 gridDim;
    gridDim.x = (1000 + blockDim.x - 1) / blockDim.x;
    gridDim.y = (1000 + blockDim.y - 1) / blockDim.y;
    gridDim.z = blockDim.z;

    uchar3* d_out;
    size_t out_pitch;
    cudaMallocPitch(&d_out, &out_pitch, 1000 * sizeof(uchar3), 1000);
    if (cuda_pitch_status != cudaSuccess) {
        printf("cudaMallocPitch failed\n");
        return 1;
    }
    ConvKernel << <blockDim, gridDim >> > (d_image, pitch, 1000, 1000, d_out, out_pitch);
    cudaError_t cuda_device_sync_status = cudaDeviceSynchronize();
    if (cuda_device_sync_status != cudaSuccess) {
        printf("Cuda sync failed");
        return 1;
    }
    cudaError_t cuda_memcpy2d_status = cudaMemcpy2D(h_image, 1000*sizeof(uchar3), d_out,
        out_pitch, 1000 * sizeof(uchar3), 1000, cudaMemcpyDeviceToHost);
    if (cuda_memcpy2d_status != cudaSuccess) {
        printf("Cuda memcpy error\n");
        return 1;
    }

    cudaFree(d_image);
    cudaFree(d_out);
    return 0;
}
