#ifndef UTIL_CUH
#define UTIL_CUH

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") \
                + cudaGetErrorString(err) \
                + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while(0)

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

struct plane 
{
    // planeDir is the unit vector parallel to the plane
    // normal is the unit vector normal to the plane
    float3 planeDir, normal;
};

template<typename T>
void cudaAsyncDeleter(T *p) {
    if (p)
        cudaFreeAsync((void *)p, cudaStreamPerThread);
};

__host__ __device__ void quatToMat(float *mat, float4 quat);

__host__ __device__ float dot(float3 lhs, float3 rhs);
__host__ __device__ float3 subtract(float3 lhs, float3 rhs);
__host__ __device__ float3 add(float3 lhs, float3 rhs);
__host__ __device__ float3 multiply(float3 vec, float s);
__host__ __device__ float magnitude(float3 vec);
__host__ __device__ float3 normalize(float3 vec);

__host__ __device__ float dot(float4 lhs, float4 rhs);
__host__ __device__ float4 multiply(float4 vec, float s);
__host__ __device__ float magnitude(float4 vec);
__host__ __device__ float4 normalize(float4 vec);
#endif
