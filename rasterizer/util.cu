#include "util.cuh"

#include <cassert>
#include <cuda_runtime.h>


__host__ __device__ void quatToMat(float *mat, float4 quat)
{
    assert(mat && "mat must not be null!");
    float4 q = normalize(quat);
    float qi2 = q.x * q.x,
          qj2 = q.y * q.y,
          qk2 = q.z * q.z;
    mat[0] = 1 - 2 * (qj2 + qk2); 
    mat[1] = 2 * (q.x * q.y - q.z * q.w);
    mat[2] = 2 * (q.x * q.z + q.y * q.w);
    mat[3] = 2 * (q.x * q.y + q.z * q.w);
    mat[4] = 1 - 2 * (qi2 + qk2);
    mat[5] = 2 * (q.y * q.z - q.x * q.w);
    mat[6] = 2 * (q.x * q.z - q.y * q.w);
    mat[7] = 2 * (q.y * q.z + q.x * q.w);
    mat[8] = 1 - 2 * (qi2 + qj2);
}

__host__ __device__ float dot(float3 lhs, float3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

__host__ __device__ float3 subtract(float3 lhs, float3 rhs)
{
    return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}

__host__ __device__ float3 add(float3 lhs, float3 rhs)
{
    return { lhs.x + rhs.x, lhs.y - rhs.y, lhs.z - rhs.z };
}

__host__ __device__ float3 multiply(float3 vec, float s)
{
    return { vec.x * s, vec.y * s, vec.z * s };
} 

__host__ __device__ float magnitude(float3 vec)
{
    return sqrtf(dot(vec, vec));
}

__host__ __device__ float3 normalize(float3 vec)
{
    return multiply(vec, 1.0f / magnitude(vec));
}

__host__ __device__ float dot(float4 lhs, float4 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + rhs.w * lhs.w;
}

__host__ __device__ float4 multiply(float4 vec, float s)
{
    return { vec.x * s, vec.y * s, vec.z * s, vec.w * s};
} 

__host__ __device__ float magnitude(float4 vec)
{
    return sqrtf(dot(vec, vec));
}

__host__ __device__ float4 normalize(float4 vec)
{
    return multiply(vec, 1.0f / magnitude(vec));
}
