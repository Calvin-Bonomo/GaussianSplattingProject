#include "rasterize.cuh"

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "util.cuh"
#include "settings.hpp"


__host__ __device__ bool isCulled(float3 viewMean, float scaleMax, float zNear, float zFar, plane *clipPlanes);
__host__ __device__ bool isClipped(plane clipPlane, float3 viewMean, float scaleMax);

__global__ void projectGaussians(
        long long numGaussians,
        float3 *means,
        float3 *scales,
        float4 *rotations,
        float3 *cov2D,
        float3 *invCov2D,
        uint2 *means2D,
        float *depths,
        float4 *aabb,
        float *viewTransform,
        float *opacities,
        uint32_t *tilesTouched,
        plane *clipPlanes,
        float2 focal,
        float zNear,
        float zFar,
        int width, int height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numGaussians)
        return;

    aabb[id] = { 0, 0, 0, 0 };
    cov2D[id] = { 0, 0, 0 };
    invCov2D[id] = { 0, 0, 0 };
    means2D[id] = { 0, 0 };
    depths[id] = 0;

    float3 scale = scales[id];
    float3 mean = means[id];
    float3 viewMean = 
    {
       mean.x * viewTransform[0] + mean.y * viewTransform[4] + mean.z * viewTransform[8] + viewTransform[12],
       mean.x * viewTransform[1] + mean.y * viewTransform[5] + mean.z * viewTransform[9] + viewTransform[13],
       mean.x * viewTransform[2] + mean.y * viewTransform[6] + mean.z * viewTransform[10] + viewTransform[14]
    };

    // Perform frustum and opacity culling
    if (opacities[id] < 1.0 / 255 || isCulled(viewMean, fmaxf(scale.x, fmaxf(scale.y, scale.z)), zNear, zFar, clipPlanes))
        return;
    
    // Calculate 2D covariance
    // Setup viewspace transform matrix
    float sr[9];
    for (int i = 0; i < 9; i++)
        sr[i] = 0;
    
    quatToMat(sr, rotations[id]);
    sr[0] *= scale.x;
    sr[4] *= scale.y;
    sr[8] *= scale.z;

    float cov3D[6] = {
        sr[0]*sr[0] + sr[1]*sr[1] + sr[2]*sr[2],  // (0,0)
        sr[0]*sr[3] + sr[1]*sr[4] + sr[2]*sr[5],  // (0,1)
        sr[0]*sr[6] + sr[1]*sr[7] + sr[2]*sr[8],  // (0,2)
        sr[3]*sr[3] + sr[4]*sr[4] + sr[5]*sr[5],  // (1,1)
        sr[3]*sr[6] + sr[4]*sr[7] + sr[5]*sr[8],  // (1,2)
        sr[6]*sr[6] + sr[7]*sr[7] + sr[8]*sr[8],  // (2,2)
    };

    float projMat[6] = {
        focal.x / viewMean.z, 0,                    -focal.x * viewMean.x / (viewMean.z * viewMean.z),
        0,                    focal.y / viewMean.z, -focal.y * viewMean.y / (viewMean.z * viewMean.z)
    };

    float worldToProj[6];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            worldToProj[i * 3 + j] = 0;
            for (int k = 0; k < 3; k++)
                worldToProj[i * 3 + j] += projMat[i * 3 + j] * viewTransform[j * 3 + k];
        }
    }

    float sigma[9] = {
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    };

    float projCov[6];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            projCov[i * 3 + j] = 0;
            for (int k = 0; k < 3; k++)
                projCov[i * 3 + j] += worldToProj[i * 3 + j] * sigma[j * 3 + k];
        }
    }

    float cov2DMat[4];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            cov2DMat[i * 2 + j] = 0;
            for (int k = 0; k < 3; k++)
                cov2DMat[i * 2 + j] += projCov[i * 3 + j] * worldToProj[j * 3 + k];
        }
    }

    // Calculate gaussian extent 
    float discriminantRT = sqrtf(powf(cov2DMat[0] - cov2DMat[3], 2) + 4 * powf(cov2DMat[1], 2)),
          lambda1 = 0.5 * (cov2DMat[0] + cov2DMat[3] + discriminantRT),
          lambda2 = 0.5 * (cov2DMat[0] + cov2DMat[3] - discriminantRT),
          theta = 0;
    if (cov2DMat[1] != 0)
        theta = atan2(lambda1, cov2DMat[1]);
    else if (cov2DMat[0] < cov2DMat[3])
        theta = PI * 0.5;
    
    // Calculate a safe bounds for the AABB
    float r1 = 3 * sqrtf(lambda1); // Major axis
    float r2 = 3 * sqrtf(lambda2); // Minor axis
    
    float cosTheta = cosf(theta),
          sinTheta = sinf(theta);

    // Basic SAT with AABB
    float2 p1 = {-r1, -r2},
           p2 = {r1, r2},
           p3 = {-r1, r2},
           p4 = {r1, r2};

    p1 = {p1.x * cosTheta + p1.y * (-sinTheta), p1.x * sinTheta + p1.y * cosTheta};
    p2 = {p2.x * cosTheta + p2.y * (-sinTheta), p2.x * sinTheta + p2.y * cosTheta};
    p3 = {p3.x * cosTheta + p3.y * (-sinTheta), p3.x * sinTheta + p3.y * cosTheta};
    p4 = {p4.x * cosTheta + p4.y * (-sinTheta), p4.x * sinTheta + p4.y * cosTheta};
    
    // Get min and max tile indices
    float2 min = 
    {
       ceilf(abs((fminf(p1.x, fminf(p2.x, fminf(p3.x, p4.x)))) / TILE_SIZE)),
       ceilf(abs((fminf(p1.y, fminf(p2.y, fminf(p3.y, p4.y)))) / TILE_SIZE))
    },
           max = 
    {
       ceilf(abs((fmaxf(p1.x, fmaxf(p2.x, fmaxf(p3.x, p4.x)))) / TILE_SIZE)),
       ceilf(abs((fmaxf(p1.y, fmaxf(p2.y, fmaxf(p3.y, p4.y)))) / TILE_SIZE)),
    };

    // Calculate inverse covariance matrix
    float det = 1.0f / (cov2DMat[0] * cov2DMat[3] - cov2DMat[1] * cov2DMat[1]); // 3 and 1 are equivalent
    float invCov2DMat[4] = {
        det * cov2DMat[3],    det * (-cov2DMat[1]),
        det * (-cov2DMat[1]), det * cov2DMat[0]
    };
    
    // Save data for future stages
    tilesTouched[id] = (min.x + max.x) * (min.y + max.y);
    aabb[id] = { min.x, min.y, max.x, max.y };
    cov2D[id] = { cov2DMat[0], cov2DMat[1], cov2DMat[3] };
    invCov2D[id] = { invCov2DMat[0], invCov2DMat[1], invCov2DMat[3] };
    means2D[id] = { uint((viewMean.x / viewMean.z + 1) * 0.5f * width), uint((viewMean.y / viewMean.z + 1) * 0.5 * height) };
    depths[id] = viewMean.z;
}

__global__ void duplicateWithKeys(
        long long numGaussians,
        uint2 *means2D,
        float *depths,
        float3 *cov2D,
        float4 *aabb,
        uint32_t *tilesTouched,
        uint64_t *gaussianKeys,
        uint64_t *gaussianIndices,
        uint32_t *gaussianOffsets,
        int xTiles, int yTiles)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= numGaussians || tilesTouched[id] == 0)
        return;

    float4 bounds = aabb[id];
    uint2 mean = means2D[id];
    float xBound = abs(bounds.x + bounds.z),
          yBound = abs(bounds.y + bounds.w);
    int startTileId = mean.x - (0.5f * xBound) + (mean.y - 0.5f * yBound) * xTiles,
        offset = gaussianOffsets[id],
        tilesWritten = 0,
        depthAsInt = *(int *)(&depths[id]); // I am so so sorry :(

    for (int i = 0; i < xBound; i++)
    {
        for (int j = 0; j < yBound; j++)
        {
            int tileId = startTileId + xBound + xTiles * yBound;
            gaussianKeys[offset + tilesWritten] = ((uint64_t)tileId) << 32 | depthAsInt;
            gaussianIndices[offset + (tilesWritten++)] = id;
        }
    }
}

__global__ void identifyTileRanges(
        int totalTiles,
        int totalTilesTouched,
        uint64_t *gaussianKeys, 
        int2 *tileRanges)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= totalTiles)
        return;

    int startIndex = -1;
    tileRanges[id] = { -1, -1 };

    for (int i = 0; i < totalTilesTouched; i++)
    {
        if (gaussianKeys[i] >> 32 == id)
        {
            if (startIndex < 0)
                startIndex = i;
        }
        else if (startIndex >= 0)
        {
            tileRanges[id] = { startIndex, i };
            return;
        }
    }
}

__host__ __device__ bool isCulled(float3 viewMean, float scaleMax, float zNear, float zFar, plane *clipPlane) 
{
    // Clip against far and near clipping planes
    if (viewMean.z + scaleMax <= zNear || viewMean.z - scaleMax >= zFar)
            return false;
    return isClipped(clipPlane[0], viewMean, scaleMax)
        || isClipped(clipPlane[1], viewMean, scaleMax)
        || isClipped(clipPlane[2], viewMean, scaleMax)
        || isClipped(clipPlane[3], viewMean, scaleMax);
}
__host__ __device__ bool isClipped(plane clipPlane, float3 viewMean, float scaleMax)
{
    float intersectionMag = dot(clipPlane.planeDir, viewMean);
    float3 distance = subtract(viewMean, multiply(clipPlane.planeDir, intersectionMag));
    return dot(normalize(distance), clipPlane.normal) >= 0 || magnitude(distance) >= scaleMax;
}
