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
        float2 *means2D,
        float *depths,
        float4 *aabb,
        float *viewTransform,
        float *opacities,
        uint32_t *tilesTouched,
        plane *clipPlanes,
        float2 focal,
        float zNear, float zFar,
        int xTiles, int yTiles,
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

    float sigma[9] = {
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[1], cov3D[3], cov3D[4],
        cov3D[2], cov3D[4], cov3D[5]
    };

    float projMat[6] = {
        focal.x / viewMean.z, 0,                    -focal.x * viewMean.x / (viewMean.z * viewMean.z),
        0,                    focal.y / viewMean.z, -focal.y * viewMean.y / (viewMean.z * viewMean.z)
    };
    // View-space rotation (3x3 from upper-left of viewTransform, column-major)
    float W[9] = {
        viewTransform[0], viewTransform[4], viewTransform[8],
        viewTransform[1], viewTransform[5], viewTransform[9],
        viewTransform[2], viewTransform[6], viewTransform[10]
    };

    float T[6];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            T[i*3 + j] = 0;
            for (int k = 0; k < 3; k++)
                T[i*3 + j] += projMat[i*3 + k] * W[k*3 + j];
        }
    
    float tmp[6];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            tmp[i*3 + j] = 0;
            for (int k = 0; k < 3; k++)
                tmp[i*3 + j] += T[i*3 + k] * sigma[k*3 + j];
        }

    float cov2DMat[4];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            cov2DMat[i*2 + j] = 0;
            for (int k = 0; k < 3; k++)
                cov2DMat[i*2 + j] += tmp[i*3 + k] * T[j*3 + k];  // T^T means swap indices
    }


    // Calculate gaussian extent 
    float discriminantRT = sqrtf(powf(cov2DMat[0] - cov2DMat[3], 2) + 4 * powf(cov2DMat[1], 2)),
          lambda1 = 0.5 * (cov2DMat[0] + cov2DMat[3] + discriminantRT),
          lambda2 = 0.5 * (cov2DMat[0] + cov2DMat[3] - discriminantRT);

    // Calculate a safe bounds for the AABB
    float r = 3 * sqrtf(fmaxf(lambda1, lambda2));
    

    // Basic SAT with AABB
    float2 pixelMean = { 
        focal.x * viewMean.x / viewMean.z + width * 0.5f, 
        focal.y * viewMean.y / viewMean.z + height * 0.5f 
    };
    
    // Get min and max tile indices
    float2 minPx = 
    {
        pixelMean.x - r,
        pixelMean.y - r
    },
           maxPx = 
    {
        pixelMean.x + r,
        pixelMean.y + r
    };

    // Calculate inverse covariance matrix
    float quo = cov2DMat[0] * cov2DMat[3] - cov2DMat[1] * cov2DMat[1]; // 3 and 1 are equivalent
    if (quo == 0) return;
    float det = 1.0f / quo;
    float invCov2DMat[4] = {
        det * cov2DMat[3],    det * (-cov2DMat[1]),
        det * (-cov2DMat[1]), det * cov2DMat[0]
    };

    int2 tMin = { max(0, min(xTiles, (int)(minPx.x / TILE_SIZE))), max(0, min(yTiles, (int)(minPx.y / TILE_SIZE))) };
    int2 tMax = { max(0, min(xTiles, (int)((maxPx.x + TILE_SIZE - 1) / TILE_SIZE))), max(0, min(yTiles, (int)((maxPx.y + TILE_SIZE - 1) / TILE_SIZE))) };
    
    int tileWidth = max(0, tMax.x - tMin.x),
        tileHeight = max(0, tMax.y - tMin.y);
    
    // Save data for future stages
    tilesTouched[id] = tileWidth * tileHeight;
    aabb[id] = { minPx.x, minPx.y, maxPx.x, maxPx.y };
    cov2D[id] = { cov2DMat[0], cov2DMat[1], cov2DMat[3] };
    invCov2D[id] = { invCov2DMat[0], invCov2DMat[1], invCov2DMat[3] };
    means2D[id] = pixelMean;
    depths[id] = viewMean.z;
}

__global__ void duplicateWithKeys(
        long long numGaussians,
        float2 *means2D,
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

    float4 bb = aabb[id];

    int2 tMin = { max(0, min(xTiles, (int)(bb.x / TILE_SIZE))), max(0, min(yTiles, (int)(bb.y / TILE_SIZE))) };
    int2 tMax = { max(0, min(xTiles, (int)((bb.z + TILE_SIZE - 1) / TILE_SIZE))), max(0, min(yTiles, (int)((bb.w + TILE_SIZE - 1) / TILE_SIZE))) };

    uint32_t offset = gaussianOffsets[id];
    uint32_t depthAsInt = __float_as_uint(depths[id]);
    uint32_t written = 0;

    for (int y = tMin.y; y < tMax.y; y++)
    {
        for (int x = tMin.x; x < tMax.x; x++)
        {
            uint32_t tileId = y * xTiles + x;
            gaussianKeys[offset + written] = ((uint64_t)tileId << 32) | depthAsInt;
            gaussianIndices[offset + (written++)] = id;
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
    if (id >= totalTilesTouched)
        return;
    
    uint64_t tileId = gaussianKeys[id] >> 32;
    if (id == 0)
        tileRanges[tileId].x = 0;
    else {
        uint64_t prevTileId = gaussianKeys[id - 1] >> 32;
        if (tileId != prevTileId)
        {
            tileRanges[prevTileId].y = id;
            tileRanges[tileId].x = id;
        }
    }

    if (id == totalTilesTouched - 1)
        tileRanges[tileId].y = totalTilesTouched;

}

__global__ void rasterize(
        long long numGaussians,
        float2 *means2D,
        float3 *invCov2D,
        float *opacities,
        float3 *colors,
        uint64_t *gaussianIndices,
        int2 *tileRanges,
        uint8_t *image,
        int xTiles, int yTiles,
        int width, int height)
{
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;
    int tileIndex = tileY * xTiles + tileX;
    
    int px = tileX * TILE_SIZE + threadIdx.x;
    int py = tileY * TILE_SIZE + threadIdx.y;
    bool inside = (px < width) && (py < height);
    
    int2 range = tileRanges[tileIndex];
    
    float3 pixelColor = {0.f, 0.f, 0.f};
    float T = 1.f;
    
    for (int i = range.x; i < range.y; i++) {
        if (!inside) break;
        
        uint32_t index = gaussianIndices[i];
        float2 mean = means2D[index];
        float3 invCov = invCov2D[index];
        float3 color = colors[index];
        float opacity = opacities[index];
        
        float dx = (float)px - mean.x;
        float dy = (float)py - mean.y;
        
        float power = -0.5f * (dx * dx * invCov.x 
                             + 2.f * dx * dy * invCov.y 
                             + dy * dy * invCov.z);
        if (power > 0.f) continue;  // numerical safety
        
        float g = expf(power);
        float alpha = fminf(0.99f, g * opacity);
        if (alpha < 1.f / 255.f) continue;
        
        pixelColor.x += T * alpha * color.x;
        pixelColor.y += T * alpha * color.y;
        pixelColor.z += T * alpha * color.z;
        T *= (1.f - alpha);
        
        if (T < 1e-4f) break;
    }

    if (inside) {
        int idx = (py * width + px) * 3;
        image[idx + 0] = (uint8_t)fminf(fmaxf(pixelColor.x * 255.f, 0.f), 255.f);
        image[idx + 1] = (uint8_t)fminf(fmaxf(pixelColor.y * 255.f, 0.f), 255.f);
        image[idx + 2] = (uint8_t)fminf(fmaxf(pixelColor.z * 255.f, 0.f), 255.f);
    }
}

__host__ __device__ bool isCulled(float3 viewMean, float scaleMax, float zNear, float zFar, plane *clipPlane) 
{
    // Clip against far and near clipping planes
    if (viewMean.z <= zNear || viewMean.z >= zFar)
            return false;
    return isClipped(clipPlane[0], viewMean, scaleMax)
        || isClipped(clipPlane[1], viewMean, scaleMax)
        || isClipped(clipPlane[2], viewMean, scaleMax)
        || isClipped(clipPlane[3], viewMean, scaleMax);
}
__host__ __device__ bool isClipped(plane clipPlane, float3 viewMean, float scaleMax)
{
    float signedDistanceToPlane = dot(clipPlane.normal, viewMean);
    return signedDistanceToPlane > -scaleMax;
}
