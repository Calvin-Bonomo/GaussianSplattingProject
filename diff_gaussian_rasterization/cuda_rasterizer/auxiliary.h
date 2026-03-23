#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Block / tile dimensions for the tile renderer.
// Each CTA covers one 16x16 pixel tile; BLOCK_SIZE threads per block.
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)  // 256

// Alpha threshold: skip Gaussians with alpha below this
#define ALPHA_THRESHOLD (1.0f / 255.0f)
// Transmittance threshold: stop accumulation when T drops below this
#define T_THRESHOLD 0.0001f

// -------------------------------------------------------------------------
// Inline device helpers
// -------------------------------------------------------------------------

/**
 * Clamp a float to [lo, hi].
 */
__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fmaxf(lo, fminf(hi, v));
}

/**
 * Convert normalised device coordinates (NDC) x in [-1, 1] to pixel index
 * for an image of `S` pixels.  Origin is at (0, 0) top-left.
 */
__device__ __forceinline__ float ndc2pix(float v, int S) {
    return ((v + 1.0f) * S - 1.0f) * 0.5f;
}

/**
 * Compute the screen-space pixel coordinate from ndc and image extent.
 */
__device__ __forceinline__ float2 ndc2screen(float2 ndc, int W, int H) {
    return {ndc2pix(ndc.x, W), ndc2pix(ndc.y, H)};
}

/**
 * Pack a tile (x, y) index and a 32-bit depth key into a single uint64
 * key suitable for CUB DeviceRadixSort.
 *
 * Layout: [tile_id (upper 32 bits)] | [depth_uint (lower 32 bits)]
 *
 * Positive IEEE-754 floats sort correctly as uint32 when interpreted as
 * unsigned integers, so we can use the raw bit pattern of the depth value
 * directly.
 */
__device__ __forceinline__ uint64_t makeKey(uint32_t tile_id, float depth) {
    uint32_t depth_uint;
    memcpy(&depth_uint, &depth, sizeof(float));
    return (static_cast<uint64_t>(tile_id) << 32) | static_cast<uint64_t>(depth_uint);
}

/**
 * Evaluate the 2D Gaussian alpha for a pixel offset (dx, dy) given the
 * conic representation (a, b, c) of the inverse covariance matrix and the
 * pre-sigmoid opacity.
 *
 * The 2D covariance Sigma = [[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]].
 * Its inverse is (1/det) * [[sigma_yy, -sigma_xy], [-sigma_xy, sigma_xx]].
 * We store: conic = (a, b, c) = (sigma_yy/det, -sigma_xy/det, sigma_xx/det).
 *
 * Power = -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
 *
 * Returns alpha = opacity * exp(power), or -1 if power > 0 (behind centre).
 */
__device__ __forceinline__ float computeAlpha(
    float dx, float dy,
    float a, float b, float c,  // conic (inv 2D cov)
    float opacity               // already sigmoid-activated
) {
    float power = -0.5f * (a * dx * dx + 2.0f * b * dx * dy + c * dy * dy);
    if (power > 0.0f) return -1.0f;
    return fminf(0.99f, opacity * __expf(power));
}

/**
 * Evaluate degree-0 SH color (constant).
 * sh[0..2] = (R, G, B) DC coefficients.
 */
__device__ __forceinline__ float3 evalSH0(const float* sh) {
    const float C0 = 0.28209479177387814f;
    return {C0 * sh[0] + 0.5f, C0 * sh[1] + 0.5f, C0 * sh[2] + 0.5f};
}

/**
 * Compute the 3D covariance matrix (upper triangle, 6 floats) from a
 * quaternion q = (w, x, y, z) and log-scale s = (sx, sy, sz).
 *
 * R = rotation_matrix(q)
 * S = diag(exp(s))
 * L = R @ S
 * Sigma = L @ L^T  => store upper triangle
 */
__device__ __forceinline__ void computeCov3D(
    const float3 scale,       // (sx, sy, sz) already exp()-activated
    float mod,                 // scale modifier
    const float4 rot,          // quaternion (w, x, y, z) normalised
    float* cov3D               // output: 6 floats (upper triangle)
) {
    // Normalise quaternion
    float qw = rot.x, qx = rot.y, qy = rot.z, qz = rot.w;
    float len = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz);
    qw /= len; qx /= len; qy /= len; qz /= len;

    // Build rotation matrix R from quaternion
    float R[3][3];
    R[0][0] = 1.f - 2.f*(qy*qy + qz*qz);
    R[0][1] = 2.f*(qx*qy - qw*qz);
    R[0][2] = 2.f*(qx*qz + qw*qy);
    R[1][0] = 2.f*(qx*qy + qw*qz);
    R[1][1] = 1.f - 2.f*(qx*qx + qz*qz);
    R[1][2] = 2.f*(qy*qz - qw*qx);
    R[2][0] = 2.f*(qx*qz - qw*qy);
    R[2][1] = 2.f*(qy*qz + qw*qx);
    R[2][2] = 1.f - 2.f*(qx*qx + qy*qy);

    // L = R * diag(s * mod)
    float sx = mod * scale.x, sy = mod * scale.y, sz = mod * scale.z;

    // Sigma = L @ L^T = R @ S^2 @ R^T
    // Expand: Sigma_ij = sum_k L_ik * L_jk = sum_k R_ik*s_k * R_jk*s_k
    // Only need upper triangle
    cov3D[0] = R[0][0]*R[0][0]*sx*sx + R[0][1]*R[0][1]*sy*sy + R[0][2]*R[0][2]*sz*sz;
    cov3D[1] = R[0][0]*R[1][0]*sx*sx + R[0][1]*R[1][1]*sy*sy + R[0][2]*R[1][2]*sz*sz;
    cov3D[2] = R[0][0]*R[2][0]*sx*sx + R[0][1]*R[2][1]*sy*sy + R[0][2]*R[2][2]*sz*sz;
    cov3D[3] = R[1][0]*R[1][0]*sx*sx + R[1][1]*R[1][1]*sy*sy + R[1][2]*R[1][2]*sz*sz;
    cov3D[4] = R[1][0]*R[2][0]*sx*sx + R[1][1]*R[2][1]*sy*sy + R[1][2]*R[2][2]*sz*sz;
    cov3D[5] = R[2][0]*R[2][0]*sx*sx + R[2][1]*R[2][1]*sy*sy + R[2][2]*R[2][2]*sz*sz;
}

/**
 * Project a 3D covariance (6-float upper triangle) to 2D using the EWA
 * splatting approximation.
 *
 * t       = point in view/camera space (x, y, z)
 * J       = Jacobian of perspective projection at t
 * W       = upper-left 3x3 of the view matrix (world->camera rotation)
 * cov3D   = upper triangle of world-space 3D Gaussian covariance
 * cov2D   = output: [a, b, c] of 2D covariance [[a,b],[b,c]]
 *
 * Returns false if the conic is degenerate (det <= 0).
 */
__device__ __forceinline__ bool computeCov2D(
    const float3 t,            // point in camera space
    float focal_x, float focal_y,
    const float* viewmatrix,   // column-major 4x4
    const float* cov3D,
    float3& cov2D              // output: (a, b, c) of 2x2 cov matrix
) {
    // Jacobian of perspective projection (approximate — ignores the z derivatives
    // of the off-diagonal that come from the full affine camera model)
    float tz = t.z;
    float tz2 = tz * tz;
    float J[2][3] = {
        { focal_x / tz,  0.f,          -(focal_x * t.x) / tz2 },
        { 0.f,           focal_y / tz, -(focal_y * t.y) / tz2 },
    };

    // W = upper-left 3x3 of the view matrix (column-major storage)
    float W[3][3] = {
        {viewmatrix[0], viewmatrix[1], viewmatrix[2]},
        {viewmatrix[4], viewmatrix[5], viewmatrix[6]},
        {viewmatrix[8], viewmatrix[9], viewmatrix[10]},
    };

    // T = J * W  (2x3)
    float T[2][3];
    for (int i = 0; i < 2; i++)
        for (int k = 0; k < 3; k++) {
            T[i][k] = 0.f;
            for (int j = 0; j < 3; j++)
                T[i][k] += J[i][j] * W[j][k];
        }

    // Sigma_3D (full symmetric 3x3 from upper triangle)
    float S3[3][3] = {
        {cov3D[0], cov3D[1], cov3D[2]},
        {cov3D[1], cov3D[3], cov3D[4]},
        {cov3D[2], cov3D[4], cov3D[5]},
    };

    // Sigma_2D = T * Sigma_3D * T^T  (2x2)
    // First: M = T * Sigma_3D  (2x3)
    float M[2][3] = {};
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                M[i][j] += T[i][k] * S3[k][j];

    // Then: result = M * T^T  (2x2)
    float a = 0.f, b = 0.f, c = 0.f;
    for (int k = 0; k < 3; k++) {
        a += M[0][k] * T[0][k];
        b += M[0][k] * T[1][k];
        c += M[1][k] * T[1][k];
    }

    // Add low-pass filter (0.3 pixel) to avoid degenerate Gaussians
    a += 0.3f;
    c += 0.3f;

    cov2D = {a, b, c};
    return true;
}

/**
 * Compute conic (inverse 2D covariance) from 2D cov (a, b, c).
 * Returns false if degenerate (det <= 0).
 */
__device__ __forceinline__ bool computeConic(float a, float b, float c, float3& conic) {
    float det = a * c - b * b;
    if (det <= 0.0f) return false;
    float inv_det = 1.0f / det;
    conic = {c * inv_det, -b * inv_det, a * inv_det};
    return true;
}
