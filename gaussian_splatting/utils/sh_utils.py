import numpy as np

# SH basis constants
C0 = 0.28209479177387814    # 1 / (2*sqrt(pi))
C1 = 0.4886025119029199     # sqrt(3) / (2*sqrt(pi))
C2 = [
    1.0925484305920792,     # sqrt(15) / (2*sqrt(pi))
    -1.0925484305920792,
    0.31539156525252005,    # sqrt(5) / (4*sqrt(pi))
    -1.0925484305920792,
    0.5462742152960396,     # sqrt(15) / (4*sqrt(pi))
]
C3 = [
    -0.5900435899266435,    # -sqrt(35/32) / sqrt(pi)
    2.890611442640554,      # sqrt(105) / (2*sqrt(pi))
    -0.4570457994644658,    # -sqrt(21/32) / sqrt(pi)
    0.3731763325901154,     # sqrt(7) / (4*sqrt(pi))
    -0.4570457994644658,
    1.4453057213903705,     # sqrt(105/16) / sqrt(pi)
    -0.5900435899266435,
]

# Conversion factor: RGB <-> SH DC coefficient (degree 0)
RGB2SH = 1.0 / (2.0 * np.sqrt(np.pi)) * 2.0   # = C0 * 2 ... actually:
# DC SH contribution = C0 * dc_coeff, so dc_coeff = color / C0
RGB2SH = 1.0 / C0                              # coefficient to store in dc for a given RGB
SH2RGB = C0                                    # multiply stored dc by this to get the RGB contribution


def eval_sh(deg, sh, dirs):
    """Evaluate spherical harmonics for degree up to `deg` for batch of directions.

    Args:
        deg: active SH degree (0, 1, 2, or 3)
        sh: SH coefficients, shape (N, (deg+1)^2, 3)  — all degrees concatenated
        dirs: view directions (world space, from Gaussian toward camera), shape (N, 3)

    Returns:
        RGB color, shape (N, 3). Not yet clamped.
    """
    assert deg <= 3
    assert sh.shape[1] >= (deg + 1) ** 2

    result = C0 * sh[:, 0]

    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    if deg > 0:
        result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]

    if deg > 1:
        result = (
            result
            + C2[0] * xy * sh[:, 4]
            + C2[1] * yz * sh[:, 5]
            + C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
            + C2[3] * xz * sh[:, 7]
            + C2[4] * (xx - yy) * sh[:, 8]
        )

    if deg > 2:
        result = (
            result
            + C3[0] * y * (3 * xx - yy) * sh[:, 9]
            + C3[1] * xy * z * sh[:, 10]
            + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
            + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
            + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
            + C3[5] * z * (xx - yy) * sh[:, 14]
            + C3[6] * x * (xx - 3 * yy) * sh[:, 15]
        )

    return result
