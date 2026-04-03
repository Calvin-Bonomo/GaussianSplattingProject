import numpy as np

from ..math_utils import quat_to_mat


def test_identity_quaternion():
    # [0, 0, 0, 1] should produce the identity matrix
    q = np.array([0.0, 0.0, 0.0, 1.0])
    result = quat_to_mat(q)
    np.testing.assert_allclose(result, np.eye(3), atol=1e-7)


def test_output_is_valid_rotation_matrix():
    # A random quaternion should produce an orthogonal matrix with det = 1
    q = np.array([0.1, 0.2, 0.3, 0.4])
    result = quat_to_mat(q)
    np.testing.assert_allclose(result @ result.T, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(np.linalg.det(result), 1.0, atol=1e-7)


def test_unnormalized_quaternion():
    # Should normalize internally and still return identity for a scaled [0,0,0,1]
    q = np.array([0.0, 0.0, 0.0, 5.0])
    result = quat_to_mat(q)
    np.testing.assert_allclose(result, np.eye(3), atol=1e-7)
