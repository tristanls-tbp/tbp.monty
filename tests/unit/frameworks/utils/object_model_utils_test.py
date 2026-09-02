# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.utils.object_model_utils import (
    orthonormal_pose_vectors,
    pose_vector_mean,
    pose_vector_merge,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    normalize,
    project_onto_tangent_plane,
)
from tbp.monty.geometry import Rotation
from tbp.monty.math import DEFAULT_TOLERANCE


def pose_frame(
    normal: npt.NDArray[np.float64], tangent_seed: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return flat right-handed orthonormal pose vectors.

    Args:
        normal: Surface normal direction.
        tangent_seed: Direction to derive the first curvature direction from.

    Returns:
        Flat array of nine elements holding the surface normal and the two curvature
        directions.
    """
    normal = normalize(normal)
    tangent = project_onto_tangent_plane(tangent_seed, normal)
    cd1 = normalize(tangent)
    return np.hstack([normal, cd1, np.cross(normal, cd1)])


class PoseVectorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.normal = np.array([0.0, 0.0, 1.0])
        self.cd1 = np.array([1.0, 0.0, 0.0])
        self.frame = np.hstack([self.normal, self.cd1, np.cross(self.normal, self.cd1)])
        self.opposite_frame = np.hstack(
            [-self.normal, self.cd1, np.cross(-self.normal, self.cd1)]
        )

    def assert_is_rotation(self, pose_vecs: npt.NDArray[np.float64], msg: str) -> None:
        """Assert that the pose vectors can be read as a rotation.

        In other words, that the surface normal and the two principal curvatures form an
        orthonormal, right-handed basis.

        Args:
            pose_vecs: Flat array of nine elements holding the pose vectors.
            msg: Assertion message.
        """
        matrix = np.array(pose_vecs).reshape((3, 3))
        self.assertAlmostEqual(np.linalg.det(matrix), 1.0, places=6, msg=msg)
        np.testing.assert_allclose(
            matrix @ matrix.T, np.identity(3), atol=1e-6, err_msg=msg
        )
        # Raises for non-positive determinant on scipy > 1.14.1
        Rotation.from_matrix(matrix)

    def test_orthonormal_pose_vectors_orthgonalizes_curvature_direction(self) -> None:
        pose_vecs = orthonormal_pose_vectors(self.normal, np.array([1.0, 0.0, 0.7]))
        self.assert_is_rotation(pose_vecs, "Pose vectors are not a rotation")
        np.testing.assert_allclose(pose_vecs[:3], self.normal, atol=DEFAULT_TOLERANCE)

    def test_orthonormal_pose_vectors_falls_back_to_arbitrary_direction_if_curvature_direction_is_parallel_to_surface_normal(  # noqa: E501
        self,
    ) -> None:
        pose_vecs = orthonormal_pose_vectors(self.normal, self.normal * 2.0)
        self.assert_is_rotation(
            pose_vecs, "Parallel curvature direction did not fall back"
        )

    def test_pose_vector_mean_of_spread_observations_is_a_rotation(self) -> None:
        rng = np.random.RandomState(0)
        for _ in range(50):
            observations = np.array(
                [
                    pose_frame(self.normal + rng.normal(0, 0.3, 3), rng.normal(size=3))
                    for _ in range(6)
                ]
            )
            pv_mean, _ = pose_vector_mean(observations, np.ones((6, 1)))
            self.assert_is_rotation(
                pv_mean, "Mean of spread observations is not a rotation"
            )

    def test_pose_vector_merge_of_opposite_curvature_directions_is_a_rotation(
        self,
    ) -> None:
        merged = pose_vector_merge(
            self.opposite_frame,
            self.frame,
            use_cds_to_update=True,
            num_new_obs=4,
            num_previous_obs=4,
        )
        self.assert_is_rotation(merged, "Merged pose vectors are not a rotation")

    def test_pose_vector_merge_keeps_the_surface_side_with_more_observations(
        self,
    ) -> None:
        keeps_new = pose_vector_merge(
            self.opposite_frame,
            self.frame,
            use_cds_to_update=True,
            num_new_obs=8,
            num_previous_obs=4,
        )
        np.testing.assert_allclose(keeps_new[:3], -self.normal, atol=DEFAULT_TOLERANCE)
        keeps_previous = pose_vector_merge(
            self.opposite_frame,
            self.frame,
            use_cds_to_update=True,
            num_new_obs=4,
            num_previous_obs=8,
        )
        np.testing.assert_allclose(
            keeps_previous[:3], self.normal, atol=DEFAULT_TOLERANCE
        )

    def test_pose_vector_merge_averages_normals_on_the_same_surface_side(self) -> None:
        tilted = pose_frame(np.array([0.0, 0.5, 1.0]), self.cd1)
        merged = pose_vector_merge(
            tilted,
            self.frame,
            use_cds_to_update=True,
            num_new_obs=4,
            num_previous_obs=4,
        )
        self.assert_is_rotation(merged, "Merged pose vectors are not a rotation")
        self.assertGreater(np.dot(merged[:3], self.normal), 0.0)

    def test_pose_vector_merge_repeated_merges_from_the_opposite_side_stay_a_rotation(
        self,
    ) -> None:
        stored = self.frame.copy()
        for update in range(1, 9):
            stored = pose_vector_merge(
                self.opposite_frame,
                stored,
                use_cds_to_update=True,
                num_new_obs=4,
                num_previous_obs=4 * update,
            )
            self.assert_is_rotation(
                stored,
                "Repeated merge from the opposite side is not a rotation after "
                f"{update} updates",
            )

    def test_repeated_merges_of_noisy_observations_stay_a_rotation(self) -> None:
        rng = np.random.RandomState(0)
        stored, observation_count = self.frame.copy(), 4
        for _ in range(50):
            observations = np.array(
                [
                    pose_frame(
                        self.normal + rng.normal(0, 0.3, 3),
                        self.cd1 * (1.0 if rng.rand() < 0.5 else -1.0),
                    )
                    for _ in range(4)
                ]
            )
            pv_mean, use_cds_to_update = pose_vector_mean(observations, np.ones((4, 1)))
            stored = pose_vector_merge(
                pv_mean,
                stored,
                use_cds_to_update=use_cds_to_update,
                num_new_obs=4,
                num_previous_obs=observation_count,
            )
            observation_count += 4
            self.assert_is_rotation(stored, "Merged frame is not a rotation")
