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
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.attention.voxel_grid import Voxel, voxelize_and_bin_points
from tests.unit.attention import strategies


@dataclass
class VoxelizedAndBinnedPoints:
    """Points built inside known voxels, and which voxel each went into.

    Voxel k is the k-th distinct voxel met walking along ``points``, so the
    voxels are already in the order ``voxelize_and_bin_points`` reports them.
    """

    voxel_size: float
    points: np.ndarray
    point_ind_to_voxel: list[Voxel]
    voxel_to_point_inds: dict[Voxel, list[int]]
    features: dict[str, np.ndarray]


@st.composite
def voxelized_and_binned_points(
    draw: st.DrawFn,
    voxel_size_strategy: st.SearchStrategy[float] = strategies.voxel_sizes,
    weights_strategy: Callable[[int], st.SearchStrategy] | None = None,
) -> VoxelizedAndBinnedPoints:
    """Construct a set of points that are known to lie inside specific voxels.

    Strategy overview
      1. Select voxels that will be occupied.
      2. Assign the number of points that will fall into each voxel.
      3. For each voxel, generate the points that fall inside it.

    Returns:
        points: The generated points inside the voxels.
    """
    voxel_size = draw(voxel_size_strategy)

    # 1. Select voxels that will be occupied. Since voxel coordinates depend on
    #    voxel sizes, we have to scale min/max voxel coordinates to make sure all
    #    points will fall in a voxel.
    min_voxel_coord = int(-strategies.MAX_POINT_COORDINATE / voxel_size)
    max_voxel_coord = int(strategies.MAX_POINT_COORDINATE / voxel_size)
    voxel_axis_length = max_voxel_coord - min_voxel_coord + 1

    min_occupied_voxels = 1
    max_occupied_voxels = min(voxel_axis_length**3, strategies.MAX_VOXELS)
    voxels = draw(
        st.lists(
            st.tuples(
                st.integers(
                    min_value=min_voxel_coord,
                    max_value=max_voxel_coord,
                ),
                st.integers(
                    min_value=min_voxel_coord,
                    max_value=max_voxel_coord,
                ),
                st.integers(
                    min_value=min_voxel_coord,
                    max_value=max_voxel_coord,
                ),
            ),
            min_size=min_occupied_voxels,
            max_size=max_occupied_voxels,
            unique=True,
        )
    )

    # 2. Assign the number of points that will fall into each voxel.
    points_per_voxel = draw(
        st.lists(
            st.integers(min_value=1, max_value=strategies.MAX_POINTS_PER_VOXEL),
            min_size=len(voxels),
            max_size=len(voxels),
        )
    )

    # 3. Generate the points inside each voxel and voxel-to-point mapping.
    points: list[npt.NDArray[np.floating]] = []
    point_ind_to_voxel: list[Voxel] = []
    voxel_to_point_inds: dict[Voxel, list[int]] = defaultdict(list)
    for voxel_ind, voxel in enumerate(voxels):
        for _ in range(points_per_voxel[voxel_ind]):
            voxel_offsets = draw(
                arrays(
                    dtype=np.float64,
                    shape=(3,),
                    elements=st.floats(
                        min_value=strategies.VOXEL_EDGE_TOLERANCE,
                        max_value=1 - strategies.VOXEL_EDGE_TOLERANCE,
                        exclude_max=True,
                    ),
                )
            )
            point = (np.array(voxel, dtype=float) + voxel_offsets) * voxel_size
            point_ind = len(points)
            points.append(point)
            point_ind_to_voxel.append(voxel)
            voxel_to_point_inds[voxel].append(point_ind)

    # Make sure this construction is correct.
    for point_ind, voxel in enumerate(point_ind_to_voxel):
        assert point_ind in voxel_to_point_inds[voxel]
    for voxel, point_inds in voxel_to_point_inds.items():
        for point_ind in point_inds:
            assert point_ind_to_voxel[point_ind] == voxel

    # 4. One value per point for each requested feature.
    features = {
        name: draw(make_strategy(len(points)))
        for name, make_strategy in (feature_strategies or {}).items()
    }

    return VoxelizedAndBinnedPoints(
        voxel_size=voxel_size,
        points=np.stack(points),
        point_ind_to_voxel=point_ind_to_voxel,
        voxel_to_point_inds=voxel_to_point_inds,
        features=features,
    )


class VoxelizeAndBinPointsTest(unittest.TestCase):
    def test_points_not_n_by_3_raises_value_error(self):
        pass

    def test_weights_not_1d_raises_value_error(self):
        pass

    def test_recovers_voxels_that_points_were_generated_from(self):
        pass

    def test_returned_dataframe_rows_contain_each_points_voxel_and_weight(self):
        pass
