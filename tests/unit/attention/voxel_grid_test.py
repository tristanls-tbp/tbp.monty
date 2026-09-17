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
from typing import Callable
from unittest.mock import patch, sentinel

import numpy as np
import numpy.testing as nptest
import numpy.typing as npt
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.attention.voxel_grid import (
    Voxel,
    VoxelGrid,
    voxelize_and_bin_points,
    voxelize_points,
)
from tests.strategies.arrays import (
    float_array_n_by_3,
    float_array_not_1d,
    float_array_not_n_by_3,
)
from tests.unit.attention import strategies


@dataclass
class VoxelizedAndBinnedPoints:
    """Points built inside known voxels, and which voxel each went into.

    Voxel k is the k-th distinct voxel met walking along ``points``, so the
    voxels are already in the order ``voxelize_and_bin_points`` reports them.
    """

    voxel_size: float
    points: npt.NDArray[np.floating]
    point_ind_to_voxel: list[Voxel]
    voxel_to_point_inds: dict[Voxel, list[int]]
    weights: npt.NDArray[np.floating]


@st.composite
def voxelized_and_binned_points(
    draw: st.DrawFn,
    voxel_size_strategy: st.SearchStrategy[float] = strategies.voxel_sizes,
    weights_strategy: Callable[
        [int], st.SearchStrategy
    ] = strategies.valid_default_attention_system_weights,
) -> VoxelizedAndBinnedPoints:
    """Construct a set of points that are known to lie inside specific voxels.

    Strategy overview
      1. Select voxels that will be occupied.
      2. Assign the number of points that will fall into each voxel.
      3. For each voxel, generate the points that fall inside it.
      4. Generate the weights for each point.

    Returns:
       Voxelized and binned points.
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
    weights = draw(weights_strategy(len(points)))

    return VoxelizedAndBinnedPoints(
        voxel_size=voxel_size,
        points=np.stack(points),
        point_ind_to_voxel=point_ind_to_voxel,
        voxel_to_point_inds=voxel_to_point_inds,
        weights=weights,
    )


class VoxelizePointsTest(unittest.TestCase):
    @given(points=float_array_not_n_by_3())
    def test_points_not_n_by_3_raises_value_error(
        self, points: npt.NDArray[np.float64]
    ):
        with self.assertRaises(ValueError):
            voxelize_points(voxel_size=1.0, points=points)

    @given(binned=voxelized_and_binned_points())
    def test_returned_tuple_list_contains_each_points_voxel(
        self,
        binned: VoxelizedAndBinnedPoints,
    ) -> None:
        result = voxelize_points(binned.voxel_size, binned.points)
        expected = binned.point_ind_to_voxel
        nptest.assert_array_equal(result, expected)


class VoxelizeAndBinPointsTest(unittest.TestCase):
    @given(points=float_array_n_by_3(), weights=float_array_not_1d())
    def test_weights_not_1d_raises_value_error(
        self, points: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
    ):
        with self.assertRaises(ValueError):
            voxelize_and_bin_points(
                voxel_size=1.0,
                points=points,
                weights=weights,
            )

    @given(binned=voxelized_and_binned_points())
    def test_returned_dataframe_rows_contain_each_points_voxel_from_voxelize_points_and_weight(  # noqa: E501
        self,
        binned: VoxelizedAndBinnedPoints,
    ):
        with patch(
            "tbp.monty.attention.voxel_grid.voxelize_points"
        ) as voxelize_points_mock:
            voxelize_points_mock.return_value = sentinel.voxels
            result = voxelize_and_bin_points(
                voxel_size=binned.voxel_size,
                points=binned.points,
                weights=binned.weights,
            )

        voxels = list(result["voxel"])
        nptest.assert_array_equal(voxels, sentinel.voxels)

        weights = result["weight"]
        nptest.assert_array_equal(weights, binned.weights)

        voxelize_points_mock.assert_called_once_with(binned.voxel_size, binned.points)


class VoxelGridTest(unittest.TestCase):
    def test_weights_at_points_returnsfill_value_for_every_point_when_voxel_grid_is_empty(  # noqa: E501
        self,
    ):
        # TODO: remove once better test is written.
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        fill_value = 8675309
        grid = VoxelGrid(voxel_size=0.1)

        result = grid.weights_at_points(points, fill_value=fill_value)

        expected = np.full(shape=(points.shape[0],), fill_value=fill_value)
        nptest.assert_array_equal(result, expected)

    def test_weights_at_points_returns_weights_for_occupied_voxels_and_fill_value_for_points_in_unoccupied_voxels(  # noqa: E501
        self,
    ):
        pass
