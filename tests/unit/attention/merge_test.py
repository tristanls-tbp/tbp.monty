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

from hypothesis import given

from tbp.monty.attention.merge import Union
from tbp.monty.attention.voxel_grid import VoxelGrid
from tests.unit.attention import strategies


class UnionTest(unittest.TestCase):
    def setUp(self):
        self.union = Union()

    @given(grid_a=strategies.default_voxel_grid())
    def test_raises_value_error_if_voxel_sizes_do_not_match(self, grid_a: VoxelGrid):
        grid_b = VoxelGrid.from_pandas(grid_a.voxel_size + 1, grid_a.to_pandas())
        with self.assertRaisesRegex(ValueError, "Voxel sizes must match for merging."):
            self.union(grid_a, grid_b)

    # disjoint, full overlap, partial overlap
