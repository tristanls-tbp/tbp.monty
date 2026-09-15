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
import numpy.testing as nptest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.attention.voxel_grid import voxelize_and_bin_points


class VoxelizeAndBinPointsTest(unittest.TestCase):
    def test_points_not_n_by_3_raises_value_error(self):
        pass

    def test_weights_not_1d_raises_value_error(self):
        pass

    def test_recovers_voxels_that_points_were_generated_from(self):
        pass

    def test_returned_dataframe_rows_contain_each_points_voxel_and_weight(self):
        pass
