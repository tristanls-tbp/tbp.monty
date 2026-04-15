# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing as nptest
import quaternion as qt
from hypothesis import given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    ActionJSONEncoder,
    LookDown,
    LookUp,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    JumpToGoal,
    MotorPolicyResult,
    PredefinedPolicy,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.math import VectorXYZ
from tests.unit.frameworks.models.fakes.cmp import FakeMessage


class SurfacePolicyCurvatureInformedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.policy = SurfacePolicyCurvatureInformed(
            alpha=0.1,
            pc_alpha=0.5,
            max_pc_bias_steps=32,
            min_general_steps=8,
            min_heading_steps=12,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
            desired_object_distance=0.025,
        )
        self.location = np.array([1.0, 2.0, 3.0])
        self.tangent_norm = np.array([0, 1, 0])
        self.percept = Message(
            location=self.location,
            morphological_features={
                "pose_vectors": np.array(
                    [self.tangent_norm.tolist(), [1, 0, 0], [0, 0, -1]]
                ),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "principal_curvatures_log": [0, 0.5],
                "hsv": [0, 1, 1],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        )

    def test_appends_to_tangent_locs_and_tangent_norms_if_last_action_is_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(len(self.policy.tangent_norms), 1)
        nptest.assert_array_equal(self.policy.tangent_norms[0], self.tangent_norm)

    def test_appends_none_to_tangent_norms_if_last_action_is_orient_vertical_but_no_pose_vectors_in_state(  # noqa: E501
        self,
    ):
        del self.percept.morphological_features["pose_vectors"]
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(self.policy.tangent_norms, [None])

    def test_does_not_append_to_tangent_locs_and_tangent_norms_if_last_action_is_not_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = LookUp(
            agent_id=self.agent_id, rotation_degrees=0
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(self.policy.tangent_locs, [])
        self.assertEqual(self.policy.tangent_norms, [])


class PredefinedPolicyReadActionFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.actions_file = Path(__file__).parent / "motor_policies_test_actions.jsonl"

    def test_read_action_file(self) -> None:
        # For this test, we write our own actions to a temporary file instead of
        # loading a file on disk. It's a better guarantee that we're loading the
        # actions exactly as expected.
        expected = [
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
            LookDown(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnLeft(agent_id=self.agent_id, rotation_degrees=10.0),
            LookUp(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
        ]
        with tempfile.TemporaryDirectory() as data_path:
            actions_file = Path(data_path) / "actions.jsonl"
            actions_file.write_text(
                "\n".join(json.dumps(a, cls=ActionJSONEncoder) for a in expected) + "\n"
            )
            loaded = PredefinedPolicy.read_action_file(actions_file)
            self.assertEqual(len(loaded), len(expected))
            for loaded_action, expected_action in zip(loaded, expected):
                self.assertEqual(dict(loaded_action), dict(expected_action))

    def test_cycles_continuously(self) -> None:
        policy = PredefinedPolicy(
            agent_id=self.agent_id,
            file_name=self.actions_file,
        )
        cycle_length = len(policy.action_list)
        ctx = RuntimeContext(rng=np.random.RandomState(42))
        observations = Observations()
        returned_actions: list[Action] = []
        for _ in range(2 * cycle_length):
            result = policy(ctx, observations, MotorSystemState(), FakeMessage(), None)
            assert len(result.actions) == 1, "Expected one action"
            returned_actions.append(result.actions[0])

        for i in range(cycle_length):
            first_occurrence = returned_actions[i]
            second_occurrence = returned_actions[i + cycle_length]
            self.assertEqual(first_occurrence, second_occurrence)


class JumpToGoalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.policy = JumpToGoal(self.agent_id)
        self.motor_system_state = MotorSystemState(
            {
                self.agent_id: AgentState(
                    sensors={
                        SensorID("sensor_id_0"): SensorState(
                            position=cast("VectorXYZ", (0, 0, 0)), rotation=qt.one
                        )
                    },
                    position=cast("VectorXYZ", (0, 0, 0)),
                    rotation=qt.one,
                )
            }
        )

    @given(
        goal_location=st.tuples(
            st.floats(min_value=-1, max_value=1),
            st.floats(min_value=-1, max_value=1),
            st.floats(min_value=-1, max_value=1),
        ),
        goal_direction=st.tuples(
            st.floats(min_value=-1, max_value=1),
            st.floats(min_value=-1, max_value=1),
            st.floats(min_value=-1, max_value=1),
        ),
    )
    def test_generates_actions_that_point_agent_at_goal_location_opposite_surface_normal(  # noqa: E501
        self,
        goal_location,
        goal_direction,
    ) -> None:
        goal_location = np.array(goal_location)
        goal_direction = np.array(goal_direction)
        if np.isclose(np.linalg.norm(goal_direction), 0.0):
            return
        goal_direction = goal_direction / np.linalg.norm(goal_direction)
        pose_vectors = np.zeros((3, 3))
        pose_vectors[0] = goal_direction

        goal = Goal(
            location=goal_location,
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=1.0,
            use_state=True,
            sender_id="test",
            sender_type="SM",
            goal_tolerances=None,
            info=None,
        )

        # We need a fresh policy for each iteration. setUp() is not called between
        # hypothesis iterations.
        policy = JumpToGoal(self.agent_id)
        policy_result = policy(
            ctx=Mock(),
            observations=Mock(),
            state=self.motor_system_state,
            percept=Mock(),
            goal=goal,
        )
        assert isinstance(policy_result, MotorPolicyResult)

        self.assertEqual(len(policy_result.actions), 2)
        set_agent_pose = policy_result.actions[0]
        assert isinstance(set_agent_pose, SetAgentPose)
        set_sensor_rotation = policy_result.actions[1]
        assert isinstance(set_sensor_rotation, SetSensorRotation)

        nptest.assert_array_equal(set_agent_pose.location, goal_location)
        rotation = Rotation.from_quat(
            [
                set_agent_pose.rotation_quat.x,
                set_agent_pose.rotation_quat.y,
                set_agent_pose.rotation_quat.z,
                set_agent_pose.rotation_quat.w,
            ]
        )
        new_forward_axis = -rotation.as_matrix()[:, 2]
        nptest.assert_allclose(new_forward_axis, goal_direction, atol=1e-6)

        # Note: the above is equivalent to the below code, but I think checking
        # the z-axis in the rotation matrix is more intuitive.
        # forward_axis = np.array([0, 0, -1])
        # new_forward_direction = qt.rotate_vectors(
        #     set_agent_pose.rotation_quat, forward_axis
        # )
        # nptest.assert_allclose(new_forward_direction, goal_direction, atol=1e-6)

        # Sensor rotation must be identity.
        nptest.assert_allclose(
            qt.as_float_array(set_sensor_rotation.rotation_quat),
            qt.as_float_array(qt.one),
            atol=1e-6,
        )

    def test_returns_undo_actions_if_undo_is_needed(self) -> None:
        pass

    def test_undo_actions_match_pre_jump_state(self) -> None:
        pass

    def test_returns_new_jump_actions_if_undo_is_not_needed_after_jump_and_goal_is_provided(
        self,
    ) -> None:
        pass

    def test_raises_error_if_undo_is_not_needed_after_jump_and_goal_is_None_and_not_suppressing_errors(
        self,
    ) -> None:
        pass

    def test_logs_error_if_undo_is_not_needed_after_jump_and_goal_is_None_and_suppressing_errors(
        self,
    ) -> None:
        pass
