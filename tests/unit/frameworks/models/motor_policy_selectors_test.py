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
from unittest.mock import Mock, patch

from tbp.monty.cmp import Goal
from tbp.monty.frameworks.models.motor_policy_selectors import (
    DistantPolicySelector,
    SinglePolicySelector,
    highest_confidence_goal,
)


class HighestConfidenceGoalTest(unittest.TestCase):
    def test_returns_goal_with_highest_confidence(self):
        best_goal = Mock(confidence=0.9)
        second_best_goal = Mock(confidence=0.8)
        goals = [best_goal, second_best_goal]
        goal = highest_confidence_goal(goals)
        self.assertIs(goal, best_goal)

    def test_returns_first_goal_when_confidence_tied(self):
        first_goal = Mock(confidence=0.9)
        second_goal = Mock(confidence=0.9)
        goals = [first_goal, second_goal]
        goal = highest_confidence_goal(goals)
        self.assertIs(goal, first_goal)


class SinglePolicySelectorTest(unittest.TestCase):
    def setUp(self):
        self.policy = Mock()
        self.selector = SinglePolicySelector(self.policy)
        self.ctx = Mock()
        self.observations = Mock()
        self.state = Mock()
        self.percept = Mock()
        self.expected_result = Mock()
        self.policy.return_value = self.expected_result

    def test_delegates_to_configured_policy(self):
        self.selector(self.ctx, self.observations, self.state, self.percept, [])
        self.policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            None,
        )

    def test_returns_result_from_policy(self):
        result = self.selector(
            self.ctx, self.observations, self.state, self.percept, []
        )
        self.assertIs(result, self.expected_result)

    @patch("tbp.monty.frameworks.models.motor_policy_selectors.highest_confidence_goal")
    def test_calls_policy_with_goal_of_highest_confidence(
        self,
        highest_confidence_goal_mock: Mock,
    ) -> None:
        best_goal = Mock()
        goals = Mock()
        highest_confidence_goal_mock.return_value = best_goal

        self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        highest_confidence_goal_mock.assert_called_once_with(goals)
        self.policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            best_goal,
        )

    def test_pre_episode_calls_pre_episode_on_policy(self):
        motor_system = Mock()
        self.selector.pre_episode(motor_system)
        self.policy.pre_episode.assert_called_once_with(motor_system)

    def test_state_dict_includes_policy_state_dict(self):
        state_dict = Mock()
        self.policy.state_dict.return_value = state_dict
        self.assertIs(self.selector.state_dict()["policy"], state_dict)


class DistantPolicySelectorTest(unittest.TestCase):
    def setUp(self):
        self.jump_to_goal = Mock()
        self.look_at_goal = Mock()
        self.default_policy = Mock()
        self.selector = DistantPolicySelector(
            self.jump_to_goal, self.look_at_goal, self.default_policy
        )
        self.ctx = Mock()
        self.observations = Mock()
        self.state = Mock()
        self.percept = Mock()
        # self.goals = [Mock(confidence=0.9), Mock(confidence=0.8)]

    def test_returns_default_policy_result_when_no_goals_are_present(self):
        default_policy_result = Mock()
        self.default_policy.return_value = default_policy_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            [],
        )

        self.default_policy.assert_called_once_with(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            None,
        )
        self.assertIs(result, default_policy_result)

    def test_returns_jump_to_goal_result_when_gsg_goal_is_present(self):
        gsg_goal = Mock(sender_type="GSG")
        goals = [
            Mock(sender_type="SM"),
            gsg_goal,
            Mock(sender_type="SM"),
        ]
        jump_to_goal_result = Mock()
        self.jump_to_goal.return_value = jump_to_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        self.jump_to_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, gsg_goal
        )
        self.assertIs(result, jump_to_goal_result)

    @patch("tbp.monty.frameworks.models.motor_policy_selectors.highest_confidence_goal")
    def test_invokes_jump_to_goal_with_highest_confidence_gsg_goal(
        self, highest_confidence_goal_mock: Mock
    ):
        best_gsg_goal = Mock(sender_type="GSG")
        gsg_goal = Mock(sender_type="GSG")
        goals = [
            Mock(sender_type="SM"),
            gsg_goal,
            best_gsg_goal,
            Mock(sender_type="SM"),
        ]
        highest_confidence_goal_mock.return_value = best_gsg_goal
        jump_to_goal_result = Mock()
        self.jump_to_goal.return_value = jump_to_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        highest_confidence_goal_mock.assert_called_once_with(
            Goals([gsg_goal, best_gsg_goal])
        )
        self.jump_to_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, best_gsg_goal
        )
        self.assertIs(result, jump_to_goal_result)

    def test_returns_look_at_goal_result_when_only_sm_goals_are_present(self):
        goals = [
            Mock(sender_type="SM", confidence=0.9),
            Mock(sender_type="SM", confidence=0.8),
        ]
        look_at_goal_result = Mock()
        self.look_at_goal.return_value = look_at_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        self.look_at_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, goals[0]
        )
        self.assertIs(result, look_at_goal_result)

    @patch("tbp.monty.frameworks.models.motor_policy_selectors.highest_confidence_goal")
    def test_invokes_look_at_goal_with_highest_confidence_gsg_goal(
        self, highest_confidence_goal_mock: Mock
    ):
        best_sm_goal = Mock(sender_type="SM")
        sm_goal = Mock(sender_type="SM")
        goals = [
            sm_goal,
            best_sm_goal,
        ]
        highest_confidence_goal_mock.return_value = best_sm_goal
        look_at_goal_result = Mock()
        self.look_at_goal.return_value = look_at_goal_result

        result = self.selector(
            self.ctx,
            self.observations,
            self.state,
            self.percept,
            goals,
        )

        highest_confidence_goal_mock.assert_called_once_with(
            Goals([sm_goal, best_sm_goal])
        )
        self.look_at_goal.assert_called_once_with(
            self.ctx, self.observations, self.state, self.percept, best_sm_goal
        )
        self.assertIs(result, look_at_goal_result)

    def test_checks_jump_to_goal_checked_after_a_jump(self):
        self.fail("Not implemented")


class Goals:  # noqa: PLW1641
    def __init__(self, goals: list[Goal]):
        self.goals = goals

    def __eq__(self, other: object) -> bool:
        if not hasattr(other, "__iter__"):
            return False
        return set(self.goals) == set(other)
