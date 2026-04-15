# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    JumpToGoal,
    MotorPolicy,
    MotorPolicyResult,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.motor_system import MotorSystem
    from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal


__all__ = [
    "MotorPolicySelector",
    "SinglePolicySelector",
    "highest_confidence_goal",
]


def highest_confidence_goal(goals: list[Goal]) -> Goal:
    """Return the goal with the highest confidence.

    If there are multiple goals with the same confidence, returns the first one.

    Args:
        goals: A list of goals. Must be non-empty.

    Returns:
        The goal with the highest confidence.

    """
    return sorted(goals, key=lambda x: x.confidence, reverse=True)[0]


class MotorPolicySelector(Protocol):
    def pre_episode(self, motor_system: MotorSystem) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: list[Goal],
    ) -> MotorPolicyResult: ...


class SinglePolicySelector(MotorPolicySelector):
    def __init__(self, policy: MotorPolicy):
        self._policy = policy
        # TODO: Get rid of this once we have another path for telemetry.
        self._selected_goals: list[Goal | None] = []

    def pre_episode(self, motor_system: MotorSystem) -> None:
        self._policy.pre_episode(motor_system)
        self._selected_goals = []

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self._policy.state_dict(),
            "selected_goals": self._selected_goals,
        }

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: list[Goal],
    ) -> MotorPolicyResult:
        goal = highest_confidence_goal(goals) if goals else None
        self._selected_goals.append(goal)
        return self._policy(ctx, observations, state, percept, goal)

class DistantPolicySelector(MotorPolicySelector):
    def __init__(
        self, jump_to_goal: JumpToGoal, look_at_goal: LookAtGoal, default: MotorPolicy
    ):
        self._is_jumping = False
        self._jump_to_goal = jump_to_goal
        self._look_at_goal = look_at_goal
        self._default = default
        self._selected_goals: list[Goal | None] = []

    def pre_episode(self, motor_system: MotorSystem) -> None:
        self._jump_to_goal.pre_episode(motor_system)
        self._look_at_goal.pre_episode(motor_system)
        self._default.pre_episode(motor_system)

    def state_dict(self) -> dict[str, Any]:
        return {
            "jump_to_goal": self._jump_to_goal.state_dict(),
            "look_at_goal": self._look_at_goal.state_dict(),
            "default": self._default.state_dict(),
        }

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: list[Goal],
    ) -> MotorPolicyResult:
        gsg_goals = []
        sm_goals = []
        for goal in goals:
            if goal.sender_type == "GSG":
                gsg_goals.append(goal)
            elif goal.sender_type == "SM":
                sm_goals.append(goal)

        if gsg_goals:
            if self._is_jumping:
                # TODO: Reset policy jump state somehow
                pass
            goal = highest_confidence_goal(gsg_goals)
            policy = self._jump_to_goal
            self._is_jumping = True
        elif self._is_jumping:
            # TODO: Add a way to check if we should undo the jump, and reuse
            # the jump
            policy = self._jump_to_goal
            goal = None
            self._is_jumping = False
        elif sm_goals:
            goal = highest_confidence_goal(sm_goals)
            policy = self._look_at_goal
            self._is_jumping = False
        else:
            goal = None
            policy = self._default
            self._is_jumping = False

        self._selected_goals.append(goal)
        return policy(ctx, observations, state, percept, goal)


# class SurfacePolicySelector(MotorPolicySelector):
#     def __init__(self, jump_to_goal: JumpToGoal, default: MotorPolicy):
#         self._jump_to_goal = jump_to_goal
#         self._default = default
