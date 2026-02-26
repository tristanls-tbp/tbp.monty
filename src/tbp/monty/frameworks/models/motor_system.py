# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Any

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState, SensorState
from tbp.monty.frameworks.sensors import SensorID

__all__ = ["MotorSystem"]


class MotorSystem:
    """The basic motor system implementation."""

    def __init__(
        self, policy: MotorPolicy, state: MotorSystemState | None = None
    ) -> None:
        """Initialize the motor system with a motor policy.

        Args:
            policy: The motor policy to use.
            state: The initial state of the motor system.
                Defaults to None.
        """
        self._policy = policy
        self._state = state
        # For each step, we store the actions produced by the policy and the current
        # motor system state as a (actions, state) tuple.
        self._action_sequence: list[tuple[list[Action], dict[AgentID, Any] | None]] = []

    @property
    def action_sequence(self) -> list[tuple[list[Action], dict[AgentID, Any] | None]]:
        return self._action_sequence

    def pre_episode(self) -> None:
        """Pre episode hook."""
        self._policy.pre_episode()
        self._action_sequence = []

    def __call__(self, ctx: RuntimeContext, observations: Observations) -> list[Action]:
        """Defines the structure for __call__.

        Delegates to the motor policy.

        Args:
            ctx: The runtime context.
            observations: The observations from the environment.

        Returns:
            The action to take.
        """
        policy_result = self._policy(ctx, observations, self._state)

        # TODO: Must change when we have multiple agents.
        if self._state is not None:
            agent_id = self._policy.agent_id
            self._state[agent_id].motor_only_step = policy_result.motor_only_step

        state_copy = self._state.convert_motor_state() if self._state else None
        self._action_sequence.append((policy_result.actions, state_copy))
        return policy_result.actions

    def sensor_state(self, sensor_module_id: SensorID) -> SensorState:
        """Return the proprioceptive state of a sensor module.

        Args:
            sensor_module_id: The ID of the sensor module.

        Returns:
            The proprioceptive state of the sensor module.

        Raises:
            RuntimeError: If the motor system state is not set.
            ValueError: If the sensor module is not found.
        """
        if self._state is None:
            raise RuntimeError("Motor system state is not set")

        # TODO: Here is an example of why proprioceptive state should be a flat data
        #       structure rather than nested.
        for agent_id in self._state:
            agent_state = self._state[agent_id]
            if sensor_module_id in agent_state.sensors:
                return agent_state.sensors[sensor_module_id]

        raise ValueError(f"Sensor module {sensor_module_id} not found")
