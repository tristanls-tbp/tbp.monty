# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from typing_extensions import Self

from tbp.monty.cmp import Goal, Message
from tbp.monty.frameworks.models.motor_system_state import AgentState
from tbp.monty.observations import SensorObservation

__all__ = [
    "Payload",
    "Transform",
    "TransformContext",
]


@dataclass
class Payload:
    observation: SensorObservation
    percept: Message | None
    goals: list[Goal]


@dataclass
class TransformContext:
    rng: np.random.RandomState
    state: AgentState | None = None
    motor_only_step: bool = False


class Transform(Protocol):
    """A transform that can be applied to a payload."""

    def __call__(
        self: Self,
        ctx: TransformContext,
        payload: Payload,
    ) -> Payload:
        """Apply the transform to the payload.

        Args:
            ctx: The transform context.
            payload: The payload to transform.

        Returns:
            A payload with the transformed observation, percept, and goals.
        """
        ...


class NoDepthSensorPresent(RuntimeError):
    """Raised when a depth sensor is expected but not found."""

    pass
