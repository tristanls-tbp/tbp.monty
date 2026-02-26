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
from unittest.mock import MagicMock

from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

SENSOR_ID = SensorID("sensor_0")


class SensorStateTest(unittest.TestCase):
    def test_raises_runtime_error_if_motor_system_state_is_not_set(self):
        motor_system = MotorSystem(policy=MagicMock())
        with self.assertRaises(RuntimeError):
            motor_system.sensor_state(SENSOR_ID)

    def test_raises_value_error_if_sensor_module_is_not_found(self):
        motor_system = MotorSystem(
            policy=MagicMock(),
            state=MotorSystemState(
                {
                    AgentID("agent_id_0"): AgentState(
                        sensors={
                            SensorID("different_sensor_id"): SensorState(
                                position=(0.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0)
                            )
                        },
                        position=(0.0, 0.0, 0.0),
                        rotation=(1.0, 0.0, 0.0, 0.0),
                    )
                }
            ),
        )
        with self.assertRaises(ValueError):
            motor_system.sensor_state(SENSOR_ID)

    def test_returns_sensor_state_if_sensor_module_is_found(self):
        sensor_state = SensorState(
            position=(1.0, 3.0, 5.0), rotation=(0.0, 1.0, 0.0, 0.0)
        )
        motor_system = MotorSystem(
            policy=MagicMock(),
            state=MotorSystemState(
                {
                    AgentID("agent_id_1"): AgentState(
                        sensors={},
                        position=(0.0, 0.0, 0.0),
                        rotation=(1.0, 0.0, 0.0, 0.0),
                    ),
                    AgentID("agent_id_0"): AgentState(
                        sensors={SENSOR_ID: sensor_state},
                        position=(0.0, 0.0, 0.0),
                        rotation=(1.0, 0.0, 0.0, 0.0),
                    )
                }
            ),
        )
        self.assertEqual(
            motor_system.sensor_state(SENSOR_ID),
            sensor_state
        )
