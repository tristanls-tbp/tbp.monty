# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)
import tempfile
from pathlib import Path
from typing import cast
from unittest import TestCase

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.models.object_model import GraphObjectModel
from tbp.monty.frameworks.run import main as run_serial
from tbp.monty.frameworks.run_parallel import main as run_parallel


def assert_graph_object_models_equal(
    left: GraphObjectModel, right: GraphObjectModel
) -> None:
    """Custom assertion comparison for GraphObjectModel.

    Args:
        left: The left GraphObjectModel instance to compare with.
        right: The right GraphObjectModel instance to compare with.

    Raises:
        AssertionError: If the two GraphObjectModel instances are not equal.
    """
    if left._graph is None and right._graph is None:
        raise AssertionError("Both GraphObjectModel instances have no graph.")

    if left._graph is None or right._graph is None:
        raise AssertionError("One of the GraphObjectModel instances has no graph.")

    if set(left._graph.keys) != set(right._graph.keys):
        raise AssertionError(
            "The keys of the two GraphObjectModel instances are not equal.\n"
            f"Left keys: {set(left._graph.keys)}\n"
            f"Right keys: {set(right._graph.keys)}\n"
        )

    for key in left._graph.keys:
        v_left, v_right = left._graph[key], right._graph[key]

        if torch.is_tensor(v_left):
            if not torch.equal(v_left, v_right):
                raise AssertionError(
                    f"The {key} values of the two GraphObjectModel instances are not "
                    f"equal.\nLeft value: {v_left}\n"
                    f"Right value: {v_right}\n"
                )
        elif v_left != v_right:
            raise AssertionError(
                f"The {key} values of the two GraphObjectModel instances are not "
                f"equal.\nLeft value: {v_left}\n"
                f"Right value: {v_right}"
            )


class EpisodeReproducibilityTest(TestCase):
    def hydra_config(self, test_name: str, output_dir: Path) -> DictConfig:
        overrides = [
            f"experiment=test/{test_name}",
            "num_parallel=1",
            f"++experiment.config.logging.output_dir={output_dir}",
            "+experiment.config.monty_config.motor_system_config"
            ".motor_system_args.policy_args.file_name="
            f"{Path(__file__).parent.parent / 'unit' / 'resources' / 'fixed_test_actions.jsonl'}",
        ]
        return hydra.compose(config_name="experiment", overrides=overrides)

    def setUp(self):
        self.output_dir = Path(tempfile.mkdtemp())
        with hydra.initialize(version_base=None, config_path="../../conf"):
            self.training_config = self.hydra_config(
                "episode_reproducibility_training", self.output_dir
            )

    def serial_run(self, config: DictConfig):
        """Executes the experiment in serial mode."""
        OmegaConf.clear_resolvers()  # main will re-register resolvers
        run_serial(config)

    def parallel_run(self, config: DictConfig):
        """Executes the experiment in parallel mode."""
        OmegaConf.clear_resolvers()  # main will re-register resolvers
        run_parallel(config)

    def assert_trained_models_equal(self, serial_model: dict, parallel_model: dict):
        if set(parallel_model["lm_dict"].keys()) != set(serial_model["lm_dict"].keys()):
            raise AssertionError("LM IDs do not match")

        for lm_id in parallel_model["lm_dict"].keys():
            p = parallel_model["lm_dict"][lm_id]
            s = serial_model["lm_dict"][lm_id]
            if set(p.keys()) != set(s.keys()):
                raise AssertionError(f"LM {lm_id} keys do not match")

            p_graph_memory = p["graph_memory"]
            s_graph_memory = s["graph_memory"]
            if set(p_graph_memory.keys()) != set(s_graph_memory.keys()):
                raise AssertionError(f"LM {lm_id} graph memory keys do not match")

            for graph_id in p_graph_memory.keys():
                p_graph = p_graph_memory[graph_id]
                s_graph = s_graph_memory[graph_id]
                if set(p_graph.keys()) != set(s_graph.keys()):
                    raise AssertionError(
                        f"LM {lm_id} graph {graph_id} keys do not match"
                    )

                for channel_id in p_graph.keys():
                    p_graph_data: GraphObjectModel = cast(
                        "GraphObjectModel", p_graph[channel_id]
                    )
                    s_graph_data: GraphObjectModel = cast(
                        "GraphObjectModel", s_graph[channel_id]
                    )
                    assert_graph_object_models_equal(p_graph_data, s_graph_data)

    def test_training_episodes_are_equal(self):
        config = self.training_config

        self.serial_run(config)
        serial_model_path = (
            Path(config.experiment.config.logging.output_dir)
            / "pretrained"
            / "model.pt"
        )

        self.parallel_run(config)
        parallel_model_path = (
            Path(config.experiment.config.logging.output_dir)
            / config.experiment.config.logging.run_name
            / "pretrained"
            / "model.pt"
        )

        serial_model = torch.load(serial_model_path)
        parallel_model = torch.load(parallel_model_path)
        self.assert_trained_models_equal(serial_model, parallel_model)

    def test_evaluation_episode_stats_are_equal(self):
        config = None  # TODO: Add config

        self.serial_run(config)
        serial_eval_stats_path = (
            config.experiment.config.logging.output_dir / "eval" / "eval_stats.csv"
        )
        self.parallel_run(config)
