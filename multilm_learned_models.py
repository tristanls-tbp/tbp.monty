from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tbp.monty.frameworks.utils.plot_utils_dev import plot_graph

# Get path to pretrained model
project_dir = Path("~/tbp/results/monty/projects").expanduser()
model_name = "dist_agent_5lm_2obj_train_mujoco"
model_path = project_dir / model_name / "pretrained" / "model.pt"
state_dict = torch.load(model_path, weights_only=False)

fig = plt.figure(figsize=(8, 3))
for lm_id in range(5):
    ax = fig.add_subplot(1, 5, lm_id + 1, projection="3d")
    graph = state_dict["lm_dict"][lm_id]["graph_memory"]["mug"][f"patch_{lm_id}"]
    plot_graph(graph, ax=ax)
    ax.view_init(-65, 0, 0)
    ax.set_title(f"LM {lm_id}")
fig.suptitle("Mug Object Models")
fig.tight_layout()
plt.show()
