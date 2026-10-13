from pathlib import Path

import matplotlib.pyplot as plt
import torch

from tbp.monty.frameworks.models.object_model import GraphObjectModel
from tbp.monty.frameworks.utils.plot_utils_dev import plot_graph


def load_graph(exp_dir: Path, epoch: int, object_name: str) -> GraphObjectModel:
    model_path = exp_dir / str(epoch) / "model.pt"
    state_dict = torch.load(model_path, weights_only=False)
    return state_dict["lm_dict"][0]["graph_memory"][object_name]["patch"]


exp_dir = Path(
    "~/tbp/results/monty/projects/surf_agent_2obj_unsupervised_mujoco"
).expanduser()
n_epochs = 3
obj_name = "new_object1"  # The generated object ID corresponding to the bowl

# Load object graphs
graphs = [load_graph(exp_dir, epoch, obj_name) for epoch in range(n_epochs)]

fig = plt.figure(figsize=(8, 3))
for epoch in range(n_epochs):
    ax = fig.add_subplot(1, n_epochs, epoch + 1, projection="3d")
    plot_graph(graphs[epoch], ax=ax)
    ax.view_init(125, 115, 0)
    ax.set_title(f"epoch {epoch}")
fig.suptitle("Bowl Object Models")
fig.tight_layout()
plt.show()
