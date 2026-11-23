"""Export the pinned public DAPG checkpoints with MJRL and CPU PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

PROTOCOL = Path(__file__).with_name("protocol.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    args.output.mkdir()
    torch.set_num_threads(1)
    for task in protocol["tasks"]:
        skill = task["skill"]
        raw = (args.checkpoints / f"{skill}-v0.pickle").read_bytes()
        if hashlib.sha256(raw).hexdigest() != task["checkpoint_sha256"]:
            raise ValueError(f"Unexpected official checkpoint hash for {skill}")
        # The legacy authors' release used pickle. Accept only the pinned files.
        model = pickle.loads(raw, encoding="latin1").model.eval()
        if model.nonlinearity not in (torch.tanh, torch.relu):
            raise ValueError(f"Unsupported activation for {skill}")
        weights = {
            name: getattr(model, name).detach().numpy()
            for name in ("in_shift", "in_scale", "out_shift", "out_scale")
        }
        for index, layer in enumerate(model.fc_layers):
            weights[f"weight{index}"] = layer.weight.detach().numpy()
            weights[f"bias{index}"] = layer.bias.detach().numpy()
        sample = np.random.default_rng(0).normal(size=(4, model.obs_dim)).astype("float32")
        with torch.no_grad():
            reference = model(torch.from_numpy(sample)).numpy()
        np.savez(
            args.output / f"{skill}.npz",
            **weights,
            activation=model.nonlinearity.__name__,
            layers=len(model.fc_layers),
            source_sha256=task["checkpoint_sha256"],
            sample=sample,
            reference=reference,
        )


if __name__ == "__main__":
    main()
