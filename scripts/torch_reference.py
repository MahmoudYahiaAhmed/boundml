"""Generate PyTorch float64 reference values for BoundML's core tests.

Writes crates/boundml-core/tests/fixtures/torch_reference.json with, per case:
dense-network parameters in BoundML layout (weight [in, out], bias [1, out]),
inputs, targets, the loss, parameter gradients, and three torch.optim.AdamW / SGD
steps applied to fixed gradients. Run: python scripts/torch_reference.py
"""

import json
import pathlib

import torch

torch.manual_seed(0)
OUT = pathlib.Path(__file__).resolve().parents[1] / "crates/boundml-core/tests/fixtures/torch_reference.json"

ACT = {"relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "sigmoid": torch.nn.Sigmoid}


def case(name: str, inputs: int, hidden: list[int], act: str, task: str, classes: int, batch: int) -> dict:
    outs = 1 if task in ("regression", "binary") else classes
    widths = [inputs, *hidden, outs]
    layers = []
    for i in range(len(widths) - 1):
        layers.append(torch.nn.Linear(widths[i], widths[i + 1]).double())
        if i < len(widths) - 2:
            layers.append(ACT[act]())
    model = torch.nn.Sequential(*layers)
    x = torch.randn(batch, inputs, dtype=torch.float64) * 1.5
    logits = model(x)
    if task == "regression":
        y = torch.randn(batch, 1, dtype=torch.float64)
        loss = torch.nn.functional.mse_loss(logits, y)
        y_out = y
    elif task == "binary":
        y = torch.randint(0, 2, (batch, 1)).double()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        y_out = y
    else:
        y = torch.randint(0, classes, (batch,))
        loss = torch.nn.functional.cross_entropy(logits, y)
        y_out = y.double().unsqueeze(1)
    loss.backward()
    params, grads = [], []
    for m in model:
        if isinstance(m, torch.nn.Linear):
            # BoundML order: weight [in, out] then bias [1, out], layer by layer.
            params += m.weight.detach().T.reshape(-1).tolist() + m.bias.detach().reshape(-1).tolist()
            grads += m.weight.grad.T.reshape(-1).tolist()
            grads += m.bias.grad.reshape(-1).tolist()
    return {
        "name": name,
        "inputs": inputs,
        "hidden": hidden,
        "activation": act,
        "task": task,
        "classes": classes,
        "batch": batch,
        "params": params,
        "x": x.reshape(-1).tolist(),
        "y": y_out.reshape(-1).tolist(),
        "loss": loss.item(),
        "grads": grads,
        "relu_inputs_min_abs": min_abs_preact(model, x) if act == "relu" and hidden else None,
    }


def min_abs_preact(model: torch.nn.Sequential, x: torch.Tensor) -> float:
    h, m = x, float("inf")
    for layer in model:
        if isinstance(layer, torch.nn.ReLU):
            m = min(m, h.abs().min().item())
        h = layer(h)
    return m


def optimizer_case() -> dict:
    """Three optimizer steps on fixed gradients for a [weight 2x1, bias 1x1] model."""
    grads = [[0.5, -1.0, 0.25], [0.1, 0.2, -0.3], [-2.0, 0.0, 1.0]]
    result = {"grads": grads, "init": [1.0, -0.5, 0.2]}
    for kind in ("adamw", "sgd"):
        w = torch.tensor([[1.0], [-0.5]], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)
        if kind == "adamw":
            groups = [{"params": [w], "weight_decay": 0.1}, {"params": [b], "weight_decay": 0.0}]
            opt = torch.optim.AdamW(groups, lr=0.01, betas=(0.9, 0.99), eps=1e-8)
        else:
            groups = [{"params": [w], "weight_decay": 0.05}, {"params": [b], "weight_decay": 0.0}]
            opt = torch.optim.SGD(groups, lr=0.1, momentum=0.9)
        trace = []
        for g in grads:
            w.grad = torch.tensor([[g[0]], [g[1]]], dtype=torch.float64)
            b.grad = torch.tensor([[g[2]]], dtype=torch.float64)
            opt.step()
            trace.append([w[0, 0].item(), w[1, 0].item(), b[0, 0].item()])
        result[kind] = trace
    return result


def main() -> None:
    cases = [
        case("regression_tanh", 4, [6, 5], "tanh", "regression", 0, 7),
        case("binary_relu", 5, [8], "relu", "binary", 2, 9),
        case("multiclass_sigmoid", 3, [7], "sigmoid", "multiclass", 4, 6),
        case("logistic", 6, [], "relu", "binary", 2, 5),
        case("softmax_regression", 3, [], "relu", "multiclass", 3, 8),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"torch_version": torch.__version__, "cases": cases, "optimizers": optimizer_case()}, indent=1))
    print(f"wrote {OUT} (torch {torch.__version__})")


if __name__ == "__main__":
    main()
