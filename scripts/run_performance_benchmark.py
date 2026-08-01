"""Fleet-only old/new kernel benchmarks with numerical equivalence checks."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from qiskit import __version__ as qiskit_version
from qiskit_aer import AerSimulator, __version__ as aer_version

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.legacy.classical_orthogonal_layer import LegacyOrthoLayer
from src.legacy.quantum_circuit import build_circuit_legacy
from src.model_definition.classical_orthogonal_layer import OrthoLayer
from src.model_definition.quantum_layer_ideal import W, data_loader
from src.utils.simulation import bind_circuit_batch, build_circuit_template


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _max_rel(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.maximum(torch.abs(left), torch.full_like(left, 1e-12))
    return float(torch.max(torch.abs(left - right) / denominator).detach().cpu())


def _synchronized_seconds(fn: Callable[[], Any], device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _orthogonal_equivalence(
    in_features: int,
    out_features: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | bool]:
    torch.manual_seed(1729 + in_features * 100 + out_features)
    legacy = LegacyOrthoLayer(in_features, out_features).to(device=device, dtype=torch.float64)
    optimized = OrthoLayer(in_features, out_features).to(device=device, dtype=torch.float64)
    optimized.load_state_dict(legacy.state_dict())

    base = torch.randn(batch_size, in_features, device=device, dtype=torch.float64)
    x_legacy = base.detach().clone().requires_grad_(True)
    x_optimized = base.detach().clone().requires_grad_(True)
    y_legacy = legacy(x_legacy)
    y_optimized = optimized(x_optimized)
    upstream = torch.randn_like(y_legacy)
    (y_legacy * upstream).sum().backward()
    (y_optimized * upstream).sum().backward()

    metrics = {
        "output_max_abs": _max_abs(y_legacy, y_optimized),
        "output_max_rel": _max_rel(y_legacy, y_optimized),
        "input_grad_max_abs": _max_abs(x_legacy.grad, x_optimized.grad),
        "theta_grad_max_abs": _max_abs(legacy.thetas.grad, optimized.thetas.grad),
        "bias_grad_max_abs": _max_abs(legacy.bias.grad, optimized.bias.grad),
    }
    metrics["passed"] = all(value <= 1e-10 for key, value in metrics.items() if key.endswith("max_abs"))
    return metrics


def _time_layer(
    layer: torch.nn.Module,
    in_features: int,
    batch_size: int,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> tuple[float, float]:
    x = torch.randn(batch_size, in_features, device=device, requires_grad=True)

    def step() -> None:
        layer.zero_grad(set_to_none=True)
        x.grad = None
        layer(x).square().mean().backward()

    for _ in range(warmup):
        step()
    samples = [_synchronized_seconds(step, device) for _ in range(repetitions)]
    return float(np.median(samples)), float(np.percentile(samples, 90))


def benchmark_orthogonal(config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    results = []
    for case in config["orthogonal_cases"]:
        in_features = int(case["in_features"])
        out_features = int(case["out_features"])
        batch_size = int(case["batch_size"])
        equivalence = _orthogonal_equivalence(in_features, out_features, batch_size, device)

        torch.manual_seed(config["seed"] + in_features * 100 + out_features)
        legacy = LegacyOrthoLayer(in_features, out_features).to(device)
        optimized = OrthoLayer(in_features, out_features).to(device)
        optimized.load_state_dict(legacy.state_dict())
        legacy_median, legacy_p90 = _time_layer(
            legacy, in_features, batch_size, config["warmup"], config["repetitions"], device
        )
        optimized_median, optimized_p90 = _time_layer(
            optimized, in_features, batch_size, config["warmup"], config["repetitions"], device
        )
        results.append({
            **case,
            "equivalence": equivalence,
            "legacy_median_ms": legacy_median * 1000,
            "legacy_p90_ms": legacy_p90 * 1000,
            "optimized_median_ms": optimized_median * 1000,
            "optimized_p90_ms": optimized_p90 * 1000,
            "speedup": legacy_median / optimized_median,
        })
    return results


def _state_probabilities(result, index: int) -> np.ndarray:
    state = np.asarray(result.data(index)["state"].data)
    return np.abs(state) ** 2


def benchmark_qiskit(config: dict[str, Any]) -> list[dict[str, Any]]:
    simulator = AerSimulator(method="statevector", device="CPU")
    output = []
    for case_index, case in enumerate(config["qiskit_cases"]):
        n_in = int(case["in_features"])
        n_out = int(case["out_features"])
        batch_size = int(case["batch_size"])
        rng = np.random.default_rng(config["seed"] + case_index)
        inputs = rng.normal(size=(batch_size, n_in))
        n_thetas = (2 * max(n_in, n_out) - 1 - min(n_in, n_out)) * min(n_in, n_out) // 2
        weight_gate = W(n_in, n_out, rng.normal(size=n_thetas))
        uniform = np.full(max(n_in, n_out), 1 / np.sqrt(max(n_in, n_out)))
        loader_gate = data_loader(uniform)
        loader_inv_gate = loader_gate.inverse()

        start = time.perf_counter()
        legacy_circuits = [
            build_circuit_legacy(
                row, n_in, n_out, weight_gate, loader_gate, loader_inv_gate, simulator
            )
            for row in inputs
        ]
        legacy_prepare_seconds = time.perf_counter() - start
        start = time.perf_counter()
        legacy_result = simulator.run(legacy_circuits, shots=1).result()
        legacy_execute_seconds = time.perf_counter() - start

        start = time.perf_counter()
        template = build_circuit_template(
            n_in, n_out, weight_gate, loader_gate, loader_inv_gate, simulator
        )
        template_seconds = time.perf_counter() - start
        start = time.perf_counter()
        optimized_circuits = bind_circuit_batch(template, inputs)
        optimized_bind_seconds = time.perf_counter() - start
        start = time.perf_counter()
        optimized_result = simulator.run(optimized_circuits, shots=1).result()
        optimized_execute_seconds = time.perf_counter() - start

        max_probability_error = max(
            float(np.max(np.abs(
                _state_probabilities(legacy_result, i) - _state_probabilities(optimized_result, i)
            )))
            for i in range(batch_size)
        )
        legacy_total = legacy_prepare_seconds + legacy_execute_seconds
        optimized_first = template_seconds + optimized_bind_seconds + optimized_execute_seconds
        optimized_steady = optimized_bind_seconds + optimized_execute_seconds
        output.append({
            **case,
            "equivalence": {
                "probability_max_abs": max_probability_error,
                "passed": max_probability_error <= 1e-10,
            },
            "legacy_prepare_ms": legacy_prepare_seconds * 1000,
            "legacy_execute_ms": legacy_execute_seconds * 1000,
            "template_once_ms": template_seconds * 1000,
            "optimized_bind_ms": optimized_bind_seconds * 1000,
            "optimized_execute_ms": optimized_execute_seconds * 1000,
            "first_batch_speedup": legacy_total / optimized_first,
            "steady_state_speedup": legacy_total / optimized_steady,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark is fleet-only and requires CUDA.")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    report = {
        "metadata": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "qiskit": qiskit_version,
            "qiskit_aer": aer_version,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_capability": torch.cuda.get_device_capability(device),
            "seed": config["seed"],
        },
        "orthogonal_layer": benchmark_orthogonal(config, device),
        "qiskit_circuit": benchmark_qiskit(config),
    }
    report["all_equivalence_checks_passed"] = all(
        row["equivalence"]["passed"]
        for group in (report["orthogonal_layer"], report["qiskit_circuit"])
        for row in group
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_equivalence_checks_passed"]:
        raise SystemExit("Equivalence check failed")


if __name__ == "__main__":
    main()
