import time
import argparse
from typing import List
import numpy as np
import torch
from torch import nn


# --- Layer Implementations ---

# This is the original version in Xiao et al.
class OrthoLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)
        size = (2 * larger_features - 1 - smaller_features) * smaller_features / 2  # number of free parameters
        # torch.manual_seed(0)
        self.thetas = torch.nn.Parameter(torch.randn(int(size)))  # normal distribution initializer for thetas
        self.bias = torch.nn.Parameter(torch.zeros(int(out_features)))

    def hidden_layer(self, x, in_features, out_features):
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)

        if larger_features == smaller_features:
            smaller_features -= 1  # 6-6 6-5 have the same pyramid
        x_end_index = np.concatenate([
            np.arange(2, larger_features + 1),
            larger_features + 1 - np.arange(2, smaller_features + 1)
        ])
        x_start_index = np.concatenate([
            np.arange(x_end_index.shape[0] + smaller_features - larger_features) % 2,  # [0, 1, 0, 1, ...]
            np.arange(larger_features - smaller_features)
        ])

        x_slice_sizes = x_end_index - x_start_index

        if in_features < out_features:  # generate the pyramid for in_features < out_features case
            x_end_index = x_end_index[::-1]
            x_start_index = x_start_index[::-1]
            x_slice_sizes = x_slice_sizes[::-1]
            x = torch.nn.functional.pad(x,
                                        (out_features - x.shape[1], 0))  # pad x fist if in_features < out_features case

        theta_start_index = 0

        for i in range(len(x_start_index)):
            theta_slice = self.thetas[theta_start_index:theta_start_index + x_slice_sizes[i] // 2]
            theta_start_index = theta_start_index + x_slice_sizes[i] // 2
            x_slice = x[:, x_start_index[i]:x_end_index[i]]

            # generate rotation matrix
            n = len(theta_slice)
            row_indices = torch.cat([torch.tensor([2 * i, 2 * i, 2 * i + 1, 2 * i + 1]) for i in range(n)])
            column_indices = torch.cat([torch.tensor([2 * i, 2 * i + 1]).repeat(2) for i in range(n)])
            indices = torch.stack([row_indices, column_indices])
            theta_slice = theta_slice.view(-1, 1)
            values = torch.cat(
                [torch.cos(theta_slice), torch.sin(theta_slice), -torch.sin(theta_slice), torch.cos(theta_slice)],
                dim=1).view(-1)
            rotation_matrix = torch.sparse_coo_tensor(indices, values, size=[2 * n, 2 * n])
            x_new = x.clone()
            x_new[:, x_start_index[i]:x_end_index[i]] = torch.mm(x_slice, rotation_matrix)
            x = x_new  # to avoid in-place operation

        if in_features > out_features:
            x = x[:, in_features - out_features:]

        return x + self.bias

    def forward(self, x):
        if x.shape[1] != self.in_features:
            raise AssertionError(
                f'x shape {x.shape} isn\'t equal to {self.in_features}'
            )
        x = self.hidden_layer(x, self.in_features, self.out_features)
        return x


# This is our version that pre-computes indices
class OrthoLayerOptimized(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)

        # This calculation needs to use the modified smaller_features for the in==out case
        size = (2 * larger_features - 1 - smaller_features) * smaller_features // 2

        self.thetas = torch.nn.Parameter(torch.randn(int(size)))
        self.bias = torch.nn.Parameter(torch.zeros(int(out_features)))

        # Precompute pyramid indices
        if larger_features == smaller_features:
            smaller_features -= 1  # Modify for index calculation

        self.x_end_index = np.concatenate([
            np.arange(2, larger_features + 1),
            larger_features + 1 - np.arange(2, smaller_features + 1)
        ])
        self.x_start_index = np.concatenate([
            np.arange(self.x_end_index.shape[0] + smaller_features - larger_features) % 2,
            np.arange(larger_features - smaller_features)
        ])
        self.x_slice_sizes = self.x_end_index - self.x_start_index

        # Precompute sparse indices for each slice
        self.precomputed_indices = []
        for slice_size in self.x_slice_sizes:
            n = slice_size // 2
            if n == 0:
                self.precomputed_indices.append(None)
                continue
            row_idx = torch.cat([torch.tensor([2 * i, 2 * i, 2 * i + 1, 2 * i + 1]) for i in range(n)])
            col_idx = torch.cat([torch.tensor([2 * i, 2 * i + 1]).repeat(2) for i in range(n)])
            self.precomputed_indices.append(torch.stack([row_idx, col_idx]))

    def hidden_layer(self, x, in_features, out_features):
        # Determine the order of operations based on layer dimensions
        if in_features < out_features:
            x_end_index = self.x_end_index[::-1]
            x_start_index = self.x_start_index[::-1]
            x_slice_sizes = self.x_slice_sizes[::-1]
            precomputed_indices = self.precomputed_indices[::-1]
            x = torch.nn.functional.pad(x, (out_features - in_features, 0))
        else:
            x_end_index = self.x_end_index
            x_start_index = self.x_start_index
            x_slice_sizes = self.x_slice_sizes
            precomputed_indices = self.precomputed_indices

        theta_start = 0
        for i, sz in enumerate(x_slice_sizes):
            n = sz // 2
            if n == 0:
                continue

            theta_end = theta_start + n
            theta_slice = self.thetas[theta_start:theta_end]
            theta_start = theta_end

            # Slice the original x from the previous step
            x_slice = x[:, x_start_index[i]:x_end_index[i]]

            cos_t = torch.cos(theta_slice)
            sin_t = torch.sin(theta_slice)
            values = torch.stack((cos_t, sin_t, -sin_t, cos_t), dim=1).view(-1)

            indices = precomputed_indices[i].to(x.device)

            rotation = torch.sparse_coo_tensor(
                indices, values, (2 * n, 2 * n)
            )

            x_new = x.clone()
            x_new[:, x_start_index[i]:x_end_index[i]] = torch.mm(x_slice, rotation)
            x = x_new

        if in_features > out_features:
            x = x[:, in_features - out_features:]

        return x + self.bias

    def forward(self, x):
        if x.shape[1] != self.in_features:
            raise AssertionError(f"x shape {x.shape} isn't equal to {self.in_features}")
        return self.hidden_layer(x, self.in_features, self.out_features)


# --- Benchmarking Utilities ---

def run_benchmark(model: nn.Module, name: str, n_runs: int, batch_size: int, in_features: int, device: torch.device):
    """Measures the average forward pass time for a given model instance."""
    print(f"Benchmarking {name}...")
    model.to(device).eval()
    dummy_input = torch.randn(batch_size, in_features, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input)

    end_time = time.perf_counter()
    avg_time_ms = (end_time - start_time) / n_runs * 1000
    print(f"  -> Average time: {avg_time_ms:.4f} ms per run")
    return avg_time_ms


def main(args):

    # Ortho layers perform much faster on CPU given batch size and feature count are not too big
    device = torch.device("cpu")

    print("=" * 60)
    print(" OrthoLayer Performance and Correctness Check")
    print(f"Device: {device.type.upper()}, Layer: {args.in_features}x{args.out_features}, Batch: {args.batch_size}")
    print("=" * 60)

    # --- Verification ---
    print("\n--- Verifying Output Correctness ---")
    baseline = OrthoLayer(args.in_features, args.out_features)
    optimized = OrthoLayerOptimized(args.in_features, args.out_features)

    # Ensure all models have the same parameters for a fair comparison
    optimized.load_state_dict(baseline.state_dict())

    baseline.to(device).eval()
    optimized.to(device).eval()

    verification_input = torch.randn(args.batch_size, args.in_features, device=device)
    with torch.no_grad():
        output_baseline = baseline(verification_input)
        output_optimized = optimized(verification_input)

    correct1 = torch.allclose(output_baseline, output_optimized, atol=1e-6)
    print(f"✅ Baseline vs. Optimized Correct: {correct1}")
    if not correct1:
        print("❌ Failure: Outputs DO NOT match!")
    print("-" * 36)

    # Benchmarking
    print("\n--- Performance Benchmark ---")
    time_baseline = run_benchmark(baseline, "Baseline", args.n_runs, args.batch_size, args.in_features, device)
    time_optimized = run_benchmark(optimized, "Optimized", args.n_runs, args.batch_size, args.in_features, device)

    print("\n" + "=" * 60)
    print(" Results")
    print("-" * 60)
    speedup = time_baseline / time_optimized
    print(f"Overall Speedup (Optimized vs. Baseline): {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Orthogonal Layer Implementations")
    parser.add_argument("--in_features", type=int, default=20, help="Input features")
    parser.add_argument("--out_features", type=int, default=20, help="Output features")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for benchmarking")
    parser.add_argument("--n_runs", type=int, default=200, help="Number of benchmark runs")
    cli_args = parser.parse_args()
    main(cli_args)
