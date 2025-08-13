import torch
from torch import nn
import numpy as np
import time
from typing import List


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


class OrthoLayerOptimized(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        larger_features = max(in_features, out_features)
        smaller_features = min(in_features, out_features)

        # This calculation needs to use the modified smaller_features for the in==out case
        temp_smaller = smaller_features
        if larger_features == smaller_features:
            temp_smaller -= 1
        size = (2 * larger_features - 1 - temp_smaller) * temp_smaller // 2

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

        if in_features < out_features:
            self.x_end_index = self.x_end_index[::-1]
            self.x_start_index = self.x_start_index[::-1]
            self.x_slice_sizes = self.x_slice_sizes[::-1]
            self.precomputed_indices = self.precomputed_indices[::-1]


    def hidden_layer(self, x, in_features, out_features):
        # Determine the order of operations based on layer dimensions

        if in_features < out_features:
            x = torch.nn.functional.pad(x, (out_features - in_features, 0))

        theta_start = 0
        for i, sz in enumerate(self.x_slice_sizes):
            n = sz // 2
            if n == 0:
                continue

            theta_end = theta_start + n
            theta_slice = self.thetas[theta_start:theta_end]
            theta_start = theta_end

            # Slice the original x from the previous step
            x_slice = x[:, self.x_start_index[i]:self.x_end_index[i]]

            cos_t = torch.cos(theta_slice)
            sin_t = torch.sin(theta_slice)
            values = torch.stack((cos_t, sin_t, -sin_t, cos_t), dim=1).view(-1)

            indices = self.precomputed_indices[i].to(x.device)

            rotation = torch.sparse_coo_tensor(
                indices, values, (2 * n, 2 * n)
            )

            x_new = x.clone()
            x_new[:, self.x_start_index[i]:self.x_end_index[i]] = torch.mm(x_slice, rotation)
            x = x_new

        if in_features > out_features:
            x = x[:, in_features - out_features:]

        return x + self.bias

    def forward(self, x):
        if x.shape[1] != self.in_features:
            raise AssertionError(f"x shape {x.shape} isn't equal to {self.in_features}")
        return self.hidden_layer(x, self.in_features, self.out_features)

def run_benchmark(model, name, n_runs, batch_size, in_features, device):
    """Measures the average forward pass time for a given model instance."""
    print(f"Benchmarking {name}...")
    model.to(device).eval()
    dummy_input = torch.randn(batch_size, in_features, device=device)
    with torch.no_grad():
        for _ in range(10): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs): _ = model(dummy_input)
    if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.perf_counter()
    avg_time_ms = (end_time - start_time) / n_runs * 1000
    print(f"  -> Average time: {avg_time_ms:.4f} ms per run")
    return avg_time_ms

IN_FEATURES = 11
OUT_FEATURES = 10
BATCH_SIZE = 200
N_RUNS = 200
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print("=" * 60)
print(" OrthoLayer Performance and Correctness Check")
print(f"Device: {device.type.upper()}, Layer: {IN_FEATURES}x{OUT_FEATURES}, Batch: {BATCH_SIZE}")
print("=" * 60)

print("\n--- Verifying Output Correctness ---")
baseline_layer = OrthoLayer(IN_FEATURES, OUT_FEATURES)
optimized_layer_instance = OrthoLayerOptimized(IN_FEATURES, OUT_FEATURES)
optimized_layer_instance.load_state_dict(baseline_layer.state_dict())
baseline_layer.to(device).eval()
optimized_layer_instance.to(device).eval()
verification_input = torch.randn(BATCH_SIZE, IN_FEATURES, device=device)
with torch.no_grad():
    output_baseline = baseline_layer(verification_input)
    output_optimized = optimized_layer_instance(verification_input)
if torch.allclose(output_baseline, output_optimized, atol=1e-6):
    print("✅ Success: Outputs from both layers are identical.")
else:
    print("❌ Failure: Outputs DO NOT match!")
print("-" * 36)

print("\n--- Performance Benchmark ---")
optimized_time = run_benchmark(optimized_layer_instance, "Optimized (JIT)", N_RUNS, BATCH_SIZE, IN_FEATURES, device)
baseline_time = run_benchmark(baseline_layer, "Baseline", N_RUNS, BATCH_SIZE, IN_FEATURES, device)

print("\n" + "=" * 60)
print(" Results")
print("-" * 60)
if optimized_time > 0 and baseline_time > 0:
    speedup = baseline_time / optimized_time
    print(f"🚀 Speedup: {speedup:.2f}x")
else:
    print("Could not calculate speedup.")
print("=" * 60)

