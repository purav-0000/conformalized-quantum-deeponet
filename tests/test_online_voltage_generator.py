import numpy as np

from data.generators import online_voltage


def test_quantum_split_preserves_real_disjoint_trajectories(tmp_path, monkeypatch):
    monkeypatch.setattr(online_voltage, "OUTPUT_DIR", tmp_path)

    # Every source trajectory has a unique integer offset, making accidental
    # window pooling or overlap between splits directly observable.
    signals = np.arange(8, dtype=np.float32)[:, None] + np.linspace(
        0.0, 0.01, 121, dtype=np.float32
    )
    config = online_voltage.Config(
        memory_window_size=10,
        prediction_horizon=1,
        time_domain_limits=[0.0, 12.0],
        seed=7,
        verbose=False,
        qsim_train_signals=2,
        qsim_calibration_signals=2,
        qsim_test_signals=2,
        qsim_windows_per_signal=4,
    )
    np.random.seed(config.seed)

    online_voltage.save_quantum_sim_dataset(config, signals)

    source_markers = []
    for split in ("train", "calibration", "test"):
        with np.load(tmp_path / f"{split}.npz", allow_pickle=False) as data:
            assert data["X0"].shape == (2, 4, 10)
            assert data["X1"].shape == (2, 4, 1)
            assert data["y"].shape == (2, 4, 1)
            assert data["exchangeability_unit"].item() == "trajectory"
            np.testing.assert_array_equal(data["trajectory_id"], [0, 1])
            source_markers.extend(np.floor(data["X0"][:, 0, 0]).astype(int))

    assert len(set(source_markers)) == 6
