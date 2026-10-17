"""File-backed parameters (model_loader/weights/host_cache.py) on four CPU devices."""

import os
import subprocess
import sys
from pathlib import Path


def test_host_weight_cache(tmp_path):
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4")
    result = subprocess.run(
        [sys.executable, __file__, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HOST_CACHE_PASS" in result.stdout, result.stdout + result.stderr


def _worker(root):
    from types import SimpleNamespace

    import jax
    import numpy as np
    from flax import nnx
    from jax.sharding import AxisType, Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P
    from safetensors.numpy import save_file

    from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec

    mesh = Mesh(np.asarray(jax.devices()), ("tensor",), axis_types=(AxisType.Explicit,))
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    rng = np.random.default_rng(0)
    tensors = {
        "w": rng.standard_normal((16, 8)).astype(np.float32),
        "norm": rng.standard_normal(16).astype(np.float32),
        "e0": rng.standard_normal((16, 8)).astype(np.float32),
        "e1": rng.standard_normal((16, 8)).astype(np.float32),
    }
    save_file(tensors, checkpoint / "model.safetensors")
    os.environ["SGLANG_JAX_HOST_WEIGHT_CACHE"] = str(root / "cache")

    def load(fail_reads=False):
        model = nnx.Module()
        model.w = nnx.Param(
            jax.ShapeDtypeStruct((8, 16), np.float32, sharding=NamedSharding(mesh, P(None, "tensor")))
        )
        model.norm = nnx.Param(
            jax.ShapeDtypeStruct((16,), np.float32, sharding=NamedSharding(mesh, P()))
        )
        model.experts = nnx.Param(jax.ShapeDtypeStruct((2, 8, 16), np.float32))
        loader = WeightLoader(model, SimpleNamespace(model_path=str(checkpoint)), mesh)
        if fail_reads:

            def unreachable(*args, **kwargs):
                raise AssertionError("a cached parameter was read from the checkpoint")

            loader.source.read_tensor = unreachable
            loader.source.read_ranges = unreachable
        with jax.set_mesh(mesh):
            loader.load(
                {
                    "w": WeightSpec("w", transpose=True, sharding=(None, "tensor")),
                    "norm": WeightSpec("norm", sharding=(None,)),
                    "experts": WeightSpec(
                        "experts",
                        sources=("e0", "e1"),
                        transpose=True,
                        sharding=(None, None, "tensor"),
                    ),
                }
            )
        expected = {
            "w": tensors["w"].T,
            "norm": tensors["norm"],
            "experts": np.stack((tensors["e0"].T, tensors["e1"].T)),
        }
        for name, want in expected.items():
            value = getattr(model, name)[...]
            assert len(value.addressable_shards) == 4, name
            for shard in value.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data), want[shard.index])
        return model

    first = load()
    # One file pair per parameter; replicated shards are stored once.
    assert len(list((root / "cache").glob("*.json"))) == 3
    second = load(fail_reads=True)
    for name in ("w", "norm", "experts"):
        assert getattr(second, name)[...].sharding == getattr(first, name)[...].sharding, name

    # A file added elsewhere in the checkpoint (a download still running)
    # leaves the entries of the files they were read from valid.
    save_file({"later": tensors["w"]}, checkpoint / "model-later.safetensors")
    load(fail_reads=True)

    # A rewritten file changes its stat: the cache must not be used.
    tensors["w"] = tensors["w"] + 1
    save_file(tensors, checkpoint / "model.safetensors")
    load()
    print("HOST_CACHE_PASS", flush=True)


if __name__ == "__main__":
    _worker(Path(sys.argv[1]))
