"""TT MoE experts match a float64 reference on the serving mesh."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.hardware_backend.tt import moe as tt_moe
from sgl_jax.srt.layers.moe import EPMoE, create_moe_weights_mapping


def test_weight_mapping_keeps_experts_on_model_mesh(monkeypatch):
    monkeypatch.setattr(tt_moe, "enabled", lambda: True)
    mappings = create_moe_weights_mapping("model.layers.0", "model.layers.0", num_experts=4)
    shardings = {name.rsplit(".", 1)[-1]: spec.sharding for name, spec in mappings.items()}
    assert shardings == {
        "wi_0": (None, None, "tensor"),
        "wi_1": (None, None, "tensor"),
        "wo": (None, "tensor", None),
    }


def reference(x, weights, ids, wg, wu, wd):
    out = np.zeros(x.shape)
    for t in range(x.shape[0]):
        for weight, e in zip(weights[t], ids[t]):
            g, u = x[t] @ wg[e], x[t] @ wu[e]
            out[t] += weight * ((g / (1 + np.exp(-g)) * u) @ wd[e])
    return out


@pytest.mark.parametrize("tokens", [1, 45, 70])
def test_experts(monkeypatch, tokens):
    if "tt" not in os.environ.get("JAX_PLATFORMS", "").split(","):
        pytest.skip("requires JAX_PLATFORMS=tt,cpu and a Tenstorrent device")
    # Two chunks at 70 tokens, the second one partly padding.
    monkeypatch.setattr(tt_moe, "MAX_CHUNK_TOKENS", 64)
    hidden, intermediate, experts, k = 256, 256, 16, 4
    devices = np.array(jax.devices("tt"))
    mesh = jax.sharding.Mesh(
        devices.reshape(1, -1), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    rng = np.random.default_rng(tokens)
    bf16 = lambda shape, scale: (rng.standard_normal(shape) * scale).astype(jnp.bfloat16)
    x = bf16((tokens, hidden), 0.5)
    wg, wu = bf16((experts, hidden, intermediate), 1 / 16), bf16(
        (experts, hidden, intermediate), 1 / 16
    )
    wd = bf16((experts, intermediate, hidden), 1 / 16)
    ids = np.stack([rng.choice(experts, k, replace=False) for _ in range(tokens)]).astype(np.int32)
    weights = rng.random((tokens, k)).astype(np.float32)
    weights /= weights.sum(-1, keepdims=True)

    with jax.set_mesh(mesh):
        moe = EPMoE(
            hidden_size=hidden,
            num_experts=experts,
            num_experts_per_tok=k,
            ep_size=1,
            mesh=mesh,
            intermediate_dim=intermediate,
        )
        assert moe.use_tt_sparse_matmul
        put = lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec))
        moe.wi_0.value = put(wg, P(None, None, "tensor"))
        moe.wi_1.value = put(wu, P(None, None, "tensor"))
        moe.wo.value = put(wd, P(None, "tensor", None))
        replicated = NamedSharding(mesh, P())
        run = jax.jit(lambda m, *args: m(*args, out_sharding=replicated))
        out = run(moe, put(x, P()), put(weights, P()), put(ids, P()))

    as64 = lambda a: np.asarray(a).astype(np.float64)
    want = reference(as64(x), weights, ids, as64(wg), as64(wu), as64(wd))
    error = np.abs(as64(out) - want).max() / np.abs(want).max()
    assert error < 0.01, error
