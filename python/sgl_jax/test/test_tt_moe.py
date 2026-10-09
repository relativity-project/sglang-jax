"""FusedEPMoE's experts on TT match a float64 reference on the serving mesh.

TTNN's fused MoE kernel stores the experts in BFP4, so the outputs track the
reference to a correlation of about 0.98.
"""

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers import fused_moe


# Decode sizes and a prefill.
@pytest.mark.parametrize("tokens", [1, 3, 45, 70])
def test_experts(monkeypatch, tokens):
    if "tt" not in os.environ.get("JAX_PLATFORMS", "").split(","):
        pytest.skip("requires JAX_PLATFORMS=tt,cpu and a Tenstorrent device")
    monkeypatch.setattr(fused_moe, "get_global_server_args", lambda: SimpleNamespace(device="tt"))
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
        layer = fused_moe.FusedEPMoE(
            hidden_size=hidden,
            num_experts=experts,
            num_experts_per_tok=k,
            ep_size=devices.size,
            intermediate_dim=intermediate,
            mesh=mesh,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
        )
        split = NamedSharding(mesh, P(("data", "tensor"), None, None))
        layer.w1.value, layer.w3.value, layer.w2.value = (
            jax.device_put(w, split) for w in (wg, wu, wd)
        )
        put = lambda value: jax.device_put(value, NamedSharding(mesh, P()))
        run = jax.jit(lambda m, *args: m(*args, out_sharding=NamedSharding(mesh, P())))
        out = run(layer, put(x), put(weights), put(ids))

    reference = np.zeros(x.shape)
    x64, wg, wu, wd = (np.asarray(a).astype(np.float64) for a in (x, wg, wu, wd))
    for t in range(tokens):
        for weight, e in zip(weights[t], ids[t]):
            g, u = x64[t] @ wg[e], x64[t] @ wu[e]
            reference[t] += weight * ((g / (1 + np.exp(-g)) * u) @ wd[e])
    assert np.corrcoef(np.asarray(out).astype(np.float64).ravel(), reference.ravel())[0, 1] > 0.97
