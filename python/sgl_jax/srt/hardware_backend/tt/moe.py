"""Mixture-of-experts layers on TT with TTNN's sparse matmul."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.xla_metadata import set_xla_metadata

from sgl_jax.srt.eplb.expert_location import get_global_server_args

# Tokens are grouped into tiles; an expert runs on a tile if any of its tokens
# routed there, so each active expert's weights are read once per tile.
TILE = 32
# The sparse matmuls allocate their outputs for every expert, tokens * experts
# * intermediate each. Cap them for long prefills.
MAX_CHUNK_TOKENS = 2048


def enabled():
    """Whether MoE experts run through this module."""
    server_args = get_global_server_args()
    if server_args is not None:
        return server_args.device == "tt"
    return jax.default_backend() == "tt"


def _expert_matmul(a, experts, active):
    """a [1, B, M, K] times each expert [E, K, N] where active [B, E] is
    nonzero: [1, B, 1, E, M, N], zeros for the rest."""
    _, blocks, rows, _ = a.shape
    count, _, width = experts.shape
    aval = jax.typeof(a)
    out = jax.ShapeDtypeStruct(
        (1, blocks, 1, count, rows, width), a.dtype, manual_axis_type=aval.manual_axis_type
    )
    # nnz="0" makes TTNN count the active pairs from the sparsity at run time.
    with set_xla_metadata(is_input_a_sparse="false", is_input_b_sparse="true", nnz="0"):
        return jax.ffi.ffi_call("tt.sparse_matmul", out, vmap_method="sequential")(
            a, experts[None], active.reshape(1, blocks, 1, count)
        )


def _experts_chunk(x, routing, w0, w1, wo):
    """x [T, H] with T a multiple of TILE; routing [T, E] holds each token's
    router weight for its selected experts and zero elsewhere."""
    T, H = x.shape
    E, _, inter = w0.shape
    blocks = T // TILE
    a = x.reshape(1, blocks, TILE, H)
    active = jnp.max((routing.reshape(blocks, TILE, E) != 0).astype(jnp.bfloat16), axis=1)
    gate, up = (_expert_matmul(a, w, active) for w in (w0, w1))
    # Apply the router weights before the down projection, which is linear, so
    # tokens in an active tile that didn't pick an expert contribute zero, as
    # do the experts the sparse matmuls skipped.
    scale = routing.reshape(blocks, TILE, E).transpose(0, 2, 1)[..., None].astype(x.dtype)
    act = (jax.nn.silu(gate) * up).reshape(blocks, E, TILE, inter) * scale
    # Down projection and the sum over experts in one dense matmul. It reads
    # every expert's weights, but a sparse down projection writes tokens *
    # experts * hidden outputs and summing those costs several times more.
    act = act.transpose(0, 2, 1, 3).reshape(T, E * inter)
    return act @ wo.reshape(E * inter, H)


def sparse_experts(hidden_states, topk_weights, topk_ids, w0, w1, wo):
    """Each token's weighted sum of its top-k SiLU-gated experts.

    hidden_states [T, H], topk_weights and topk_ids [T, k], w0 and w1
    [E, H, I], wo [E, I, H]. The experts may hold a tensor-parallel slice of
    the intermediate dim, in which case the result is a partial sum.
    """
    T = hidden_states.shape[0]
    E = w0.shape[0]
    onehot = topk_ids[..., None] == jnp.arange(E, dtype=topk_ids.dtype)
    routing = jnp.sum(jnp.where(onehot, topk_weights[..., None], 0), axis=1)
    padded = -(-T // TILE) * TILE
    if padded != T:
        hidden_states = jnp.pad(hidden_states, ((0, padded - T), (0, 0)))
        routing = jnp.pad(routing, ((0, padded - T), (0, 0)))
    chunks = [
        _experts_chunk(
            hidden_states[start : start + MAX_CHUNK_TOKENS],
            routing[start : start + MAX_CHUNK_TOKENS],
            w0,
            w1,
            wo,
        )
        for start in range(0, padded, MAX_CHUNK_TOKENS)
    ]
    out = chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks)
    return out[:T]
