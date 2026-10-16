"""PLE dilated depthwise conv + SiLU reference.

The pool stores S=(K-1)*dilation consecutive tokens, oldest first. Nonzero
request slots must be unique; slot zero is padding and is never written.
"""

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.gdn.gated_delta import _scatter_track


def _initial_state(pool, slots, has_initial_state, channels, kernel, dilation):
    if dilation < 1:
        raise ValueError("PLE conv dilation must be positive")
    state_len = (kernel - 1) * dilation
    assert pool.shape[1:] == (channels, state_len)
    # Keep reads ahead of writes when the caller donates the full pool.
    pool = jax.lax.optimization_barrier(pool)
    state = pool[slots]
    if has_initial_state is not None:
        assert has_initial_state.shape == slots.shape
        state = jnp.where(has_initial_state[:, None, None], state, 0)
    return pool, state


def _depthwise(window, weight, dtype, *, channel_axis):
    """sum_k window[..., k] * weight[c, k] -> window.shape[:-1], accumulated in
    FP32 and rounded once, as the einsum it replaces. Written as one
    multiply-add per tap: as an einsum, the contraction over k is a batched
    matmul over every channel, which TT runs as one padded tile product per
    channel (3.1 ms per decode step for 2560 channels)."""
    shape = [1] * (window.ndim - 1)
    shape[channel_axis] = weight.shape[0]
    total = None
    for k in range(weight.shape[1]):
        term = window[..., k].astype(jnp.float32) * weight[:, k].astype(jnp.float32).reshape(shape)
        total = term if total is None else total + term
    return total.astype(dtype)


def _write_state(pool, slots, state):
    # Drop dummy-slot updates, including repeated zeros in a padded batch.
    indices = jnp.where(slots == 0, pool.shape[0], slots)
    return pool.at[indices].set(state.astype(pool.dtype), mode="drop")


def ngram_conv_prefill(
    x,  # [C, T], packed activations
    weight,  # [C, K]
    *,
    cu_seqlens,  # [B+1]
    conv_state,  # [slots, C, S]
    state_indices,  # [B]
    has_initial_state=None,
    dilation=3,
    track_indices=None,  # [B] req → track slot (None = OFF)
    track_mask=None,  # [B] bool boundary mask
):
    """Return SiLU(conv(x)) and the updated full pool for ragged extend."""
    channels, tokens = x.shape
    kernel = weight.shape[1]
    state_len = (kernel - 1) * dilation
    batch = cu_seqlens.shape[0] - 1
    assert weight.shape == (channels, kernel)
    assert state_indices.shape == (batch,)
    pool, state = _initial_state(
        conv_state, state_indices, has_initial_state, channels, kernel, dilation
    )
    if tokens == 0:
        return x, pool

    starts = cu_seqlens[:-1]
    lengths = cu_seqlens[1:] - starts
    t = jnp.arange(tokens)
    seq = jnp.searchsorted(cu_seqlens, t, side="right") - 1
    lookback = jnp.arange(kernel) * dilation
    source = t[:, None] - lookback[None, :]
    window = x[:, jnp.clip(source, 0, tokens - 1)]  # [C, T, K]
    if state_len:
        position = source - starts[seq, None]
        prior_idx = jnp.clip(state_len + position, 0, state_len - 1)
        prior = state[seq[:, None], :, prior_idx].transpose(2, 0, 1)
        window = jnp.where(position[None] >= 0, window, prior)
    y = jax.nn.silu(_depthwise(window, weight[:, ::-1], x.dtype, channel_axis=0))

    if state_len:
        position = lengths[:, None] - state_len + jnp.arange(state_len)[None, :]
        source = jnp.clip(starts[:, None] + position, 0, tokens - 1)
        from_x = x[:, source].transpose(1, 0, 2)
        prior_idx = jnp.clip(state_len + position, 0, state_len - 1)
        prior = state[jnp.arange(batch)[:, None], :, prior_idx].transpose(0, 2, 1)
        final = jnp.where(position[:, None, :] >= 0, from_x, prior)
    else:
        final = state
    updated = _write_state(pool, state_indices, final)
    if track_indices is not None:
        updated = _scatter_track(updated, track_indices, track_mask, final)
    return jax.lax.optimization_barrier((y, updated))


def ngram_conv_update(
    x,  # [B, C]
    conv_state,
    state_indices,
    weight,
    *,
    has_initial_state=None,
    dilation=3,
    track_indices=None,  # [B] req → track slot (None = OFF)
    track_mask=None,  # [B] bool boundary mask
):
    """Single-token decode with the same pool contract as prefill."""
    batch, channels = x.shape
    assert state_indices.shape == (batch,)
    assert weight.shape[0] == channels
    pool, state = _initial_state(
        conv_state, state_indices, has_initial_state, channels, weight.shape[1], dilation
    )
    window = jnp.concatenate([state, x[..., None]], axis=-1)
    y = jax.nn.silu(_depthwise(window[..., ::dilation], weight, x.dtype, channel_axis=1))
    updated = _write_state(pool, state_indices, window[..., 1:])
    if track_indices is not None:
        updated = _scatter_track(updated, track_indices, track_mask, window[..., 1:])
    return jax.lax.optimization_barrier((y, updated))
