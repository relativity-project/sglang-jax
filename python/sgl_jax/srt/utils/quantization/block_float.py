"""Block floating point weights.

In a block floating point format, blocks of values share one exponent, so it
has no JAX dtype. A weight quantized to one stays a regular array in JAX, typed
``BlockFloatWeight``, and the device stores it in the format: on Tenstorrent,
the TT attention backend tells libtt to.
"""

import enum

from flax import nnx


class BlockFloat(enum.StrEnum):
    """Block floating point formats: blocks of 16 values share one exponent."""

    BFP_BF8 = "bfp_bf8"
    BFP_BF4 = "bfp_bf4"


class BlockFloatWeight(nnx.Param):
    """A weight the device stores in the block floating point format named by
    its ``block_float`` metadata."""


def to_block_float(weight: nnx.Variable, block_float: BlockFloat) -> BlockFloatWeight:
    """``weight`` as a ``block_float`` weight, with the same value and metadata."""
    return BlockFloatWeight(
        weight.value, **{**weight.get_metadata(), "block_float": str(block_float)}
    )
