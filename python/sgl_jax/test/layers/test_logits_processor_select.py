"""Selecting each prompt token's logprob from [tokens, vocab] logprobs.

The selection runs on each vocab shard and reduces over the vocab, so it must
give the exact logprob for every sharding of the logprobs.
"""

import unittest
from types import SimpleNamespace

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase


class TestSelectInputTokenLogprobs(CustomTestCase):
    def test_matches_indexing_for_each_sharding(self):
        mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
        tokens, vocab = 64, 8 * 96
        rng = np.random.default_rng(0)
        logprobs = rng.standard_normal((tokens, vocab)).astype(np.float32)
        logprobs[3, :] = -np.inf
        token_ids = rng.integers(0, vocab, tokens).astype(np.int32)
        token_ids[:2] = [0, vocab - 1]
        expected = logprobs[np.arange(tokens), token_ids]

        processor = SimpleNamespace(mesh=mesh)
        select = jax.jit(
            lambda logprobs, token_ids: LogitsProcessor._select_input_token_logprobs(
                processor, logprobs, token_ids
            )
        )
        with jax.set_mesh(mesh):
            for spec in (P("data", "tensor"), P("data", None), P(None, "tensor"), P()):
                with self.subTest(spec=spec):
                    selected = select(
                        jax.device_put(logprobs, NamedSharding(mesh, spec)),
                        jax.device_put(token_ids, NamedSharding(mesh, P(None))),
                    )
                    np.testing.assert_array_equal(np.asarray(selected), expected)
                    self.assertEqual(jax.typeof(selected).sharding.spec, P(None))


if __name__ == "__main__":
    unittest.main()
