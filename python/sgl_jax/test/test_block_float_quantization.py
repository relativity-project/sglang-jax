"""Block floating point quantization: weights keep their values and get a device format."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.gate import GateLogit
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.utils.quantization.block_float import BlockFloat, BlockFloatWeight
from sgl_jax.srt.utils.quantization.quantization_utils import apply_quantization


def test_tt_bfp8_config():
    config = QuantizationConfig.from_yaml("tt_bfp8.yaml")
    assert config.get_linear_rules()[0]["weight_dtype"] == "bfp_bf8"
    # The MoE layers see unquantized experts.
    assert config.moe_weight_dtype is None and config.moe_activation_dtype is None
    assert config.moe_block_float == BlockFloat.BFP_BF8
    assert config.lm_head_block_float == BlockFloat.BFP_BF8
    assert config.has_moe_quantization() and not config.is_static_checkpoint


def test_block_float_quantization_retypes_weights():
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    class Model(nnx.Module):
        def __init__(self):
            self.proj = LinearBase(
                input_size=64,
                output_size=128,
                use_bias=False,
                mesh=mesh,
                kernel_axes=(None, "tensor"),
                params_dtype=jnp.bfloat16,
                scope_name="proj",
            )
            self.moe_gate = GateLogit(input_size=64, num_experts=8, kernel_dtype=jnp.bfloat16)
            self.experts = FusedEPMoE(
                hidden_size=64,
                num_experts=8,
                num_experts_per_tok=2,
                ep_size=1,
                mesh=mesh,
                intermediate_dim=32,
            )
            self.lm_head = ParallelLMHead(256, 64, mesh=mesh)

    with jax.set_mesh(mesh):
        model = Model()
    proj = model.proj.weight.value

    class ModelConfig:
        quantization_config = QuantizationConfig.from_yaml("tt_bfp8.yaml")

    model = apply_quantization(ModelConfig(), model)

    for weight in (
        model.proj.weight,
        model.experts.w1,
        model.experts.w2,
        model.experts.w3,
        model.lm_head.embedding,
    ):
        assert isinstance(weight, BlockFloatWeight)
        assert weight.get_metadata()["block_float"] == "bfp_bf8"
    assert model.proj.weight.value is proj  # Same values and sharding.
    assert isinstance(model.proj, LinearBase)  # Not a QuantizedLinear.
    assert not isinstance(model.moe_gate.kernel, BlockFloatWeight)
