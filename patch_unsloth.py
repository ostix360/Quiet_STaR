from peft import PeftModelForCausalLM
from unsloth.models.llama import LlamaDecoderLayer_fast_forward, LlamaModel_fast_forward, \
    PeftModelForCausalLM_fast_forward, LlamaRotaryEmbedding
from unsloth.models.mistral import MistralAttention_fast_forward

from modeling_mistral import MistralAttention, MistralSdpaAttention, MistralFlashAttention2, MistralDecoderLayer, \
    MistralModel


@staticmethod
def pre_patch():
    MistralAttention      .forward = MistralAttention_fast_forward
    MistralSdpaAttention  .forward = MistralAttention_fast_forward
    MistralFlashAttention2.forward = MistralAttention_fast_forward
    MistralDecoderLayer   .forward = LlamaDecoderLayer_fast_forward
    MistralModel          .forward = LlamaModel_fast_forward
    PeftModelForCausalLM  .forward = PeftModelForCausalLM_fast_forward

    # Solves https://github.com/unslothai/unsloth/issues/168
    # Static KV Cache was introduced in 4.38.0, causing training to be much slower.
    # Inferene can now be CUDAGraphed, but we shall retain the old rotary embeddings.
    # https://github.com/huggingface/transformers/pull/27931
    # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/llama/modeling_llama.py
    import transformers.models.mistral.modeling_mistral
    transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding = LlamaRotaryEmbedding
    return
pass

def patch():
    from unsloth.models import mistral
    mistral.FastMistralModel.pre_patch = pre_patch
