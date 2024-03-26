import torch

from trainer import SFTrainer

torch.backends.cuda.matmul.allow_tf32 = True
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights, dispatch_model
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from transformers import TrainingArguments, Trainer
import os
import time
import wandb
from transformers import EarlyStoppingCallback
from eval_helpers import preprocess_eval_function_gsm, preprocess_eval_function_csqa, preprocess_function, compute_metrics, truncate_or_pad
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

# MAIN SETUP

wandb_cache_dir ="./cache/quietstar/wandb_cache"
project_name = "quiet-star"
os.environ["WANDB_PROJECT"] = project_name + "-all-ds"
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
n_ahead_talk_global = 4
n_passes_global = 2
n_ahead_global = 12
n_examples = 1_000
full_batch_size = 8
eval_and_logging_steps = 10
save_steps = 100

batch_size = full_batch_size // n_passes_global
global_gradient_accumulation_steps = full_batch_size // batch_size
run_id = int(time.time())
training_args = TrainingArguments(
    output_dir=f"./cache/quietstar/{run_id}",
    learning_rate=1e-6,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=global_gradient_accumulation_steps,
    max_grad_norm=1.0,
    max_steps=100000,
    warmup_steps=20,
    auto_find_batch_size=True,
    weight_decay=0.001,
    label_names=["labels"],
    include_inputs_for_metrics=True,
    logging_steps=eval_and_logging_steps,
    eval_steps=eval_and_logging_steps,
    evaluation_strategy="steps",
    save_steps=save_steps,
    run_name=f"n={n_ahead_global}_nt={n_ahead_talk_global}_np={n_passes_global}",
)

model_args = {
    "rank": 128,
    "update_proj_gap": 100,
    "galore_scale": 1,
    "proj_type": "topk",
    "lr": 1e-6,
    "weight_decay": 0.001,
    "warmup_steps": 20,
}

model_name = "unsloth/mistral-7b-bnb-4bit"
trainer = SFTrainer(model_name, 2048)

# patch unsloth and transformers
from transformers.models.mistral import configuration_mistral as original_configuration_mistral
from transformers.models.mistral import modeling_mistral as original_modeling_mistral

import configuration_mistral
import modeling_mistral
from patch_unsloth import patch

original_modeling_mistral.MistralModel = modeling_mistral.MistralModel
original_modeling_mistral.MistralForCausalLM = modeling_mistral.MistralForCausalLM
original_configuration_mistral.MistralConfig = configuration_mistral.MistralConfig
patch()

model, tokenizer = trainer.load_model(seed=12, use_gradient_checkpointing=True)

tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

mistral_model = model.base_model.model
mistral_config = mistral_model.config

mistral_config.max_thoughts = n_ahead_global + n_ahead_talk_global + 1
mistral_config.merged_talk_heads = True
mistral_config.merged_lm_and_talk_heads = False
mistral_config.merged_lm_and_think_heads = True
mistral_config.use_concat_talk_head = True
mistral_config.use_shallow_think = True
mistral_config.use_shallow_talk = False
mistral_config.use_complex_think_head = False
mistral_config.use_complex_talk_head = True
mistral_config.use_weighted_talk_head = True

# set config to model
mistral_model.config = mistral_config
mistral_model.max_thoughts = mistral_config.max_thoughts
mistral_model.merged_talk_heads = mistral_config.merged_talk_heads
mistral_model.merged_lm_and_talk_heads = mistral_config.merged_lm_and_talk_heads
mistral_model.merged_lm_and_think_heads = mistral_config.merged_lm_and_think_heads
mistral_model.use_concat_talk_head = mistral_config.use_concat_talk_head
mistral_model.use_shallow_think = mistral_config.use_shallow_think
mistral_model.use_shallow_talk = mistral_config.use_shallow_talk
mistral_model.use_complex_think_head = mistral_config.use_complex_think_head
mistral_model.use_complex_talk_head = mistral_config.use_complex_talk_head
mistral_model.use_weighted_talk_head = mistral_config.use_weighted_talk_head

special_tokens_to_add = []
if mistral_model.use_start_thought_token:
    special_tokens_to_add.append("<|startthought|>")
if mistral_model.use_end_thought_token:
    special_tokens_to_add.append("<|endthought|>")
if special_tokens_to_add:
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
    mistral_model.resize_token_embeddings(len(tokenizer))

mistral_model.tokenizer = tokenizer
mistral_model.gumbel_detach = True
mistral_model.include_policy_loss = True
mistral_model.use_end_thought_token = True
mistral_model.use_start_thought_token = True
mistral_model.n_ahead = n_ahead_global
mistral_model.n_ahead_talk = n_ahead_talk_global
mistral_model.n_passes = n_passes_global
mistral_model.n_tokens_print = global_gradient_accumulation_steps
mistral_model.gradient_accumulation_steps = global_gradient_accumulation_steps
mistral_model.residual_think_head = False
mistral_model.optimize_lm_head_only_at_start = False
mistral_model.gumbel_temperature = 1
mistral_model.wandb_enabled = True
mistral_model.original_mode = False
mistral_model.config_params = {}
mistral_model.run_start = int(time.time())
mistral_model.kill_after = 100

model.base_model = mistral_model
model.train()

trainer.prepare_dataset(ds_split=0.001)
trainer.process_model_and_datasets(**model_args)

trainer.train("./cache/quietstar/trains/01", args=training_args)
