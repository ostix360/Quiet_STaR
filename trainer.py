import json
import os

import transformers
import wandb
from datasets import concatenate_datasets, load_dataset, Dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils import logging
from unsloth import FastLanguageModel
from galore_torch import GaLoreAdamW8bit
import bitsandbytes as bnb
import torch
import time
from transformers import DataCollatorForLanguageModeling

logger = logging.get_logger(__name__)

user_key = "query"
assistant_key = "response"
conv_key = "messages"
prompt_format = """<|im_start|>system
You are an helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""


def formatting_prompts_func(examples):
    inputs = examples[user_key]
    outputs = examples[assistant_key]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt_format.format(input, output)
        texts.append(text)
    return {"text": texts, }


def formatting_prompt_for_conv(examples, eos):
    convs = examples[conv_key]
    output = []
    for conv in convs:
        context = """<|im_start|>system
You are an helpful assistant.<|im_end|>"""
        for message in conv:
            if message["role"] == user_key:
                context += "<|im_start|>user\n" + message["content"] + "<|im_end|>\n"
            elif message["role"] == assistant_key:
                context += "<|im_start|>assistant\n" + message["content"] + "<|im_end|>" + eos + "\n"
        output.append(context)
    return Dataset.from_dict({"text": output, })


def formatting_prompt_for_openmath(examples, eos):
    convs = examples[conv_key]
    output = []
    for conv in convs:
        context = """<|im_start|>system
You are an helpful assistant.<|im_end|>"""
        for message in conv:
            if message["from"] == user_key:
                context += "<|im_start|>user\n" + message["value"] + "<|im_end|>\n"
            elif message["from"] == assistant_key:
                context += "<|im_start|>assistant\n" + message["value"] + eos + "<|im_end|>\n"
        output.append(context)
    return Dataset.from_dict({"text": output, })


class SFTrainer:
    def __init__(self, model_name, max_seq_length, ):
        self.dataloader = None
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.max_seq_length = max_seq_length

    def load_model(self, seed, add_target_modules=[], r=32, lora_alpha=16, use_gradient_checkpointing=True):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        self.seed = seed

        self.model: PeftModel = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                            "down_proj", ] + add_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=seed,
            max_seq_length=self.max_seq_length,
        )
        self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def prepare_dataset(self, ds_split=1, **kwargs):
        global user_key, assistant_key, prompt_format, conv_key

        code_split = int(ds_split * 75_000)
        metamath_split = int(ds_split * 100_000)
        code2_split = int(ds_split * 10_000)
        ultra_chat_split = int(ds_split * 100_000)
        open_platypus_split = int(ds_split * 24_000)
        stack_mathQA_split = int(ds_split * 99_000)
        gpt4_llm_cleaned_split = int(ds_split * 50_000)
        wizardLM_alpaca_evol_instruct_70k_unfiltered_split = int(ds_split * 50_000)
        openhermes_2_5_1k_split = int(ds_split * 1_000)
        openmath_instruct = int(ds_split * 100_000)
        code_feedback_split = int(ds_split * 60_000)
        orca_math = int(ds_split * 100_000)

        eval_split = int(ds_split * 100) + 1

        if "oss_code_split" in kwargs:
            code_split = kwargs["oss_code_split"]
        if "metamath_split" in kwargs:
            metamath_split = kwargs["metamath_split"]
        if "evol_code_split" in kwargs:
            code2_split = kwargs["evol_code_split"]
        if "ultra_chat_split" in kwargs:
            ultra_chat_split = kwargs["ultra_chat_split"]
        if "eval_split" in kwargs:
            eval_split = kwargs["eval_split"]
        if "open_platypus_split" in kwargs:
            open_platypus_split = kwargs["open_platypus_split"]
        if "stack_mathQA_split" in kwargs:
            stack_mathQA_split = kwargs["stack_mathQA_split"]
        if "gpt4_llm_cleaned_split" in kwargs:
            gpt4_llm_cleaned_split = kwargs["gpt4_llm_cleaned_split"]
        if "wizardLM_alpaca_evol_split" in kwargs:
            wizardLM_alpaca_evol_instruct_70k_unfiltered_split = kwargs["wizardLM_alpaca_evol_split"]
        if "openhermes_2_5_1k_split" in kwargs:
            openhermes_2_5_1k_split = kwargs["openhermes_2_5_1k_split"]
        if "openmath_instruct" in kwargs:
            openmath_instruct = kwargs["openmath_instruct"]
        if "code_feedback_split" in kwargs:
            code_feedback_split = kwargs["code_feedback_split"]
        if "orca_math" in kwargs:
            orca_math = kwargs["orca_math"]

        t_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=f"train[:{code_split}]")
        e_code_dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=f"train[-{eval_split}:]")

        t_metamath_dataset = load_dataset("meta-math/MetaMathQA", split=f"train[:{metamath_split}]")  # max 395k
        e_metamath_dataset = load_dataset("meta-math/MetaMathQA", split=f"train[-{eval_split}:]")

        t_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split=f"train[:{code2_split}]")
        e_code2_dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split=f"train[-{eval_split}:]")

        t_ultra_chat_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{ultra_chat_split}]")
        e_ultra_chat_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"test_sft[-{eval_split}:]")

        t_open_platypus_dataset = load_dataset("garage-bAInd/Open-Platypus", split=f"train[:{open_platypus_split}]")
        e_open_platypus_dataset = load_dataset("garage-bAInd/Open-Platypus", split=f"train[-{eval_split}:]")

        t_stack_mathQA_dataset = load_dataset("math-ai/StackMathQA", "stackmathqa100k",
                                              split=f"train[:{stack_mathQA_split}]")
        e_stack_mathQA_dataset = load_dataset("math-ai/StackMathQA", "stackmathqa100k", split=f"train[-{eval_split}:]")

        t_gpt4_llm_cleaned_dataset = load_dataset("teknium/GPT4-LLM-Cleaned", split=f"train[:{gpt4_llm_cleaned_split}]")
        e_gpt4_llm_cleaned_dataset = load_dataset("teknium/GPT4-LLM-Cleaned", split=f"train[-{eval_split}:]")

        t_wizardLM_alpaca_evol_instruct_70k_unfiltered_dataset = load_dataset(
            "cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered",
            split=f"train[:{wizardLM_alpaca_evol_instruct_70k_unfiltered_split}]")
        e_wizardLM_alpaca_evol_instruct_70k_unfiltered_dataset = load_dataset(
            "cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered", split=f"train[-{eval_split}:]")

        t_openhermes_2_5_1k_dataset = load_dataset("HuggingFaceH4/OpenHermes-2.5-1k-longest",
                                                   split=f"train_sft[:{openhermes_2_5_1k_split}]")
        e_openhermes_2_5_1k_dataset = load_dataset("HuggingFaceH4/OpenHermes-2.5-1k-longest",
                                                   split=f"test_sft[-{eval_split}:]")

        t_openmath_instruct_dataset = load_dataset("nz/openmathinstruct-math-processed",
                                                   split=f"train[:{openmath_instruct}]")
        e_openmath_instruct_dataset = load_dataset("nz/openmathinstruct-math-processed", split=f"train[-{eval_split}:]")

        t_code_feed_back_dataset = load_dataset("m-a-p/Code-Feedback", split=f"train[:{code_feedback_split}]")
        e_code_feed_back_dataset = load_dataset("m-a-p/Code-Feedback", split=f"train[-{eval_split}:]")

        t_orca_math_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split=f"train[:{orca_math}]")
        e_orca_math_dataset = load_dataset("microsoft/orca-math-word-problems-200k", split=f"train[-{eval_split}:]")

        user_key = "query"
        assistant_key = "response"
        t_metamath_formated_datasets = t_metamath_dataset.map(formatting_prompts_func, batched=True,
                                                              remove_columns=["query", "response"])
        e_metamath_formated_datasets = e_metamath_dataset.map(formatting_prompts_func, batched=True,
                                                              remove_columns=["query", "response"])

        user_key = "problem"
        assistant_key = "solution"

        t_code_formated_datasets = t_code_dataset.map(formatting_prompts_func, batched=True,
                                                      remove_columns=["problem", "solution"])
        e_code_formated_datasets = e_code_dataset.map(formatting_prompts_func, batched=True,
                                                      remove_columns=["problem", "solution"])

        user_key = "instruction"
        assistant_key = "response"

        t_code2_formated_datasets = t_code2_dataset.map(formatting_prompts_func, batched=True,
                                                        remove_columns=["instruction", "response"])
        e_code2_formated_datasets = e_code2_dataset.map(formatting_prompts_func, batched=True,
                                                        remove_columns=["instruction", "response"])

        user_key = "user"
        assistant_key = "assistant"

        t_ultra_chat_formated_datasets = formatting_prompt_for_conv(t_ultra_chat_dataset, eos=self.tokenizer.eos_token)
        e_ultra_chat_formated_datasets = formatting_prompt_for_conv(e_ultra_chat_dataset, eos=self.tokenizer.eos_token)

        t_openhermes_2_5_1k_formated_datasets = formatting_prompt_for_conv(t_openhermes_2_5_1k_dataset,
                                                                           eos=self.tokenizer.eos_token)
        e_openhermes_2_5_1k_formated_datasets = formatting_prompt_for_conv(e_openhermes_2_5_1k_dataset,
                                                                           eos=self.tokenizer.eos_token)

        t_code_feed_back_formated_datasets = formatting_prompt_for_conv(t_code_feed_back_dataset,
                                                                        eos=self.tokenizer.eos_token)
        e_code_feed_back_formated_datasets = formatting_prompt_for_conv(e_code_feed_back_dataset,
                                                                        eos=self.tokenizer.eos_token)

        user_key = "human"
        assistant_key = "gpt"
        conv_key = "conversations"

        t_openmath_instruct_formated_datasets = formatting_prompt_for_openmath(t_openmath_instruct_dataset,
                                                                               eos=self.tokenizer.eos_token)
        e_openmath_instruct_formated_datasets = formatting_prompt_for_openmath(e_openmath_instruct_dataset,
                                                                               eos=self.tokenizer.eos_token)

        user_key = "instruction"
        assistant_key = "output"

        t_open_platypus_formated_datasets = t_open_platypus_dataset.map(formatting_prompts_func, batched=True,
                                                                        remove_columns=["instruction", "output"])
        e_open_platypus_formated_datasets = e_open_platypus_dataset.map(formatting_prompts_func, batched=True,
                                                                        remove_columns=["instruction", "output"])

        t_gpt4_llm_cleaned_formated_datasets = t_gpt4_llm_cleaned_dataset.map(formatting_prompts_func, batched=True,
                                                                              remove_columns=["instruction", "output"])
        e_gpt4_llm_cleaned_formated_datasets = e_gpt4_llm_cleaned_dataset.map(formatting_prompts_func, batched=True,
                                                                              remove_columns=["instruction", "output"])

        t_wizardLM_alpaca_evol_instruct_70k_unfiltered_formated_datasets = t_wizardLM_alpaca_evol_instruct_70k_unfiltered_dataset.map(
            formatting_prompts_func, batched=True, remove_columns=["instruction", "output"])
        e_wizardLM_alpaca_evol_instruct_70k_unfiltered_formated_datasets = e_wizardLM_alpaca_evol_instruct_70k_unfiltered_dataset.map(
            formatting_prompts_func, batched=True, remove_columns=["instruction", "output"])

        user_key = "Q"
        assistant_key = "A"

        t_stack_mathQA_formated_datasets = t_stack_mathQA_dataset.map(formatting_prompts_func, batched=True,
                                                                      remove_columns=["Q", "A"])
        e_stack_mathQA_formated_datasets = e_stack_mathQA_dataset.map(formatting_prompts_func, batched=True,
                                                                      remove_columns=["Q", "A"])

        user_key = "question"
        assistant_key = "answer"

        t_orca_math_formated_datasets = t_orca_math_dataset.map(formatting_prompts_func, batched=True,
                                                                remove_columns=["question", "answer"])
        e_orca_math_formated_datasets = e_orca_math_dataset.map(formatting_prompts_func, batched=True,
                                                                remove_columns=["question", "answer"])

        t_list = []
        e_list = []

        def add_ds(t_ds, e_ds):
            t_list.append(t_ds)
            e_list.append(e_ds)

        add_ds(t_metamath_formated_datasets, e_metamath_formated_datasets)
        add_ds(t_code_formated_datasets, e_code_formated_datasets)
        add_ds(t_code2_formated_datasets, e_code2_formated_datasets)
        add_ds(t_ultra_chat_formated_datasets, e_ultra_chat_formated_datasets)
        add_ds(t_open_platypus_formated_datasets, e_open_platypus_formated_datasets)
        add_ds(t_stack_mathQA_formated_datasets, e_stack_mathQA_formated_datasets)
        add_ds(t_gpt4_llm_cleaned_formated_datasets, e_gpt4_llm_cleaned_formated_datasets)
        add_ds(t_wizardLM_alpaca_evol_instruct_70k_unfiltered_formated_datasets,
               e_wizardLM_alpaca_evol_instruct_70k_unfiltered_formated_datasets)
        add_ds(t_openhermes_2_5_1k_formated_datasets, e_openhermes_2_5_1k_formated_datasets)
        add_ds(t_openmath_instruct_formated_datasets, e_openmath_instruct_formated_datasets)
        add_ds(t_code_feed_back_formated_datasets, e_code_feed_back_formated_datasets)
        # add_ds(t_orca_math_formated_datasets, e_orca_math_formated_datasets)

        self.t_formated_datasets = concatenate_datasets(t_list).shuffle(seed=self.seed)
        self.e_formated_datasets = concatenate_datasets(e_list).shuffle(seed=self.seed)

        # Training
        self.batch_size = 3
        self.steps = len(self.t_formated_datasets)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.t_formated_datasets, self.e_formated_datasets

    def tokenize(self, examples):
        outputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    def get_args(self, save_dire, learning_rate=5e-5, save_total_limit=1, batch_size=None, gradient_checkpointing=True,
                 steps=None, seed=None):
        if batch_size is not None:
            self.batch_size = batch_size
            logger.info(f"Changing batch_size from {self.batch_size} to {batch_size}")
        if steps is not None:
            self.steps = steps
            logger.info(f"Changing steps from {self.steps} to {steps}")
        if seed is not None:
            self.seed = seed
            logger.info(f"Changing seed from {self.seed} to {seed}")
            logger.warning("seed has been changed, deterministic results are not guaranteed")
        from transformers import TrainingArguments
        self.args = TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_checkpointing=gradient_checkpointing,
            gradient_accumulation_steps=1,
            load_best_model_at_end=False,
            resume_from_checkpoint=save_dire,
            warmup_steps=20,
            num_train_epochs=1,
            report_to=["none"],
            evaluation_strategy="steps",
            eval_steps=self.steps // self.batch_size,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            bf16_full_eval=torch.cuda.is_bf16_supported(),
            fp16_full_eval=not torch.cuda.is_bf16_supported(),
            logging_steps=50 // self.batch_size,
            optim="adamw_8bit",
            max_steps=self.steps // self.batch_size,
            save_total_limit=save_total_limit,
            save_strategy="steps",
            save_steps=self.steps // self.batch_size,
            weight_decay=0.02,
            lr_scheduler_type="linear",
            seed=self.seed,
            output_dir="./trains" + save_dire,
        )
        return self.args

    def process_model_and_datasets(self, **args):
        # check if args contains the required keys
        if "rank" not in args:
            raise ValueError("rank must be specified in args")
        if "update_proj_gap" not in args:
            raise ValueError("update_proj_gap must be specified in args")
        if "galore_scale" not in args:
            raise ValueError("galore_scale must be specified in args")
        if "proj_type" not in args:
            raise ValueError("proj_type must be specified in args")
        if "lr" not in args:
            raise ValueError("lr must be specified in args")
        if "weight_decay" not in args:
            raise ValueError("weight_decay must be specified in args")
        if "warmup_steps" not in args:
            raise ValueError("warmup_steps must be specified in args")
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]

        optimizer_dict = {}
        for p in self.model.parameters():
            if p.requires_grad:
                if id(p) in id_galore_params:
                    optimizer_dict[p] = GaLoreAdamW8bit([{'params': [p], 'rank': args["rank"],
                                                          'update_proj_gap': args["update_proj_gap"] * 2,
                                                          'scale': args["galore_scale"], 'proj_type': args["proj_type"]}],
                                                        lr=args["lr"],
                                                        weight_decay=args["weight_decay"])
                else:
                    optimizer_dict[p] = bnb.optim.Adam8bit([p], lr=args["lr"], weight_decay=args["weight_decay"])

        # get scheduler dict
        scheduler_dict = {}
        for p in self.model.parameters():
            if p.requires_grad:
                scheduler_dict[p] = transformers.get_linear_schedule_with_warmup(
                    optimizer=optimizer_dict[p],
                    num_training_steps=self.steps * 2,
                    num_warmup_steps=args["warmup_steps"] * 2,
                )

        def optimizer_hook(p):
            if p.grad is None:
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        # tokenize ds and create dataloader
        self.t_formated_datasets = (self.t_formated_datasets.map(self.tokenize, batched=True)
                                    .remove_columns(['type', 'original_question', 'text', 'lang', 'raw_index', 'index', 'seed', 'openai_fingerprint', 'input', 'data_source', 'meta',]))

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.dataloader = DataLoader(self.t_formated_datasets, batch_size=self.batch_size,
                                                      shuffle=True, collate_fn=data_collator)

    def train(self, save_folder, dataset_text_field="text", train_dataset=None, eval_dataset=None, data_collator=None,
              args=None, ):
        if train_dataset is None:
            if self.t_formated_datasets is None:
                raise ValueError("train_dataset must be specified")
            train_dataset = self.t_formated_datasets
        else:
            logger.info("Using the specified train_dataset")
        if eval_dataset is None:
            if self.e_formated_datasets is None:
                raise ValueError("eval_dataset must be specified")
            eval_dataset = self.e_formated_datasets
        else:
            logger.info("Using the specified eval_dataset")
        if args is None:
            if self.args is None:
                raise ValueError("args must be specified")
            args = self.args
        else:
            logger.info("Using the specified args")

        global_step = 0
        update_step = 0
        beginning_step = 0
        tokens_seen = 0
        tokens_seen_before = 0

        pad_idx = self.tokenizer.pad_token_id
        update_time = time.time()
        local_step = 0  # when continue_from is used, local_step != global_step

        pbar = tqdm(total=self.steps - update_step, desc="Update steps", ncols=80)

        for batch_idx, batch in enumerate(self.dataloader):

            global_step += 1
            local_step += 1

            if update_step > self.steps:
                logger.info(f"Reached max number of update steps (f{self.steps}). Stopping training.")
                break

            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item()

            loss = self.model(**batch, labels=labels).loss # TODO Error pointing here
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            if global_step % args.gradient_accumulation_steps != 0:
                continue

            # The below code is only executed during the update step

            # add grad clipping
            if args.max_grad_norm != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

            pbar.update(1)

            update_step += 1
            update_time = time.time() - update_time

            # save checkpoint by save_every
            if local_step > args.gradient_accumulation_steps and update_step % args.save_steps == 0 and False:
                raise NotImplementedError("save_every is not implemented yet")
                current_model_directory = f"{args.save_dir}/model_{update_step}"
                logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
                os.makedirs(args.save_dir, exist_ok=True)
                self.model.module.save_pretrained(current_model_directory, max_shard_size='100GB')

                optimizer_checkpoint = {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir,
                    "dtype": self.model.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

                # save wandb related info
                wandb_info = {
                    "wandb_id": wandb.run.id,
                }
                with open(f"{args.save_dir}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

            # evaluation
            if update_step % args.eval_steps == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    self.model, preprocess_batched, pad_idx, global_rank, world_size, device,
                    args.per_device_eval_batch_size
                )

                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                },
                    step=global_step,
                )
                logger.info(f"Eval loss at step {update_step}: {total_loss}")

            lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation_steps

            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.per_device_eval_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
            },
                step=global_step,
            )
            update_time = time.time()

        # ##############################
        # END of training loop
        # ##############################
        logger.info("Training finished")
        pbar.close()

        current_model_directory = f"{args.save_dir}/model_{update_step}"
        if not os.path.exists(current_model_directory):
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            self.model.module.save_pretrained(current_model_directory)

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": self.model.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

        self.model.save_pretrained(save_folder)
        return trainer

    def merge_and_save(self, save_folder):
        self.model.save_pretrained_merged(save_folder, self.tokenizer, save_method="merged_16bit", )

    def save_model(self, save_folder):
        self.model.save_pretrained(save_folder)
