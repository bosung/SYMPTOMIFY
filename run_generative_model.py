#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random
from pathlib import Path
import json

import datasets
import nltk
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    default_data_collator,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name, is_offline_mode
from transformers.utils.versions import require_version
from torch.utils.tensorboard import SummaryWriter
from metrics_vsed import VSEDMetric
from models import MultiLabelBart, MultimodalBart, MultiTaskBart
from utils import get_num_labels, postprocess_text

import pdb

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--dev_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument("--pc_file", type=str, default=None)
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--tb_log_dir", type=str, default="runs/null", help="Tensorboard log directory")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_pretrain", action="store_true")
    parser.add_argument("--do_debug", action="store_true")
    parser.add_argument("--aux_task", choices=["multi-cls", "match-pred"])
    parser.add_argument("--multi_task", action="store_true")
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--multi_cls", action="store_true")
    parser.add_argument("--paraph_cls", action="store_true")
    parser.add_argument("--symptoms_file", type=str, default=None)
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    writer = SummaryWriter(log_dir=args.tb_log_dir)

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process: 
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    data_files = {}
    if args.do_train or args.do_pretrain:
        data_files["train"] = args.train_file
        data_files["validation"] = args.dev_file
        extension = args.train_file.split(".")[-1]
        if args.multi_task:
            if args.multi_task == "match_pred":
                pass
    if args.do_predict:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    
    raw_datasets = load_dataset(extension, data_files=data_files)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Model
    if args.model_name_or_path:
        if args.multi_task:
            # Multi-task learning setting
            mc_num_labels = get_num_labels(args.symptoms_file)
            config.mc_num_labels = mc_num_labels
            config.pc_num_labels = 2
            config.output_hidden_states = True
            if args.paraph_cls:
                config.pc_problem_type = "single_label_classification"
                #config.problem_type = "contrastive"
            model_cls = MultiTaskBart(config)
        elif args.multi_cls and (not args.multi_task):
            config.output_hidden_states = True
            config.num_labels = get_num_labels(args.symptoms_file)
            config.problem_type = "multi_label_classification"
            model_cls = MultiLabelBart(config)
        elif args.metadata:
            config.n_vax_type = 75
            config.n_vax_name = 150
            model_cls = MultimodalBart(config)
        else:
            model_cls = AutoModelForSeq2SeqLM

        model = model_cls.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.do_train or args.do_pretrain:
        column_names = raw_datasets["train"].column_names
    elif args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        raise ValueError("Need datasets")

    # Get the column names for input/target.
    if args.text_column is None:
        raise ValueError("--text_colum value is required")
    text_column = args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    if args.summary_column is None:
        logger.info("Summary column is None. summary_column = None")
        #summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        summary_column = None
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # 1. preprocess inputs
        inputs = examples[text_column]

        # 2. preprocess targets
        targets = [", ".join(x) for x in examples[summary_column]]        
        
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        # -> model_inputs["input_ids"], model_inputs["attention_mask"]

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        if args.multi_cls:
            # class_labels
            model_inputs["class_labels"] = [np.zeros(mc_num_labels) for l in examples["symptom_ids"]]
            for i, _labels in enumerate(examples["symptom_ids"]):
                for l in _labels:
                    model_inputs["class_labels"][i][l-1] = 1 # label should (symptom_id - 1): symptom ids range: [1, # of symptoms]
                assert model_inputs["class_labels"][i].sum() > 0
        if args.metadata:
            model_inputs["sex_ids"] = examples["sex"]
            model_inputs["age_ids"] = examples["age"]
            model_inputs["vax_type_ids"] = examples["vax_type"]
            model_inputs["vax_name_ids"] = examples["vax_name"] 

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["vid"] = examples["vid"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            keep_in_memory=True,
            desc="Running tokenizer on dataset",
        )

    if args.do_train or args.do_pretrain:
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 1):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        batch_gold_entities = []
        ge_batch = []
        for line in open(args.dev_file):
            if len(ge_batch) == args.per_device_eval_batch_size:
                batch_gold_entities.append(ge_batch)
                ge_batch = []
            obj = json.loads(line.strip())
            ge_batch.append({"vid": obj["vid"], "symptoms": obj["symptoms"]})
        if len(ge_batch) != 0:
            batch_gold_entities.append(ge_batch)

        assert len(batch_gold_entities) == len(eval_dataloader)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        if args.multi_task:
            train_multitask(args,config, model, optimizer, tokenizer, accelerator, train_dataloader, eval_dataloader,
                batch_gold_entities, len(train_dataset))
        else:
            train(args,config, model, optimizer, tokenizer, accelerator, train_dataloader, eval_dataloader,
                batch_gold_entities, len(train_dataset))

    if args.do_predict:
        test_dataset = processed_datasets["test"]

        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.use_fp16 else None,
        )
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        batch_gold_entities = []
        ge_batch = []
        for line in open(args.test_file):
            if len(ge_batch) == args.per_device_eval_batch_size:
                batch_gold_entities.append(ge_batch)
                ge_batch = []
            obj = json.loads(line.strip())
            ge_batch.append({"vid": obj["vid"], "symptoms": obj["symptoms"]})
        if len(ge_batch) != 0:
            batch_gold_entities.append(ge_batch)

        assert len(batch_gold_entities) == len(test_dataloader)

        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        evaluate(args, config, model, tokenizer, accelerator, test_dataloader, batch_gold_entities)


def train_multitask(args,config, model, optimizer, tokenizer, accelerator, train_dataloader, eval_dataloader,
            batch_gold_entities, len_train_dataset):
    # prepare additional datasets for mutltask

    data_files = {}
    if args.do_train and (args.pc_file and args.paraph_cls):
        data_files["mp_train"] = args.pc_file
        extension = args.pc_file.split(".")[-1]
    raw_datasets_mp = load_dataset(extension, data_files=data_files)

    padding = "max_length" if args.pad_to_max_length else False
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples["symptom_text1"], examples["symptom_text2"]) # two sentences are concatnated in the tokenizer function
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        result["labels"] = examples["labels"]
        return result  # result: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    with accelerator.main_process_first():
        processed_datasets = raw_datasets_mp.map(
            preprocess_function,
            batched=True,
            remove_columns=(raw_datasets_mp["mp_train"].column_names),
            desc="Running tokenizer on dataset",
            keep_in_memory=True
        )
    # data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt",
    #                         pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    # # data_collator = default_data_collator
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    
    pc_train_dataset = processed_datasets["mp_train"]
    pc_train_dataloader = DataLoader(
        pc_train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, pc_train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, pc_train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil((len(train_dataloader) + len(pc_train_dataloader)) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num examples [main] = {len_train_dataset}")
    logger.info(f"  Num examples [PC] = {len(pc_train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step in range(len(train_dataloader)):
            task_type = "gen"
            try: # should be checked!! next error
                batch = next(iter_gen) 
            except:
                iter_gen = iter(train_dataloader)
                batch = next(iter_gen) 
            outputs = model(
                task_type = task_type,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            # for auxilary tasks
            if args.multi_cls:
                task_type = "mc"
                outputs = model(
                    task_type = task_type,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    class_labels=batch["class_labels"]
                )
                loss_mc = outputs.loss
                loss += (0.1) * loss_mc

            if args.paraph_cls:
                task_type = "pc"
                try:
                    batch = next(iter_mp)
                except:
                    iter_mp = iter(pc_train_dataloader)
                    batch = next(iter_mp)

                outputs = model(
                    task_type = task_type,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss_mp = outputs.loss
                loss += (0.1) * loss_mp

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # evaluate on the validation set
        model.eval()
        evaluate(args, config, model, tokenizer, accelerator, eval_dataloader, batch_gold_entities)
        # end of epoch

    # end of traning
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


def train(args,config, model, optimizer, tokenizer, accelerator, train_dataloader, eval_dataloader,
            batch_gold_entities, len_train_dataset):
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            del batch["vid"]
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        # evaluate on the validation set
        model.eval()
        evaluate(args, config, model, tokenizer, accelerator, eval_dataloader, batch_gold_entities)
        # end of epoch

    # end of traning
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


def evaluate(args, config, model, tokenizer, accelerator, test_dataloader, batch_gold_entities):
    # Metric
    metric = load_metric("rouge")
    metric_vsed = VSEDMetric()

    progress_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process)
    model.eval()

    gen_kwargs = {
        "max_length": args.max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
    }
    for step, (batch, ref) in enumerate(zip(test_dataloader, batch_gold_entities)):
        with torch.no_grad():
            if args.metadata:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    **batch,
                    **gen_kwargs,
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
            
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            assert batch["vid"][0] == ref[0]["vid"]
            metric_vsed.add_batch(predictions=decoded_preds, references=ref)

            progress_bar.update(1)
    
    # evaluate rouge
    result = metric.compute(use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    logger.info(result)

    eval_results = metric_vsed.compute()
    for k in eval_results:
        logger.info(f"Predict result [{k}]: {eval_results[k]}")

if __name__ == "__main__":
    main()