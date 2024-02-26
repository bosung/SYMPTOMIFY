# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from typing import Optional, Union, Tuple

from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from metrics_cls import VSEDMetric
import pdb
import torch
from torch.nn import Sigmoid
import torch.nn as nn

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

accelerator = Accelerator()
device = accelerator.device


class BertCNNForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        filter_sizes = [1,2,3,4,5]
        num_filters = 32
        self.convs1 = nn.ModuleList([nn.Conv2d(4, num_filters, (K, config.hidden_size)) for K in filter_sizes])
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, self.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        x = outputs.hidden_states[-4:]
        x = torch.stack(x, dim=1) # (N, 4, seq_len, hidden_size)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, dim=1) # (N, 32*5)
        x = self.dropout(x)  
        logits = self.fc1(x)

        loss = None
        if labels is not None:
            if self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise ValueError()
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
    )
    parser.add_argument(
        "--symptoms_file", type=str, default='data/symptoms.tsv',
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
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
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--do_train", action="store_true", help="Wheter or not to train")
    parser.add_argument("--do_predict", action="store_true", help="Wheter or not to test (inference)")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    return args


def predict(args, logger, processed_datasets, data_collator, model, metric, writer, accelerator, num_labels):
    test_dataset = processed_datasets["test"]
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    progress_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process)

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.eval()
    for step, batch in enumerate(test_dataloader):
        outputs = model(**batch)
    
        predictions = (Sigmoid()(outputs.logits) > args.threshold).float() # multi_lable cls uses BCEWithLogitsLoss()

        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )
        progress_bar.update(1)

    eval_metric = metric.compute()
    logger.info(f"Predict result (th: {args.threshold}): {eval_metric}")


def train(args, logger, processed_datasets, data_collator, model, metric, writer):
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

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
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    gstep = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            gstep += 1
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            if gstep % 40 == 0:
                writer.add_scalar('Loss/train', loss.item(), gstep)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        best_f1, best_th = 0, args.threshold
        # get threshold on the dev set
        for th in [0.05, 0.1, 0.15, 0.2, 0.25]:
            for step, batch in enumerate(eval_dataloader):
                outputs = model(**batch)
                predictions = (Sigmoid()(outputs.logits) > th).float() # multi_lable cls uses BCEWithLogitsLoss()
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(batch["labels"]),
                )
            eval_metric = metric.compute()
            logger.info(f"th: {th} | epoch {epoch+1}: {eval_metric}")
            if (eval_metric["full"]['macro_f1'] + eval_metric["full"]['micro_f1'])/2 > best_f1:
                best_f1 = (eval_metric["full"]['macro_f1'] + eval_metric["full"]['micro_f1'])/2
                best_th = th
        logger.info("The best (macro f1 + macro f1)/2 score: %.4f with threshold %.2f" % (best_f1, best_th))

    # end of traning
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training")
    
    return model


if __name__ == "__main__":
    args = parse_args()

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
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, args.fp16))

    # Labels: We need "label_list" and "num_labels":
    label_list = []
    with open(args.symptoms_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            toks = line.split("\t")
            label_list.append(toks[0])
    num_labels = len(label_list)       

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    config.output_hidden_states = True
    config.problem_type = "multi_label_classification"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    #model = AutoModelForSequenceClassification.from_pretrained(
    model = BertCNNForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # Preprocessing the datasets
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples["symptom_text"],)
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
        #result["labels"] = [[0] * num_labels for l in examples["symptom_ids"]]
        
        result["labels"] = [np.zeros(num_labels) for l in examples["symptom_ids"]]
        for i, _labels in enumerate(examples["symptom_ids"]):
            for l in _labels:
                result["labels"][i][l] = 1 # label: symptom ids range: [0, # of symptoms-1]
        return result  # result: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # Loading the dataset from local csv or json file.
    data_files = {}
    if args.do_train:
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
    if args.do_predict:
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=(raw_datasets["train"].column_names if args.do_train else raw_datasets["test"].column_names),
            desc="Running tokenizer on dataset",
            keep_in_memory=True
        )

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    metric = VSEDMetric()

    if args.do_train:
        train(args, logger, processed_datasets, data_collator, model, metric, writer)
    
    if args.do_train and args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.output_dir,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        predict(args, logger, processed_datasets, data_collator, model, metric, writer)
    elif args.do_predict: # only inference
        predict(args, logger, processed_datasets, data_collator, model, metric, writer, accelerator, num_labels)
