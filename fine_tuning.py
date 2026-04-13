# -*- coding: utf-8 -*-
"""CPU-only fine-tuning script for JSON extraction examples.

This version avoids CUDA-specific libraries and can run on a standard
CPU-only EC2 instance. It fine-tunes a small causal language model on
`json_extraction_dataset_500.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_records(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must be a non-empty JSON list.")
    return records


def format_prompt(record: dict) -> str:
    return (
        f"### Input: {record['input']}\n"
        f"### Output: {json.dumps(record['output'], ensure_ascii=False)}"
    )


def build_dataset(records: list[dict]) -> Dataset:
    formatted = [format_prompt(record) for record in records]
    return Dataset.from_dict({"text": formatted})


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only fine-tuning for product extraction examples")
    parser.add_argument("--dataset", default="json_extraction_dataset_500.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    records = load_records(dataset_path)
    print(records[0])

    print("Device: cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(records)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        use_cpu=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    test_prompt = (
        "### Input: <div class='product'><h2>iPad Air</h2><span class='price'>$1344</span>"
        "<span class='category'>audio</span><span class='brand'>Dell</span></div>\n### Output:"
    )
    inputs = tokenizer(test_prompt, return_tensors="pt")
    generated = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
