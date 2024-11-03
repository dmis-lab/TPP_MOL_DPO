from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import fsspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class MyDataset(Dataset):
    texts: list[str]
    labels: np.ndarray
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 128

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[i],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return (
            torch.tensor(encoding["input_ids"], dtype=torch.int32),
            torch.tensor(encoding["attention_mask"], dtype=torch.int32),
            torch.tensor(self.labels[i], dtype=torch.float32),
        )


def create_train_dataloader(
    args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase
) -> DataLoader:
    with fsspec.open(args.dataset) as fp:
        data = pd.read_csv(fp)

    labels = []
    for target in args.target_columns:
        name, direction = target.split(":")[:2]
        labels.append(data[name] * (1 if direction == "max" else -1))

    dataset = MyDataset(
        texts=data["safe"],
        labels=np.stack(labels, axis=1),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.shuffle_seed),
        persistent_workers=True,
    )
