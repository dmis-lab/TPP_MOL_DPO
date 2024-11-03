from __future__ import annotations

import argparse
import os
import warnings

import flax.jax_utils
import jax
import numpy as np
import pandas as pd
import safe as sf
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.training.common_utils import shard, shard_prng_key
from flax.training.train_state import TrainState
from rdkit import Chem
from transformers import FlaxGPT2LMHeadModel, PreTrainedTokenizerBase
from utils import AverageMeter, Evaluator

from dataset import create_train_dataloader
from training import create_train_state, generation_step, training_step

warnings.filterwarnings("ignore")


def evaluate(
    args: argparse.Namespace,
    step: int,
    epoch: int,
    state: TrainState,
    evaluator: Evaluator,
    tokenizer: PreTrainedTokenizerBase,
    eval_batches: int,
):
    tokens = np.array([[tokenizer.bos_token_id]] * args.batch_size, dtype=np.int32)
    tokens = shard(tokens)

    generated, average_meter = [], AverageMeter()
    for i in tqdm.trange(eval_batches, desc="Generation", dynamic_ncols=True):
        rng = shard_prng_key(epoch + jax.random.PRNGKey(i))
        preds, metrics = generation_step(state, tokens, rng, args.max_length)
        average_meter.update(**unreplicate(metrics))

        for pred in jax.device_get(preds).reshape(-1, *preds.shape[2:]):
            pred = pred[np.cumsum(pred == tokenizer.eos_token_id) < 1]
            generated.append(tokenizer.decode(pred.tolist(), skip_special_tokens=True))

    # Evaluate the generated mols with considering the format type. After calculating
    # validity and uniqueness of the generated samples, the invalid and duplicated
    # samples will be removed for ADMET-AI evaluation.
    smiles = [sf.decode(x, canonical=False, ignore_errors=True) for x in generated]
    mols = pd.DataFrame((generated, smiles)).T
    mols.columns = ["safe", "smiles"]

    mols["Validity"] = mols["smiles"].notnull()
    mols["Uniqueness"] = mols["smiles"].nunique() / mols["smiles"].notnull().sum()
    mols = mols[mols.smiles.notnull()].drop_duplicates("smiles")

    admet = evaluator([x for x in smiles if x and Chem.MolFromSmiles(x)])
    admet = pd.merge(mols, admet, left_on="smiles", right_index=True, how="outer")

    # Compute the average metrics of the evaluated results and log them with the
    # validation scores.
    metrics = average_meter.summary(prefix="valid/")
    for name in args.eval_metrics:
        metrics[f"valid/{name}"] = admet[name].mean()
    metrics["epoch"] = epoch
    wandb.log(metrics, step)


def main(args: argparse.Namespace):
    evaluator = Evaluator()
    model = FlaxGPT2LMHeadModel.from_pretrained("datamol-io/safe-gpt", from_pt=True)
    tokenizer = sf.SAFETokenizer.from_pretrained("datamol-io/safe-gpt").get_pretrained()
    dataloader = create_train_dataloader(args, tokenizer)

    state = create_train_state(args, model, steps_per_epoch=len(dataloader))
    state = flax.jax_utils.replicate(state)

    wandb.init(name=args.name, project=args.project, config=args)
    average_meter, step = AverageMeter(use_latest=["learning_rate"]), 0

    # Before training, we will evaluate the initial performance of the model.
    evaluate(
        args=args,
        step=0,
        epoch=0,
        state=state,
        evaluator=evaluator,
        tokenizer=tokenizer,
        eval_batches=args.eval_batches,
    )

    for epoch in range(args.epochs):
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True):
            state, metrics = training_step(state, shard(jax.tree.map(np.array, batch)))
            average_meter.update(**unreplicate(metrics))
            step += 1

            if args.log_interval > 0 and step % args.log_interval == 0:
                metrics = average_meter.summary(prefix="train/")
                metrics["epoch"] = step / len(dataloader)
                wandb.log(metrics, step)

        if (
            args.eval_interval > 0
            and (epoch + 1) % args.eval_interval == 0
            or epoch == args.epochs - 1
        ):
            evaluate(
                args=args,
                step=step,
                epoch=epoch + 1,
                state=state,
                evaluator=evaluator,
                tokenizer=tokenizer,
                eval_batches=args.eval_batches,
            )

    os.makedirs(args.output_dir, exist_ok=True)
    model.params = unreplicate(state.params["act"])
    model.save_pretrained(os.path.join(args.output_dir, args.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)

    parser.add_argument("--target-columns", nargs="+")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--penalty-beta", type=float, default=0.1)
    parser.add_argument("--eval-metrics", nargs="+", default=["Validity"])

    parser.add_argument("--use-moco", action="store_true", default=False)
    parser.add_argument("--jacmom", type=float, default=0.99)
    parser.add_argument("--lammom", type=float, default=0.5)
    parser.add_argument("--lamreg", type=float, default=0.1)

    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-b1", type=float, default=0.9)
    parser.add_argument("--adam-b2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=0)

    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--ipaddr")
    parser.add_argument("--hostname")
    parser.add_argument("--output-dir", default="./")
    main(parser.parse_args())
