from __future__ import annotations

import argparse
import multiprocessing as mp

import pandas as pd
import safe as sf
import torch
import tqdm
from admet_ai import ADMETModel
from rdkit import Chem, rdBase
from rdkit.Contrib.SA_Score import sascorer  # type: ignore
from transformers import GPT2LMHeadModel

rdBase.DisableLog("rdApp.*")


class Evaluator:
    def __init__(self):
        self.admet_ai = ADMETModel()

    def __call__(self, smiles_list: list[str]) -> pd.DataFrame:
        mols = [Chem.MolFromSmiles(x) for x in smiles_list]
        sa_scores = [sascorer.calculateScore(m) for m in mols]
        max_ring = [max(map(len, m.GetRingInfo().AtomRings() or [[]])) for m in mols]

        admet = self.admet_ai.predict(smiles_list)
        admet["SAScore"] = sa_scores
        admet["CycleScore"] = [max(x - 6, 0) for x in max_ring]
        admet["plogP"] = admet["logP"] - admet["SAScore"] - admet["CycleScore"]
        return admet


def decode_smiles_from_valid_safe(safe: str) -> str | None:
    smiles = sf.decode(safe, canonical=False, ignore_errors=True)
    if smiles and Chem.MolFromSmiles(smiles) is not None:
        return smiles
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", default="safe-de-novo-dataset.csv")
    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained("datamol-io/safe-gpt")
    model = model.bfloat16().cuda().eval().requires_grad_(False)
    tokenizer = sf.SAFETokenizer.from_pretrained("datamol-io/safe-gpt").get_pretrained()

    inputs = torch.tensor([[tokenizer.bos_token_id]] * args.batch_size, device="cuda")
    generated, tqdm_bar = [], tqdm.trange(args.num_samples, desc="Generation")

    with mp.Pool() as pool:
        while len(generated) < args.num_samples:
            outputs = model.generate(
                inputs,
                do_sample=True,
                temperature=1.0,
                max_length=args.max_length,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            smiles = pool.map(decode_smiles_from_valid_safe, outputs)

            for safe, smiles in zip(outputs, smiles):
                if smiles:
                    generated.append({"safe": safe, "smiles": smiles})
                    tqdm_bar.update()

    generated = pd.DataFrame(generated)
    admet = Evaluator()(generated["smiles"])
    admet = pd.merge(generated, admet, left_on="smiles", right_index=True, how="outer")
    admet.to_csv(args.output, index=False)
