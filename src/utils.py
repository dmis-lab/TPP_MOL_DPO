from __future__ import annotations

from collections import defaultdict

import chemfunc
import numpy as np
import pandas as pd
from admet_ai import ADMETModel
from rdkit import Chem, rdBase
from rdkit.Contrib.SA_Score import sascorer  # type: ignore
from transformers import FlaxGPT2LMHeadModel, FlaxPreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_flax_gpt2 import FlaxGPT2LMHeadModule

rdBase.DisableLog("rdApp.*")


class FlaxGPT2LMHeadModelWrapper(FlaxGPT2LMHeadModel):
    def __init__(self, config: GPT2Config, module: FlaxGPT2LMHeadModule):
        FlaxPreTrainedModel.__init__(self, config, module, _do_init=False)


class AverageMeter:
    def __init__(self, use_latest: list[str] = []):
        self.buffer = defaultdict(list)
        self.use_latest = use_latest

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def summary(self, prefix: str = "") -> dict[str, float]:
        buffer = {k: np.array(v) for k, v in self.buffer.items()}
        self.buffer.clear()

        return {
            f"{prefix}{k}": v[-1] if k in self.use_latest else np.mean(v)
            for k, v in buffer.items()
        }


class Evaluator:
    def __init__(self):
        self.admet_ai = ADMETModel()

    def __call__(self, smiles_list: list[str]) -> pd.DataFrame:
        # Calculate additional scores from SMILES.
        mols = [Chem.MolFromSmiles(x) for x in smiles_list]
        sa_scores = [sascorer.calculateScore(m) for m in mols]
        max_ring = [max(map(len, m.GetRingInfo().AtomRings() or [[]])) for m in mols]

        # Calculate interval diversity from morgan fingerprints.
        morgan_fp = np.stack([chemfunc.compute_morgan_fingerprint(m) for m in mols])
        dot, norm = morgan_fp @ morgan_fp.T, morgan_fp.sum(-1, keepdims=True)
        tanimoto = dot / (norm + norm.T - dot)
        intdiv = 1 - (tanimoto.sum(-1) - 1) / (tanimoto.shape[1] - 1)

        # Use ADMET-AI model to predict ADMET from SMILES.
        admet = self.admet_ai.predict(smiles_list)
        admet["SAScore"] = sa_scores
        admet["CycleScore"] = [max(x - 6, 0) for x in max_ring]
        admet["plogP"] = admet["logP"] - admet["SAScore"] - admet["CycleScore"]
        admet["IntDiv"] = intdiv
        return admet
