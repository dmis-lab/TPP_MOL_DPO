from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
from functools import partial

import pandas as pd
import tqdm
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers


def convert_smiles_to_pdbqt(args, output_dir: str = "./"):
    try:
        with Chem.SDWriter("{}/ligand_{}.sdf".format(output_dir, args[0])) as writer:
            mol = Chem.MolFromSmiles(args[1])
            mol = Chem.AddHs(mol)
            etkdgv3 = rdDistGeom.ETKDGv3()
            rdDistGeom.EmbedMolecule(mol, etkdgv3)

            try:
                rdForceFieldHelpers.UFFOptimizeMolecule(mol)
            except Exception:
                print(f"{args[1]} UFF optimization failed")

            if mol is not None:
                writer.write(mol)

        os.system(
            f"mk_prepare_ligand.py"
            f" -i {output_dir}/ligand_{args[0]}.sdf"
            f" -o {output_dir}/ligand_{args[0]}.pdbqt"
        )
    except Exception:
        print("ligand_prep failed: {}".format(args[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="smiles.csv")
    parser.add_argument("--output-dir", default="ligands")
    parser.add_argument("--use-index", action="store_true", default=False)
    parser.add_argument("--subset", default="0:1")
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    subset_idx, subset_cnt = map(int, args.subset.split(":"))

    dataset = pd.read_csv(args.dataset, index_col=0 if args.use_index else None)
    dataset = dataset.iloc[subset_idx::subset_cnt]
    with mp.Pool() as pool:
        convert_fn = partial(convert_smiles_to_pdbqt, output_dir=args.output_dir)
        it = pool.imap_unordered(convert_fn, zip(dataset.index, dataset["smiles"]))
        list(tqdm.tqdm(it, total=len(dataset)))
