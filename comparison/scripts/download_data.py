"""Download the comparison datasets and pre-cache the HuggingFace models.

Writes (under $RELAG_EXP_BASE, default: this directory):
  data/protein/dbaasp_amp.parquet    id, sequence, log_mic
  data/protein/enzyme_topt.parquet   sequence, label
  data/molecule/{dataset}.parquet    smiles, label

Existing files are left untouched: compute_embeddings.py rewrites the protein
parquets filtered, and the splits/embeddings index into those filtered files.

Run with the deep_learning venv (see setup.sh):
    $HOME/venvs/refnd_exp/deep_learning/bin/python download_data.py
"""
import os
from pathlib import Path

BASE = Path(os.environ.get("RELAG_EXP_BASE", str(Path(__file__).resolve().parent)))
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))
os.environ.setdefault("TDC_DATA_PATH", str(BASE / "tdc_data"))

import numpy as np
import pandas as pd

PROTEIN_DIR = BASE / "data" / "protein"
MOLECULE_DIR = BASE / "data" / "molecule"

# name -> (TDC module, TDC dataset name, label_name or None)
MOLECULE_TASKS = {
    "cyp2c19_veith":   ("ADME", "CYP2C19_Veith", None),
    "caco2_wang":      ("ADME", "Caco2_Wang", None),
    "pgp_broccatelli": ("ADME", "Pgp_Broccatelli", None),
    "ames":            ("Tox",  "AMES", None),
    "lipophilicity":   ("ADME", "Lipophilicity_AstraZeneca", None),
    # SR-ARE is one of Tox21's 12 label columns, not a standalone TDC endpoint.
    "sr_are":          ("Tox",  "Tox21", "SR-ARE"),
}


def download_dbaasp() -> None:
    out = PROTEIN_DIR / "dbaasp_amp.parquet"
    if out.exists():
        print(f"exists, skip: {out}")
        return
    from qmap import DBAASPDataset

    df = DBAASPDataset().tabular(["id", "sequence", "Escherichia coli"])
    df = df.loc[df["Escherichia coli"].notna()].reset_index(drop=True)
    df = df.rename(columns={"Escherichia coli": "mic"})
    df["log_mic"] = np.log10(df["mic"].astype(float))
    df = df[["id", "sequence", "log_mic"]].dropna().reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} protein sequences")


def download_enzyme_topt() -> None:
    out = PROTEIN_DIR / "enzyme_topt.parquet"
    if out.exists():
        print(f"exists, skip: {out}")
        return
    from huggingface_hub import snapshot_download

    # DeepET-sourced Topt regression, the version xTrimoPGLM/Hestia used.
    # Pool the published train+test and re-split downstream (matches the GUE convention).
    d = snapshot_download(repo_id="proteinglm/optimal_temperature", repo_type="dataset")
    frames = [pd.read_parquet(f"{d}/data/{s}-00000-of-00001.parquet") for s in ("train", "test")]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"seq": "sequence"})[["sequence", "label"]].dropna()
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} enzyme sequences (Topt regression)")


def download_molecules() -> None:
    from tdc.single_pred import ADME, Tox

    module_map = {"ADME": ADME, "Tox": Tox}
    for name, (module_name, tdc_name, label_name) in MOLECULE_TASKS.items():
        out = MOLECULE_DIR / f"{name}.parquet"
        if out.exists():
            print(f"exists, skip: {out}")
            continue
        cls = module_map[module_name]
        data = cls(name=tdc_name, label_name=label_name) if label_name else cls(name=tdc_name)
        # TDC returns Drug (SMILES) and Y (label)
        df = data.get_data()[["Drug", "Y"]].dropna().reset_index(drop=True)
        df.columns = ["smiles", "label"]
        df.to_parquet(out, index=False)
        print(f"Saved {len(df)} molecules for {name}")


def cache_models() -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    print("Downloading ESM-C 300M weights...")
    snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12")
    print("Downloading ChemBERTa...")
    AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    print("HuggingFace models cached.")


def main() -> None:
    PROTEIN_DIR.mkdir(parents=True, exist_ok=True)
    MOLECULE_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["TDC_DATA_PATH"]).mkdir(parents=True, exist_ok=True)

    print("=== Downloading DBAASP (protein) ===")
    download_dbaasp()
    print("=== Downloading enzyme optimal-temperature (proteinglm/optimal_temperature) ===")
    download_enzyme_topt()
    print("=== Downloading TDC molecule datasets ===")
    download_molecules()
    print("=== Pre-caching HuggingFace models ===")
    cache_models()


if __name__ == "__main__":
    main()
