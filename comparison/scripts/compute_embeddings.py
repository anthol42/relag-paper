"""Compute the frozen foundation-model embeddings used for training.

Reads data/ (see download_data.py) and writes (under $RELAG_EXP_BASE, default:
this directory):
  embeddings/protein_esmc.pt          ESM-C 300M, dbaasp_amp
  embeddings/protein_enzyme_topt.pt   ESM-C 300M, enzyme_topt
  embeddings/mol_{name}.pt            ChemBERTa, molecule datasets
Each output is skipped if it already exists. The protein parquets are rewritten
filtered, so the splits use the same indices as the embeddings.

Run with the deep_learning venv (see setup.sh); uses a GPU if available:
    $HOME/venvs/refnd_exp/deep_learning/bin/python compute_embeddings.py
"""
import os
from pathlib import Path

BASE = Path(os.environ.get("RELAG_EXP_BASE", str(Path(__file__).resolve().parent)))
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"       # all models must be pre-cached by download_data.py
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DATA_DIR = BASE / "data"
EMB_DIR = BASE / "embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# molecule dataset -> embedding file stem
MOLECULE_EMBEDDINGS = {
    "cyp2c19_veith":   "mol_cyp2c19",
    "caco2_wang":      "mol_caco2",
    "pgp_broccatelli": "mol_pgp",
    "ames":            "mol_ames",
    "lipophilicity":   "mol_lipophilicity",
    "sr_are":          "mol_sr_are",
}


def _load_esmc():
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained("esmc_300m")
    model.eval()
    return model.to(DEVICE)


def _filter_proteins(path: Path, label: str, valid_aa: set[str]) -> list[str]:
    """Drop rows with missing/non-finite labels or non-standard residues, and
    rewrite the parquet so splits use the same indices as the embeddings."""
    df = pd.read_parquet(path)
    mask = (
        df["sequence"].notna()
        & df[label].notna()
        & df[label].apply(np.isfinite)
        & df["sequence"].apply(lambda s: isinstance(s, str) and len(s) > 0 and set(s.upper()).issubset(valid_aa))
    )
    df = df[mask].reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(f"Sequences after filtering: {len(df)}")
    return df["sequence"].str.upper().tolist()


def embed_dbaasp() -> None:
    out = EMB_DIR / "protein_esmc.pt"
    if out.exists():
        print("Already computed, skipping.")
        return
    from esm.sdk.api import ESMProtein, LogitsConfig

    sequences = _filter_proteins(DATA_DIR / "protein" / "dbaasp_amp.parquet", "log_mic",
                                 set("ACDEFGHIKLMNPQRSTVWY"))
    model = _load_esmc()
    cfg = LogitsConfig(return_embeddings=True)
    batch_size = 32
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-C protein", unit="batch"):
            tensors = [model.encode(ESMProtein(sequence=s)) for s in sequences[i:i + batch_size]]
            # embeddings shape: [seq_len, dim] -- mean pool over sequence length
            embeds = torch.stack([model.logits(t, cfg).embeddings.squeeze(0).mean(dim=0) for t in tensors])
            all_embeds.append(embeds.cpu().float())
    embeddings = torch.cat(all_embeds, dim=0)
    torch.save(embeddings, out)
    print(f"Saved protein embeddings: {embeddings.shape}")


def embed_enzyme_topt() -> None:
    out = EMB_DIR / "protein_enzyme_topt.pt"
    if out.exists():
        print("Already computed, skipping.")
        return
    from esm.sdk.api import ESMProtein, LogitsConfig

    # Keep the 20 standard AAs plus X; embed one sequence at a time.
    sequences = _filter_proteins(DATA_DIR / "protein" / "enzyme_topt.parquet", "label",
                                 set("ACDEFGHIKLMNPQRSTVWYX"))
    model = _load_esmc()
    cfg = LogitsConfig(return_embeddings=True)
    all_embeds = []
    with torch.no_grad():
        for s in tqdm(sequences, desc="ESM-C enzyme_topt", unit="seq"):
            t = model.encode(ESMProtein(sequence=s))
            all_embeds.append(model.logits(t, cfg).embeddings.squeeze(0).mean(dim=0).cpu().float())
    embeddings = torch.stack(all_embeds, dim=0)
    torch.save(embeddings, out)
    print(f"Saved enzyme_topt embeddings: {embeddings.shape}")


def embed_molecules() -> None:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.eval()
    model = model.to(DEVICE)

    batch_size = 64
    for name, stem in MOLECULE_EMBEDDINGS.items():
        out = EMB_DIR / f"{stem}.pt"
        if out.exists():
            print(f"Skipping {name}, already computed.")
            continue
        path = DATA_DIR / "molecule" / f"{name}.parquet"
        if not path.exists():
            print(f"ERROR: No data found for {name}")
            continue
        df = pd.read_parquet(path)
        df = df[df["smiles"].notna() & df["label"].notna()].reset_index(drop=True)
        df.to_parquet(path, index=False)

        smiles = df["smiles"].tolist()
        all_embeds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles), batch_size), desc=f"ChemBERTa {name}", unit="batch"):
                enc = tok(smiles[i:i + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=512)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                all_embeds.append(model(**enc).last_hidden_state[:, 0, :].cpu().float())
        embeddings = torch.cat(all_embeds, dim=0)
        torch.save(embeddings, out)
        print(f"Saved {name}: {embeddings.shape}")


def main() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ESM-C 300M protein embeddings ===")
    embed_dbaasp()
    print("=== ESM-C 300M enzyme_topt embeddings ===")
    embed_enzyme_topt()
    print("=== ChemBERTa molecule embeddings ===")
    embed_molecules()


if __name__ == "__main__":
    main()
