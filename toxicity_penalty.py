from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Iterable

DEFAULT_CSV_PATH = "kinase_inhibitor_toxicity_matrix"

BUCKETS: List[str] = [
    "Hepatotoxicity", "Dermatologic", "Pulmonary", "Cardiotoxicity", "Ocular",
    "Neurotoxicity", "Myotoxicity/Rhabdo", "Nephrotoxicity",
    "Hematologic/Myelosuppression", "Embryo-fetal/Reproductive", "Hemorrhagic",
]

BUCKET_WEIGHTS: np.ndarray = np.ones(len(BUCKETS), dtype=float)

def _norm(name: str) -> str:
    return (name or "").strip().lower()

def _ensure_binary_int(col: pd.Series) -> pd.Series:
    x = pd.to_numeric(col, errors="coerce").fillna(0)
    return (x > 0).astype(int)

def load_toxicity_vectors(
    csv_path: str | None = None,
    buckets: Iterable[str] = BUCKETS
) -> Dict[str, np.ndarray]:

    path = csv_path or DEFAULT_CSV_PATH
    df = pd.read_csv(path)

    if "Drug" not in df.columns:
        for alt in ("drug", "DRUG", "Compound", "compound", "Name", "name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Drug"})
                break
        if "Drug" not in df.columns:
            raise KeyError("CSV must include a 'Drug' column.")

   
    for c in buckets:
        if c not in df.columns:
            df[c] = 0
        df[c] = _ensure_binary_int(df[c])


    vecs: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        vecs[_norm(row["Drug"])] = row[list(buckets)].to_numpy(dtype=int)

    return vecs


def ae_overlap_penalty(
    names: List[str],
    tox_map: Dict[str, np.ndarray],
    alpha: float = 1.0,
    beta: float = 1.0,
    bucket_weights: np.ndarray | None = None,
) -> float:
    if not names or tox_map is None or len(names) <= 1:
        return 0.0

    try:
        template = next(iter(tox_map.values()))
    except StopIteration:
        return 0.0

    vecs = []
    for n in names:
        v = tox_map.get(_norm(n))
        if v is None:
            v = np.zeros_like(template)
        vecs.append(v.astype(float))

    M = np.vstack(vecs)  # shape: (k_drugs, k_buckets)

    w = bucket_weights if bucket_weights is not None else BUCKET_WEIGHTS
    if w is not None:
        w = np.asarray(w, dtype=float)
        if w.shape[0] == M.shape[1]:
            M = M * w  

    k = M.shape[0]
    jaccs: List[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = M[i], M[j]
            inter = float(np.minimum(a, b).sum())   # weighted AND
            union = float(np.maximum(a, b).sum())   # weighted OR
            jaccs.append(inter / max(union, 1e-12))
    pair_term = float(np.mean(jaccs)) if jaccs else 0.0

 
    col_hits = (M > 0).sum(axis=0)               
    shared_term = float((col_hits >= 2).sum()) / M.shape[1]

    return alpha * pair_term + beta * shared_term


if __name__ == "__main__":
    tm = load_toxicity_vectors()  # uses DEFAULT_CSV_PATH
    print(f"[toxicity_penalty] Loaded {len(tm)} drugs; vector length = {len(BUCKETS)}.")
    if len(tm) >= 2:
        two = list(tm.keys())[:2]
        pen = ae_overlap_penalty(two, tm, alpha=1.0, beta=1.0)
        print(f"Example combo {two} -> overlap penalty = {pen:.4f}")
    else:
        print("Not enough drugs found in the CSV to demo the penalty.")



def jaccard_overlap(names, tox_map):
    if not names or tox_map is None or len(names) <= 1:
        return 0.0

    try:
        ref = next(iter(tox_map.values()))
    except StopIteration:
        return 0.0

    vecs = []
    for n in names:
        key = (n or "").strip().lower()
        v = tox_map.get(key)
        if v is None:
            v = np.zeros_like(ref)
        vecs.append(v)

    M = np.vstack(vecs)
    k = M.shape[0]
    vals = []
    for i in range(k):
        for j in range(i+1, k):
            a, b = M[i], M[j]
            inter = int(np.logical_and(a, b).sum())
            union = int(np.logical_or(a, b).sum()) or 1
            vals.append(inter / union)
    return float(np.mean(vals)) if vals else 0.0
