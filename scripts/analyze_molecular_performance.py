#!/usr/bin/env python
"""Relate molecule-level predictions to properties, structure, and functional groups.

The input is produced by ``scripts/evaluate.py --analysis``. Outputs include
an enriched molecule table, numerical-property associations, binned accuracy,
functional-group associations and recovery metrics, summary JSON, and compact
diagnostic plots. MMP output from ``ir_raman_freq/evaluation.py
--save-predictions`` is also supported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import fisher_exact, norm, pointbiserialr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import project_path  # noqa: E402


PROPERTY_COLUMNS = (
    "dipole_norm",
    "vip",
    "vea",
    "polar_aniso",
    "polar_iso",
    "homolumo",
    "deen",
)

DESCRIPTOR_COLUMNS = (
    "exact_molecular_weight",
    "total_atom_count",
    "hetero_atom_count",
    "formal_charge",
    "rotatable_bond_count",
    "ring_count",
    "aromatic_ring_count",
    "h_bond_donor_count",
    "h_bond_acceptor_count",
    "tpsa",
    "logp",
    "fraction_csp3",
)

# Functional groups are deliberately non-exclusive: one molecule can contain
# multiple groups. Counts are substructure-match counts; the *_present columns
# are used for group-wise accuracy tests.
FUNCTIONAL_GROUP_SMARTS = {
    "hydroxyl": "[OX2H]",
    "alcohol": "[CX4][OX2H]",
    "phenol": "[c][OX2H]",
    "ether": "[OD2]([#6])[#6]",
    "carbonyl": "[CX3]=[OX1]",
    "aldehyde": "[CX3H1](=O)[#6,H]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "carbamate": "[NX3][CX3](=[OX1])[OX2][#6]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "amine": "[NX3;H2,H1,H0;!$(N-C=O)]",
    "imine": "[CX3]=[NX2]",
    "azo": "[NX2]=[NX2]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "thiol": "[#16X2H]",
    "sulfide": "[#16X2]([#6])[#6]",
    "sulfoxide": "[#16X3](=[OX1])",
    "sulfone": "[#16X4](=[OX1])(=[OX1])",
    "sulfonamide": "[#16X4](=[OX1])(=[OX1])[NX3]",
    "phosphate": "[PX4](=[OX1])([OX2])[OX2]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "halogen": "[F,Cl,Br,I]",
    "aromatic": "[a]",
    "aromatic_heterocycle": "[nH,n,o,s]",
}
FUNCTIONAL_GROUP_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Molecule-level evaluation CSV")
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: <input stem>_analysis next to input)",
    )
    parser.add_argument(
        "--target",
        choices=("top1_correct", "top3_correct", "top5_correct", "top1_valid"),
        default="top1_correct",
    )
    parser.add_argument("--bins", type=int, default=5, help="Quantile bins for accuracy curves")
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=20,
        help="Minimum present and absent samples for functional-group tests",
    )
    return parser.parse_args()


def rdkit_features(smiles: object) -> dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        result = {name: math.nan for name in DESCRIPTOR_COLUMNS}
        for group in FUNCTIONAL_GROUP_PATTERNS:
            result[f"fg_{group}_count"] = math.nan
            result[f"fg_{group}_present"] = math.nan
        return result

    result = {
        "exact_molecular_weight": Descriptors.ExactMolWt(mol),
        "total_atom_count": float(Chem.AddHs(mol).GetNumAtoms()),
        "hetero_atom_count": float(Lipinski.NumHeteroatoms(mol)),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "rotatable_bond_count": float(Lipinski.NumRotatableBonds(mol)),
        "ring_count": float(Lipinski.RingCount(mol)),
        "aromatic_ring_count": float(Lipinski.NumAromaticRings(mol)),
        "h_bond_donor_count": float(Lipinski.NumHDonors(mol)),
        "h_bond_acceptor_count": float(Lipinski.NumHAcceptors(mol)),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Crippen.MolLogP(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }
    for group, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
        if pattern is None:
            raise RuntimeError(f"Invalid built-in SMARTS for {group}")
        count = len(mol.GetSubstructMatches(pattern))
        result[f"fg_{group}_count"] = float(count)
        result[f"fg_{group}_present"] = int(count > 0)
    return result


def candidate_smiles(row: pd.Series) -> list[str]:
    """Return canonical/raw beam candidates, falling back to the Top-1 column."""
    for column in (
        "candidate_canonical_smiles_json",
        "candidate_smiles_json",
    ):
        value = row.get(column)
        if pd.isna(value):
            continue
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list) and parsed:
            # Preserve empty/invalid candidates so Top-k rank boundaries do
            # not shift when, for example, rank 1 is invalid but rank 4 is not.
            return [str(item) if item else "" for item in parsed]
    for column in ("predicted_canonical_smiles", "predicted_smiles"):
        value = row.get(column)
        if pd.notna(value) and str(value):
            return [str(value)]
    return []


def functional_group_presence(smiles: object) -> dict[str, int]:
    molecule = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if molecule is None:
        return {group: 0 for group in FUNCTIONAL_GROUP_PATTERNS}
    return {
        group: int(molecule.HasSubstructMatch(pattern))
        for group, pattern in FUNCTIONAL_GROUP_PATTERNS.items()
        if pattern is not None
    }


def candidate_functional_group_hits(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark whether any Top-3/5 candidate contains each target group."""
    rows = []
    for _, row in frame.iterrows():
        candidates = candidate_smiles(row)[:5]
        candidate_presence = [
            functional_group_presence(smiles) for smiles in candidates
        ]
        result = {}
        for group in FUNCTIONAL_GROUP_PATTERNS:
            result[f"fg_{group}_hit_at_3"] = int(
                any(values[group] for values in candidate_presence[:3])
            )
            result[f"fg_{group}_hit_at_5"] = int(
                any(values[group] for values in candidate_presence[:5])
            )
        rows.append(result)
    return pd.DataFrame(rows, index=frame.index)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def functional_group_recovery(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure whether predicted structures preserve each target functional group."""
    rows = []
    for group in FUNCTIONAL_GROUP_PATTERNS:
        target = pd.to_numeric(
            frame[f"fg_{group}_present"], errors="coerce"
        ).fillna(0).astype(bool)
        predicted = pd.to_numeric(
            frame[f"predicted_fg_{group}_present"], errors="coerce"
        ).fillna(0).astype(bool)
        hit_at_3 = pd.to_numeric(
            frame[f"fg_{group}_hit_at_3"], errors="coerce"
        ).fillna(0).astype(bool)
        hit_at_5 = pd.to_numeric(
            frame[f"fg_{group}_hit_at_5"], errors="coerce"
        ).fillna(0).astype(bool)
        true_positive = int((target & predicted).sum())
        false_negative = int((target & ~predicted).sum())
        false_positive = int((~target & predicted).sum())
        true_negative = int((~target & ~predicted).sum())
        precision = safe_ratio(true_positive, true_positive + false_positive)
        recall = safe_ratio(true_positive, true_positive + false_negative)
        precision_ci = wilson_interval(
            true_positive, true_positive + false_positive
        )
        recall_ci = wilson_interval(
            true_positive, true_positive + false_negative
        )
        top3_correct = int((target & hit_at_3).sum())
        top5_correct = int((target & hit_at_5).sum())
        top3_ci = wilson_interval(top3_correct, int(target.sum()))
        top5_ci = wilson_interval(top5_correct, int(target.sum()))
        target_count = pd.to_numeric(
            frame[f"fg_{group}_count"], errors="coerce"
        ).fillna(0)
        predicted_count = pd.to_numeric(
            frame[f"predicted_fg_{group}_count"], errors="coerce"
        ).fillna(0)
        rows.append(
            {
                "functional_group": group,
                "samples": len(frame),
                "target_present": int(target.sum()),
                "predicted_top1_present": int(predicted.sum()),
                "true_positive": true_positive,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "true_negative": true_negative,
                "top1_precision": precision,
                "top1_precision_ci_low": precision_ci[0],
                "top1_precision_ci_high": precision_ci[1],
                "top1_recall": recall,
                "top1_recall_ci_low": recall_ci[0],
                "top1_recall_ci_high": recall_ci[1],
                "top1_f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else math.nan
                ),
                "top1_specificity": safe_ratio(
                    true_negative, true_negative + false_positive
                ),
                "top1_presence_accuracy": safe_ratio(
                    true_positive + true_negative, len(frame)
                ),
                "target_group_recall_at_3": safe_ratio(
                    top3_correct, int(target.sum())
                ),
                "target_group_recall_at_3_ci_low": top3_ci[0],
                "target_group_recall_at_3_ci_high": top3_ci[1],
                "target_group_recall_at_5": safe_ratio(
                    top5_correct, int(target.sum())
                ),
                "target_group_recall_at_5_ci_low": top5_ci[0],
                "target_group_recall_at_5_ci_high": top5_ci[1],
                "top1_group_count_mae": float(
                    (target_count - predicted_count).abs().mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top1_recall", "target_group_recall_at_5"], ascending=False
    )


def functional_group_recovery_by_size(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for heavy_size, subset in frame.groupby("heavy_size", observed=True):
        result = functional_group_recovery(subset)
        result.insert(0, "heavy_size", heavy_size)
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values, preserving missing values."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return result
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(
        (ordered.to_numpy() * len(ordered) / ranks)[::-1]
    )[::-1]
    result.loc[ordered.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def adjusted_logistic_coefficient(
    frame: pd.DataFrame, feature: str, target: str
) -> float:
    """Standardized feature coefficient, adjusted for heavy atom count."""
    columns = [feature]
    if feature != "heavy_size":
        columns.append("heavy_size")
    subset = frame[[target, *columns]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 10 or subset[target].nunique() < 2:
        return math.nan
    x = subset[columns].to_numpy(dtype=float)
    if np.any(np.nanstd(x, axis=0) == 0):
        return math.nan
    x = StandardScaler().fit_transform(x)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(x, subset[target].astype(int))
    return float(model.coef_[0, 0])


def numerical_associations(
    frame: pd.DataFrame, target: str, features: list[str]
) -> pd.DataFrame:
    rows = []
    for feature in features:
        subset = frame[[target, feature]].apply(pd.to_numeric, errors="coerce").dropna()
        pb_r = pb_p = sp_r = sp_p = math.nan
        if (
            len(subset) >= 3
            and subset[target].nunique() == 2
            and subset[feature].nunique() > 1
        ):
            pb = pointbiserialr(subset[target], subset[feature])
            sp = spearmanr(subset[target], subset[feature])
            pb_r, pb_p = float(pb.statistic), float(pb.pvalue)
            sp_r, sp_p = float(sp.statistic), float(sp.pvalue)
        rows.append(
            {
                "feature": feature,
                "n": len(subset),
                "mean": subset[feature].mean(),
                "std": subset[feature].std(),
                "point_biserial_r": pb_r,
                "point_biserial_p": pb_p,
                "spearman_r": sp_r,
                "spearman_p": sp_p,
                "standardized_logistic_coefficient_adjusted_for_heavy_size": (
                    adjusted_logistic_coefficient(frame, feature, target)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["point_biserial_q"] = benjamini_hochberg(result["point_biserial_p"])
    result["spearman_q"] = benjamini_hochberg(result["spearman_p"])
    return result


def binned_accuracy(
    frame: pd.DataFrame, target: str, features: list[str], bins: int
) -> pd.DataFrame:
    rows = []
    for feature in features:
        subset = frame[[target, feature]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) < bins or subset[feature].nunique() < 2:
            continue
        try:
            labels = pd.qcut(subset[feature], q=bins, duplicates="drop")
        except ValueError:
            continue
        grouped = subset.assign(bin=labels).groupby("bin", observed=True)
        for interval, values in grouped:
            rows.append(
                {
                    "feature": feature,
                    "bin": str(interval),
                    "bin_midpoint": values[feature].mean(),
                    "n": len(values),
                    "accuracy": values[target].mean(),
                }
            )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = float(norm.ppf(0.975))
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - margin, center + margin


def functional_group_associations(
    frame: pd.DataFrame, target: str, min_group_size: int
) -> pd.DataFrame:
    rows = []
    present_columns = [
        column for column in frame if column.startswith("fg_") and column.endswith("_present")
    ]
    for column in present_columns:
        subset = frame[[target, column, "heavy_size"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        present = subset[subset[column] == 1]
        absent = subset[subset[column] == 0]
        present_correct = int(present[target].sum())
        absent_correct = int(absent[target].sum())
        present_ci = wilson_interval(present_correct, len(present))
        absent_ci = wilson_interval(absent_correct, len(absent))
        fisher_odds = fisher_p = math.nan
        if len(present) >= min_group_size and len(absent) >= min_group_size:
            table = [
                [present_correct, len(present) - present_correct],
                [absent_correct, len(absent) - absent_correct],
            ]
            fisher = fisher_exact(table)
            fisher_odds, fisher_p = float(fisher.statistic), float(fisher.pvalue)
        rows.append(
            {
                "functional_group": column.removeprefix("fg_").removesuffix("_present"),
                "n_present": len(present),
                "n_absent": len(absent),
                "accuracy_present": present[target].mean(),
                "accuracy_present_ci_low": present_ci[0],
                "accuracy_present_ci_high": present_ci[1],
                "accuracy_absent": absent[target].mean(),
                "accuracy_absent_ci_low": absent_ci[0],
                "accuracy_absent_ci_high": absent_ci[1],
                "accuracy_difference": present[target].mean() - absent[target].mean(),
                "fisher_odds_ratio": fisher_odds,
                "fisher_p": fisher_p,
                "standardized_logistic_coefficient_adjusted_for_heavy_size": (
                    adjusted_logistic_coefficient(subset, column, target)
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["fisher_q"] = benjamini_hochberg(result["fisher_p"])
    return result.sort_values("accuracy_difference")


def plot_numeric(result: pd.DataFrame, path: Path) -> None:
    plotted = result.dropna(subset=["point_biserial_r"]).sort_values("point_biserial_r")
    if plotted.empty:
        return
    height = max(4.0, 0.32 * len(plotted))
    fig, axis = plt.subplots(figsize=(8, height))
    axis.barh(plotted["feature"], plotted["point_biserial_r"], color="#4C78A8")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Point-biserial correlation with accuracy")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_functional_groups(result: pd.DataFrame, path: Path) -> None:
    plotted = result.dropna(subset=["fisher_p"])
    if plotted.empty:
        return
    height = max(4.0, 0.38 * len(plotted))
    fig, axis = plt.subplots(figsize=(8, height))
    axis.barh(
        plotted["functional_group"], plotted["accuracy_difference"], color="#F58518"
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Accuracy difference (group present - absent)")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_functional_group_recovery(result: pd.DataFrame, path: Path) -> None:
    plotted = result.dropna(subset=["top1_recall"]).sort_values("top1_recall")
    if plotted.empty:
        return
    positions = np.arange(len(plotted))
    height = max(4.0, 0.4 * len(plotted))
    fig, axis = plt.subplots(figsize=(9, height))
    axis.barh(
        positions - 0.22,
        plotted["top1_recall"],
        height=0.22,
        label="Top-1",
    )
    axis.barh(
        positions,
        plotted["target_group_recall_at_3"],
        height=0.22,
        label="Top-3",
    )
    axis.barh(
        positions + 0.22,
        plotted["target_group_recall_at_5"],
        height=0.22,
        label="Top-5",
    )
    axis.set_yticks(positions, plotted["functional_group"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Target functional-group recall")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.min_group_size < 1:
        raise ValueError("--min-group-size must be at least 1")
    RDLogger.DisableLog("rdApp.error")

    input_path = project_path(args.input)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else input_path.parent / f"{input_path.stem}_analysis"
    )
    frame = pd.read_csv(input_path)
    # These duplicate exact_molecular_weight and heavy_size, respectively.
    # Drop them even when a previously enriched analysis CSV is supplied.
    frame = frame.drop(
        columns=["molecular_weight", "rdkit_heavy_atom_count"], errors="ignore"
    )
    required = {args.target, "canonical_smiles", "heavy_size", *PROPERTY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(sorted(missing))}")
    target_values = pd.to_numeric(frame[args.target], errors="coerce")
    if not target_values.dropna().isin([0, 1]).all():
        raise ValueError(f"{args.target} must contain only 0/1 values")
    frame[args.target] = target_values

    structure_rows = [rdkit_features(value) for value in frame["canonical_smiles"]]
    structure = pd.DataFrame(structure_rows, index=frame.index)
    overlapping = set(frame).intersection(structure)
    if overlapping:
        frame = frame.drop(columns=sorted(overlapping))
    prediction_column = next(
        (
            column
            for column in ("predicted_canonical_smiles", "predicted_smiles")
            if column in frame
        ),
        None,
    )
    if prediction_column is None:
        raise ValueError(
            "Input is missing predicted_canonical_smiles or predicted_smiles"
        )
    predicted_structure = pd.DataFrame(
        [rdkit_features(value) for value in frame[prediction_column]],
        index=frame.index,
    ).add_prefix("predicted_")
    candidate_hits = candidate_functional_group_hits(frame)
    enriched = pd.concat([frame, structure, predicted_structure, candidate_hits], axis=1)
    numeric_features = ["heavy_size", *PROPERTY_COLUMNS, *DESCRIPTOR_COLUMNS]

    numeric = numerical_associations(enriched, args.target, numeric_features)
    binned = binned_accuracy(enriched, args.target, numeric_features, args.bins)
    functional = functional_group_associations(
        enriched, args.target, args.min_group_size
    )
    recovery = functional_group_recovery(enriched)
    recovery_by_size = functional_group_recovery_by_size(enriched)

    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "molecule_analysis.csv", index=False)
    numeric.to_csv(output_dir / "numeric_associations.csv", index=False)
    binned.to_csv(output_dir / "numeric_binned_accuracy.csv", index=False)
    functional.to_csv(output_dir / "functional_group_associations.csv", index=False)
    recovery.to_csv(output_dir / "functional_group_recovery.csv", index=False)
    recovery_by_size.to_csv(
        output_dir / "functional_group_recovery_by_heavy_size.csv", index=False
    )
    plot_numeric(numeric, output_dir / "numeric_correlations.png")
    plot_functional_groups(functional, output_dir / "functional_group_accuracy.png")
    plot_functional_group_recovery(
        recovery, output_dir / "functional_group_recovery.png"
    )

    valid_target = enriched[args.target].dropna()
    summary = {
        "input": str(input_path),
        "target": args.target,
        "samples": int(len(enriched)),
        "samples_with_target": int(len(valid_target)),
        "overall_accuracy": float(valid_target.mean()),
        "invalid_canonical_smiles": int(
            structure["exact_molecular_weight"].isna().sum()
        ),
        "min_group_size": args.min_group_size,
        "outputs": {
            "molecules": str(output_dir / "molecule_analysis.csv"),
            "numeric": str(output_dir / "numeric_associations.csv"),
            "binned": str(output_dir / "numeric_binned_accuracy.csv"),
            "functional_groups": str(output_dir / "functional_group_associations.csv"),
            "functional_group_recovery": str(
                output_dir / "functional_group_recovery.csv"
            ),
            "functional_group_recovery_by_heavy_size": str(
                output_dir / "functional_group_recovery_by_heavy_size.csv"
            ),
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
