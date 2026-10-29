#!/usr/bin/env python3
"""Validation-led multi-head LSTM experiment for the year-only UOP panel.

Model and feature candidates are chosen using validation windows only. The
selected recipe is then refit with fresh seeds and evaluated on two later
fiscal-year windows. Exact filing-time validity is not claimed because the
source does not provide filing availability dates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fme.features import RobustFeatureTransform, build_sequences
from fme.model import MultiHeadTemporalClassifier
from fme.baselines import TemporalOnly


FOLDS = {
    "early": {"train_end": 2012, "val_start": 2013, "val_end": 2014, "test_start": 2015, "test_end": 2016},
    "late": {"train_end": 2016, "val_start": 2017, "val_end": 2018, "test_start": 2019, "test_end": 2021},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def score_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan"),
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
    }


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_fold(data_path: Path, ratio_path: Path, fold_name: str, history: int):
    spec = FOLDS[fold_name]
    data = pd.read_csv(data_path)
    ratios = pd.read_csv(ratio_path)
    data["company_id"] = data.company_id.astype(str)
    ratios["company_id"] = ratios.company_id.astype(str)
    data = data.sort_values(["company_id", "fiscal_year"], kind="stable").reset_index(drop=True)
    ratios = ratios.sort_values(["company_id", "fiscal_year"], kind="stable").reset_index(drop=True)
    if not data[["company_id", "fiscal_year", "label"]].equals(ratios[["company_id", "fiscal_year", "label"]]):
        raise ValueError("Raw and ratio panels do not have identical company/year/label ordering.")

    split = data.fiscal_year.astype(int)
    train_mask = split.le(spec["train_end"]).to_numpy()
    val_mask = split.between(spec["val_start"], spec["val_end"]).to_numpy()
    test_mask = split.between(spec["test_start"], spec["test_end"]).to_numpy()
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(f"Empty split in fold {fold_name}.")

    views = {}
    for view_name, frame in (("raw", data), ("ratios", ratios)):
        excluded = {"company_id", "fiscal_year", "label", "available_date", "event_date", "filing_date"}
        columns = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
        missing = frame[columns].isna().to_numpy(dtype=np.float32)
        transform = RobustFeatureTransform(columns).fit(frame.loc[train_mask])
        values = transform.transform(frame)
        seq, _, _ = build_sequences(frame, values, history_length=history)
        mask_seq, _, _ = build_sequences(frame, missing, history_length=history)
        stacked = np.stack([seq, mask_seq], axis=-1)
        views[view_name] = {"x": stacked, "values": seq, "columns": columns}

    return {
        "frame": data,
        "y": data.label.to_numpy(dtype=np.float32),
        "masks": {"train": train_mask, "validation": val_mask, "test": test_mask},
        "views": views,
        "split": {k: int(v) for k, v in spec.items()},
    }


def model_spec(config: dict, n_features: int) -> MultiHeadTemporalClassifier:
    if config["name"] == "single_lstm":
        return TemporalOnly(n_features, hidden_dim=config["hidden_dim"], dropout=config["dropout"])
    return MultiHeadTemporalClassifier(
        n_features=n_features,
        input_channels=2 if config["missingness"] else 1,
        hidden_dim=config["hidden_dim"],
        pooling=config["pooling"],
        dropout=config["dropout"],
        input_skip=config.get("input_skip", False),
    )


def tensors_for(view: dict, config: dict, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    if config["name"] == "single_lstm":
        arr = view["values"][indices]
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    if config["missingness"]:
        arr = view["x"][indices]
    else:
        arr = view["values"][indices][..., None]
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def run_fit(
    prepared: dict,
    config: dict,
    *,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    negative_ratio: int,
    evaluate_test: bool,
) -> dict:
    seed_everything(seed)
    view = prepared["views"][config["view"]]
    y_all = prepared["y"]
    train_ids = np.flatnonzero(prepared["masks"]["train"])
    val_ids = np.flatnonzero(prepared["masks"]["validation"])
    pos_ids = train_ids[y_all[train_ids] == 1]
    neg_ids = train_ids[y_all[train_ids] == 0]
    if len(pos_ids) < 2:
        raise ValueError("Training fold has too few positive examples.")

    model = model_spec(config, len(view["columns"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_type = config.get("loss_type", "bce")
    val_x = tensors_for(view, config, val_ids, device)
    val_y = torch.as_tensor(y_all[val_ids], dtype=torch.float32, device=device)
    val_np = y_all[val_ids].astype(int)
    rng = np.random.default_rng(seed)
    best_ap = -1.0
    best_roc = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    trace = []

    for epoch in range(1, max_epochs + 1):
        take = min(len(neg_ids), negative_ratio * len(pos_ids))
        sampled_neg = rng.choice(neg_ids, size=take, replace=False)
        sampled = np.concatenate([pos_ids, sampled_neg])
        rng.shuffle(sampled)
        train_x = tensors_for(view, config, sampled, device)
        train_y = torch.as_tensor(y_all[sampled], dtype=torch.float32, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        if loss_type == "pairwise":
            positive_logits = logits[train_y > 0.5]
            negative_logits = logits[train_y <= 0.5]
            loss = F.softplus(-(positive_logits[:, None] - negative_logits[None, :])).mean()
        elif loss_type == "focal":
            element_loss = F.binary_cross_entropy_with_logits(logits, train_y, reduction="none")
            correct_probability = torch.exp(-element_loss)
            class_weight = torch.where(train_y > 0.5, 0.8, 0.2)
            loss = (class_weight * (1.0 - correct_probability).square() * element_loss).mean()
        elif loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(logits, train_y)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(val_x)).cpu().numpy()
        val_metric = score_metrics(val_np, val_scores)
        trace.append({"epoch": epoch, "loss": float(loss.item()), "validation": val_metric})
        ap, roc = val_metric["average_precision"], val_metric["roc_auc"]
        print(
            f"{prepared['fold']} {config['name']} loss={loss_type} seed={seed} epoch={epoch:03d} "
            f"loss={loss.item():.4f} val_AP={ap:.4f} val_ROC={roc:.4f}",
            flush=True,
        )
        if (ap, roc) > (best_ap, best_roc):
            best_ap, best_roc = ap, roc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch, stale = epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint selected on validation data.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_val_scores = torch.sigmoid(model(val_x)).cpu().numpy()
    result = {
        "fold": prepared["fold"], "config": config, "seed": seed,
        "best_epoch": best_epoch,
        "validation": {"average_precision": best_ap, "roc_auc": best_roc, "n": len(val_ids), "positives": int(y_all[val_ids].sum())},
        "validation_predictions": {"row_ids": val_ids.tolist(), "labels": val_np.tolist(), "scores": best_val_scores.tolist()},
        "trace": trace,
    }
    if evaluate_test:
        test_ids = np.flatnonzero(prepared["masks"]["test"])
        test_x = tensors_for(view, config, test_ids, device)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(test_x)).cpu().numpy()
        result["test"] = score_metrics(y_all[test_ids].astype(int), scores)
        result["test_predictions"] = {
            "row_ids": test_ids.tolist(), "labels": y_all[test_ids].astype(int).tolist(), "scores": scores.tolist(),
        }
    return result


def candidate_configs() -> list[dict]:
    configs = []
    for view in ("raw", "ratios"):
        for history in (4, 5):
            configs.append({"name": "single_lstm", "view": view, "history": history, "pooling": "none", "missingness": False, "hidden_dim": 32, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.001})
            configs.append({"name": "multihead_concat", "view": view, "history": history, "pooling": "concat", "missingness": False, "hidden_dim": 8, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.001})
            configs.append({"name": "multihead_attention", "view": view, "history": history, "pooling": "attention", "missingness": False, "hidden_dim": 16, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.001})
            configs.append({"name": "multihead_missingness", "view": view, "history": history, "pooling": "concat", "missingness": True, "hidden_dim": 8, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.001})
            configs.append({"name": "multihead_skip", "view": view, "history": history, "pooling": "concat", "missingness": True, "input_skip": True, "hidden_dim": 8, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.001})
    for loss_type in ("pairwise", "focal"):
        configs.append({"name": "multihead_attention", "view": "raw", "history": 5, "pooling": "attention", "missingness": False, "hidden_dim": 16, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.001, "loss_type": loss_type})
        configs.append({"name": "multihead_missingness", "view": "raw", "history": 4, "pooling": "concat", "missingness": True, "hidden_dim": 8, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 0.001, "loss_type": loss_type})
    return configs


def config_key(config: dict) -> str:
    base = f"{config['name']}|{config['view']}|{config['history']}"
    loss_type = config.get("loss_type", "bce")
    return base if loss_type == "bce" else f"{base}|{loss_type}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="data/processed/uop_numeric_panel.csv")
    parser.add_argument("--ratio-file", default="data/processed/uop_numeric_ratio_panel.csv")
    parser.add_argument("--output-dir", default="results/multihead-lstm-v1")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--screen-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--final-seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105])
    parser.add_argument("--folds", nargs="+", choices=list(FOLDS), default=list(FOLDS), help="Fold names to screen/refit; default uses both chronological folds.")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--top-configs", type=int, default=2)
    parser.add_argument("--config-keys", nargs="*", default=None, help="Optional exact candidate keys, such as multihead_missingness|raw|4.")
    parser.add_argument("--screening-file", default=None, help="Reuse a completed validation-only experiment.json and skip screening.")
    parser.add_argument("--screen-only", action="store_true", help="Do not run the frozen final test evaluation.")
    parser.add_argument("--validation-only", action="store_true", help="Refit fold-selected configurations and save validation scores without test evaluation.")
    args = parser.parse_args()

    data_path, ratio_path = ROOT / args.data_file, ROOT / args.ratio_file
    if not data_path.exists() or not ratio_path.exists():
        raise FileNotFoundError(f"Required panels not found: {data_path}, {ratio_path}")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    active_folds = args.folds
    print(f"Device: {device}; data={data_path}; ratios={ratio_path}", flush=True)
    configs = candidate_configs()
    if args.config_keys is not None and args.screening_file is None:
        selected_keys = set(args.config_keys)
        configs = [config for config in configs if config_key(config) in selected_keys]
        missing_keys = selected_keys - {config_key(config) for config in configs}
        if missing_keys:
            raise ValueError(f"Unknown candidate keys: {sorted(missing_keys)}")
    if not configs:
        raise ValueError("No candidate configurations selected.")
    screening_results = []
    summary = {}
    start = time.time()
    if args.screening_file:
        prior = json.loads((ROOT / args.screening_file).read_text())
        screening_results = prior["screening_runs"]
        summary = prior["candidate_summary"]
        configs = [item["config"] for item in summary.values()]
        print(f"Reusing validation-only screening from {args.screening_file}.", flush=True)
    else:
        for fold_name in active_folds:
            for config in configs:
                prepared = prepare_fold(data_path, ratio_path, fold_name, config["history"])
                prepared["fold"] = fold_name
                for seed in args.screen_seeds:
                    screening_results.append(run_fit(prepared, config, seed=seed, device=device, max_epochs=args.max_epochs, patience=args.patience, negative_ratio=args.negative_ratio, evaluate_test=False))
            print(f"Completed validation screening for fold={fold_name}.", flush=True)
            if device.type == "mps":
                torch.mps.empty_cache()

        for config in configs:
            rows = [r for r in screening_results if r["config"] == config]
            summary[config_key(config)] = {
                "config": config,
                "mean_validation_ap": float(np.mean([r["validation"]["average_precision"] for r in rows])),
                "sd_validation_ap": float(np.std([r["validation"]["average_precision"] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "mean_validation_roc_auc": float(np.mean([r["validation"]["roc_auc"] for r in rows])),
                "fold_seed_scores": [{"fold": r["fold"], "seed": r["seed"], **r["validation"]} for r in rows],
            }
    ordered = sorted(summary.values(), key=lambda x: (x["mean_validation_ap"], x["mean_validation_roc_auc"]), reverse=True)
    fold_rankings = {}
    selected_configs_by_fold = {}
    for fold_name in active_folds:
        fold_candidates = []
        for item in summary.values():
            rows = [r for r in item["fold_seed_scores"] if r["fold"] == fold_name]
            if not rows:
                continue
            fold_candidates.append({
                "config": item["config"],
                "mean_validation_ap": float(np.mean([r["average_precision"] for r in rows])),
                "mean_validation_roc_auc": float(np.mean([r["roc_auc"] for r in rows])),
                "seed_scores": rows,
            })
        fold_candidates.sort(key=lambda x: (x["mean_validation_ap"], x["mean_validation_roc_auc"]), reverse=True)
        fold_rankings[fold_name] = fold_candidates
        selected_configs_by_fold[fold_name] = [item["config"] for item in fold_candidates[: args.top_configs]]
    report = {
        "protocol": {
            "status": "fiscal-year exploratory; no exact filing-availability dates in source",
            "screening_metric": "validation average precision; ROC-AUC tie-break",
            "folds": active_folds,
            "screen_seeds": args.screen_seeds,
            "final_seeds": args.final_seeds,
            "negative_sampling_ratio": args.negative_ratio,
            "test_used_for_selection": False,
        },
        "environment": {"device": str(device), "torch": torch.__version__, "numpy": np.__version__},
        "candidate_summary": summary,
        "fold_validation_rankings": fold_rankings,
        "selected_configs_by_fold": selected_configs_by_fold,
        "screening_runs": [{k: v for k, v in r.items() if k != "trace"} for r in screening_results],
        "elapsed_seconds_before_final": time.time() - start,
    }

    if args.validation_only:
        selected_validation = []
        for fold_name in active_folds:
            for config in selected_configs_by_fold[fold_name]:
                prepared = prepare_fold(data_path, ratio_path, fold_name, config["history"])
                prepared["fold"] = fold_name
                for seed in args.final_seeds:
                    selected_validation.append(run_fit(prepared, config, seed=seed, device=device, max_epochs=args.max_epochs, patience=args.patience, negative_ratio=args.negative_ratio, evaluate_test=False))
        report["selected_validation_runs"] = selected_validation
    elif not args.screen_only:
        final_results = []
        for fold_name in active_folds:
            for config in selected_configs_by_fold[fold_name]:
                prepared = prepare_fold(data_path, ratio_path, fold_name, config["history"])
                prepared["fold"] = fold_name
                for seed in args.final_seeds:
                    final_results.append(run_fit(prepared, config, seed=seed, device=device, max_epochs=args.max_epochs, patience=args.patience, negative_ratio=args.negative_ratio, evaluate_test=True))
        report["frozen_test_runs"] = final_results
        report["final_summary"] = {}
        for fold_name, configs_for_fold in selected_configs_by_fold.items():
            for config in configs_for_fold:
                key = config_key(config)
                rows = [r for r in final_results if r["config"] == config and r["fold"] == fold_name]
                labels = np.asarray(rows[0]["test_predictions"]["labels"], dtype=int)
                ensemble_scores = np.mean([np.asarray(r["test_predictions"]["scores"]) for r in rows], axis=0)
                report["final_summary"][f"{fold_name}|{key}"] = {
                    "fold": fold_name,
                    "selected_on_validation_only": True,
                    "seed_metrics": [r["test"] for r in rows],
                    "ensemble_mean_seed_scores": score_metrics(labels, ensemble_scores),
                }
    report["elapsed_seconds_total"] = time.time() - start
    (out / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"device": str(device), "validation_ranking": [{"config": item["config"], "mean_validation_ap": item["mean_validation_ap"], "sd_validation_ap": item["sd_validation_ap"], "mean_validation_roc_auc": item["mean_validation_roc_auc"]} for item in ordered], "fold_selected": {fold: [config_key(c) for c in configs] for fold, configs in selected_configs_by_fold.items()}, "final_summary": report.get("final_summary", {}), "output": str(out / "experiment.json")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
