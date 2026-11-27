import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.ablation import AblationExperiment, AblationStudyManager
from benchmark.cross_domain import CrossDomainBenchmark
from benchmark.temporal_cv import TemporalLeakageFreeCV
from loader.data_loader import UniversalDataLoader
from preprocessor.data_config import DATASET_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Run prespecified revision experiments")
    parser.add_argument("config", type=Path, help="Path to experiment JSON")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "revision")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = config.get("seeds", [42, 123, 456, 789, 1011])
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    targets = config.get("precision_targets", [0.5, 0.7, 0.8, 0.9, 0.95])
    experiments = config["datasets"]
    if not experiments:
        raise ValueError("Specify at least one dataset and its configurations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "data_format": args.data_format, "data_root": str(args.data_root.resolve())}
    for package in ("scikit-learn", "torch", "xgboost", "imbalanced-learn", "pyarrow"):
        from importlib.metadata import version, PackageNotFoundError
        try:
            metadata[package] = version(package)
        except PackageNotFoundError:
            metadata[package] = None
    (run_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manager = AblationStudyManager(output_dir=str(run_dir))
    results = []
    for filename, definitions in experiments.items():
        dataset_config = DATASET_CONFIG[filename]
        if not definitions:
            raise ValueError(f"No experiments specified for {filename}")
        label = dataset_config["name"]
        benchmark = CrossDomainBenchmark(
            datasets={label: filename}, output_dir=str(run_dir), data_root=str(args.data_root),
            random_seeds=seeds, sensitivity_targets=targets,
            gan_cache_path=str(run_dir / "synthetic_cache" / Path(filename).stem),
            **config.get("benchmark_options", {}),
        )
        loader = UniversalDataLoader(filename, str(PROJECT_ROOT), str(args.data_root), verbose=False, data_format=args.data_format)
        temporal_cv = TemporalLeakageFreeCV(gap_config=benchmark.gap_configs[filename])
        seen = set()
        for definition in definitions:
            exp_id = definition["exp_id"]
            if exp_id in seen:
                raise ValueError(f"Duplicate experiment ID {exp_id} for {filename}")
            seen.add(exp_id)
            components = {name: component.baseline for name, component in manager.components.items()}
            components.update(definition["components"])
            for name, option in components.items():
                if name not in manager.components or option not in manager.components[name].options:
                    raise ValueError(f"Invalid component {name}={option}")
            experiment = AblationExperiment(exp_id, components)
            cache_id = hashlib.sha256(exp_id.encode("utf-8")).hexdigest()[:16]
            benchmark.gan_cache_path = str(run_dir / "synthetic_cache" / Path(filename).stem / cache_id)
            for seed in seeds:
                started = time.perf_counter()
                row = {"dataset": label, "filename": filename, "exp_id": exp_id, "seed": seed, **components}
                sensitivity_start = len(benchmark.sensitivity_results)
                try:
                    row.update(benchmark.run_single_experiment(label, experiment, seed, loader, temporal_cv))
                    row["status"] = "success"
                except Exception as exc:
                    del benchmark.sensitivity_results[sensitivity_start:]
                    logging.exception("Failed %s / %s / %s", filename, exp_id, seed)
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                row["elapsed_seconds"] = time.perf_counter() - started
                results.append(row)
                pd.DataFrame(results).to_csv(run_dir / "runs.csv", index=False)
                pd.DataFrame(benchmark.sensitivity_results).to_csv(run_dir / f"sensitivity_{Path(filename).stem}.csv", index=False)
        logging.info("Completed dataset %s", filename)
    frame = pd.DataFrame(results)
    successful = frame[frame.status == "success"]
    if not successful.empty:
        summary = successful.groupby(["dataset", "exp_id"], as_index=False).agg(
            n_success=("pr_auc", "count"), pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"), pr_auc_min=("pr_auc", "min"), pr_auc_max=("pr_auc", "max"),
        )
        summary["n_expected"] = len(seeds)
        summary.to_csv(run_dir / "summary.csv", index=False)
        paired_rows = []
        for filename, definitions in experiments.items():
            baseline_id = config.get("baseline_ids", {}).get(filename)
            if not baseline_id:
                continue
            label = DATASET_CONFIG[filename]["name"]
            subset = successful[successful["dataset"] == label]
            baseline = subset[subset.exp_id == baseline_id][["seed", "pr_auc"]]
            for definition in definitions:
                exp_id = definition["exp_id"]
                if exp_id == baseline_id:
                    continue
                current = subset[subset.exp_id == exp_id][["seed", "pr_auc"]]
                matched = baseline.merge(current, on="seed", suffixes=("_baseline", "_experiment"), validate="one_to_one")
                delta = matched.pr_auc_experiment - matched.pr_auc_baseline
                paired_rows.append({"dataset": label, "baseline": baseline_id, "exp_id": exp_id,
                                    "n_pairs": len(delta), "n_expected": len(seeds),
                                    "delta_mean": delta.mean(), "delta_std": delta.std()})
        if paired_rows:
            pd.DataFrame(paired_rows).to_csv(run_dir / "paired_deltas.csv", index=False)
    logging.info("Results: %s", run_dir)
    if (frame.status == "failed").any():
        raise SystemExit("One or more experiments failed; inspect runs.csv")


if __name__ == "__main__":
    main()
