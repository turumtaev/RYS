#!/usr/bin/env python3
"""Beam-search relayer exploration using measured single-block results.

Workflow:
1) Validate arbitrary multi-block layer expansion (including repeated/double blocks).
2) Load existing single-block Math/EQ results (seed state).
3) Build a candidate block pool from top single-block scores (Method B style z-delta).
4) Run beam expansion across block sequences (depth >= 2), allowing repeated blocks.
5) Evaluate only unseen candidates via existing custom-config workers:
   - src.workers.math_worker
   - src.workers.eq_worker

This script is resume-friendly: evaluated candidates are cached under --work-dir.
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def expand_multi_block_config(num_layers: int, blocks: tuple[tuple[int, int], ...]) -> list[int]:
    """Pure-Python multi-block expansion (matches src.core.layer_duplicator behavior)."""
    if not blocks:
        return list(range(num_layers))

    i0, j0 = blocks[0]
    result = list(range(0, j0)) + list(range(i0, num_layers)) if (i0, j0) != (0, 0) else list(range(num_layers))

    for i, j in blocks[1:]:
        if (i, j) == (0, 0):
            continue
        insert_layers = list(range(i, j))
        if not insert_layers:
            continue
        try:
            insert_pos = len(result) - 1 - result[::-1].index(j - 1) + 1
        except ValueError:
            continue
        result = result[:insert_pos] + insert_layers + result[insert_pos:]

    return result


@dataclass
class ZStats:
    baseline_math: float
    baseline_eq: float
    math_delta_mean: float
    math_delta_std: float
    eq_delta_mean: float
    eq_delta_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beam search for multi-block relayering.")

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-layers", type=int, default=64)

    parser.add_argument(
        "--seed-math-results",
        default="results/math_results.pkl",
        help="Existing single-block math results (keys like (i,j)).",
    )
    parser.add_argument(
        "--seed-eq-results",
        default="results/eq_results.pkl",
        help="Existing single-block EQ results (keys like (i,j)).",
    )
    parser.add_argument(
        "--seed-rescore-config-file",
        default=None,
        help=(
            "Optional block-spec config file to re-score on the active datasets "
            "before beam expansion (e.g., baseline + top-N single blocks)."
        ),
    )
    parser.add_argument(
        "--seed-rescore-math-results",
        default=None,
        help="Optional output pickle for seed-rescore math scores (layer-keyed).",
    )
    parser.add_argument(
        "--seed-rescore-eq-results",
        default=None,
        help="Optional output pickle for seed-rescore EQ scores (layer-keyed).",
    )
    parser.add_argument(
        "--seed-rescore-reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip seed-rescore worker run when both rescore result files already exist.",
    )
    parser.add_argument(
        "--seed-rescore-require-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require baseline config in the seed-rescore config set.",
    )

    parser.add_argument("--math-dataset-path", default="datasets/math_16.json")
    parser.add_argument("--eq-dataset-path", default="datasets/eq_16.json")

    parser.add_argument("--work-dir", default="results/beam-search")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--start-depth", type=int, default=2, help="Beam depth to start from (default: 2).")
    parser.add_argument("--max-depth", type=int, default=3, help="1=single blocks only, 2=double blocks, etc.")
    parser.add_argument(
        "--seed-top-k",
        type=int,
        default=16,
        help="Top single-block seeds used as depth-1 frontier.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=64,
        help="Top single blocks to use as expansion pool.",
    )
    parser.add_argument(
        "--expand-per-node",
        type=int,
        default=16,
        help="How many top pool blocks to try appending per frontier node.",
    )
    parser.add_argument(
        "--max-candidates-per-depth",
        type=int,
        default=64,
        help="Upper cap on newly evaluated candidates per depth.",
    )
    parser.add_argument(
        "--max-extra-layers",
        type=int,
        default=None,
        help="Optional cap on duplicated layers count: len(layer_list)-num_layers <= cap.",
    )
    parser.add_argument(
        "--min-beam-hours",
        type=float,
        default=0.0,
        help="Do not allow plateau-based early stop until this many beam hours have elapsed.",
    )
    parser.add_argument(
        "--max-beam-hours",
        type=float,
        default=None,
        help="Hard beam-phase wall-clock stop (hours).",
    )
    parser.add_argument(
        "--plateau-min-improvement",
        type=float,
        default=0.0,
        help="Minimum method-score improvement to reset plateau streak (<=0 disables plateau stopping).",
    )
    parser.add_argument(
        "--plateau-streak",
        type=int,
        default=0,
        help="Consecutive depths below improvement threshold before plateau stop (requires >0).",
    )
    parser.add_argument(
        "--plateau-no-replace-streak",
        type=int,
        default=0,
        help="Consecutive depths with no global best replacement before plateau stop (requires >0).",
    )

    parser.add_argument("--math-batch-size", type=int, default=16)
    parser.add_argument("--eq-batch-size", type=int, default=16)
    parser.add_argument("--math-max-new", type=int, default=64)
    parser.add_argument("--eq-max-new", type=int, default=64)
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional limit on benchmark examples per worker for fast local debugging.",
    )

    parser.add_argument("--padding-mode", default="inprompt_space", choices=["masked", "inprompt_space"])
    parser.add_argument("--attention-impl", default="eager", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype forwarded to Hugging Face workers.",
    )
    parser.add_argument(
        "--math-device-map",
        default=None,
        help="Override --device-map for math worker (e.g., cuda:0).",
    )
    parser.add_argument(
        "--eq-device-map",
        default=None,
        help="Override --device-map for EQ worker (e.g., cuda:1).",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-worker-preflight", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Prepare/print beam steps without worker eval.")
    parser.add_argument(
        "--dynamic-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use queue-mode workers with dynamic cross-metric GPU handoff.",
    )
    parser.add_argument(
        "--allow-cross-metric-handoff",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When one metric finishes early, reassign the freed GPU to the other metric queue.",
    )
    parser.add_argument(
        "--monitor-interval-sec",
        type=int,
        default=20,
        help="Polling interval for dynamic queue worker monitoring.",
    )
    parser.add_argument(
        "--overhead-penalty-lambda",
        type=float,
        default=0.0,
        help=(
            "End-stage relative-overhead penalty for final ranking: "
            "final_score = method_score - lambda * ((len(layer_key)-num_layers)/num_layers)."
        ),
    )

    return parser.parse_args()


def _extract_score(raw: Any) -> float | None:
    if isinstance(raw, dict):
        if "score" in raw:
            raw = raw["score"]
        elif "math_score" in raw:
            raw = raw["math_score"]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def load_pair_score_map(path: Path) -> dict[tuple[int, int], float]:
    with path.open("rb") as f:
        data = pickle.load(f)

    out: dict[tuple[int, int], float] = {}
    for k, v in data.items():
        if not isinstance(k, (tuple, list)) or len(k) != 2:
            continue
        try:
            key = (int(k[0]), int(k[1]))
        except Exception:
            continue
        score = _extract_score(v)
        if score is None:
            continue
        out[key] = score
    return out


def load_layer_score_map(path: Path) -> dict[tuple[int, ...], float]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = pickle.load(f)

    out: dict[tuple[int, ...], float] = {}
    for k, v in data.items():
        if not isinstance(k, (tuple, list)):
            continue
        try:
            key = tuple(int(x) for x in k)
        except Exception:
            continue
        score = _extract_score(v)
        if score is None:
            continue
        out[key] = score
    return out


def safe_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = float(statistics.mean(values))
    std = float(statistics.pstdev(values))
    if std < 1e-12:
        std = 1.0
    return mean, std


def score_method_b(math_score: float, eq_score: float, stats: ZStats) -> tuple[float, float, float]:
    math_delta = float(math_score - stats.baseline_math)
    eq_delta = float(eq_score - stats.baseline_eq)
    z_math = (math_delta - stats.math_delta_mean) / stats.math_delta_std
    z_eq = (eq_delta - stats.eq_delta_mean) / stats.eq_delta_std
    return float(z_math + z_eq), math_delta, eq_delta


def extra_layers_from_key(layer_key: tuple[int, ...], num_layers: int) -> int:
    return int(len(layer_key) - num_layers)


def relative_overhead_from_key(layer_key: tuple[int, ...], num_layers: int) -> float:
    if num_layers <= 0:
        return 0.0
    return float(extra_layers_from_key(layer_key, num_layers) / float(num_layers))


def final_score_with_overhead(
    *,
    method_score: float,
    layer_key: tuple[int, ...],
    num_layers: int,
    penalty_lambda: float,
) -> float:
    rel = relative_overhead_from_key(layer_key, num_layers)
    return float(method_score - (penalty_lambda * rel))


def apply_efficiency_fields(
    entry: dict[str, Any],
    *,
    num_layers: int,
    penalty_lambda: float,
) -> None:
    layer_key = tuple(int(x) for x in entry["layer_key"])
    method_score = float(entry["method_score"])
    extra_layers = extra_layers_from_key(layer_key, num_layers)
    rel_overhead = relative_overhead_from_key(layer_key, num_layers)
    entry["extra_layers"] = int(extra_layers)
    entry["relative_overhead"] = float(rel_overhead)
    entry["final_score"] = float(
        final_score_with_overhead(
            method_score=method_score,
            layer_key=layer_key,
            num_layers=num_layers,
            penalty_lambda=penalty_lambda,
        )
    )


def rank_key(entry: dict[str, Any], *, use_final_score: bool) -> tuple[float, float, float, float]:
    primary = float(entry["final_score"] if use_final_score else entry["method_score"])
    return (
        primary,
        float(entry["method_score"]),
        float(entry["eq_delta"]),
        float(entry["math_delta"]),
    )


def blocks_to_spec(blocks: tuple[tuple[int, int], ...]) -> str:
    if not blocks:
        return "0,0"
    return ";".join(f"{i},{j}" for i, j in blocks)


def blocks_to_layer_key(num_layers: int, blocks: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    return tuple(expand_multi_block_config(num_layers, blocks))


def parse_block_spec(spec: str) -> tuple[tuple[int, int], ...]:
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty block specification.")
    if raw.lower().startswith("blocks:"):
        raw = raw.split(":", 1)[1].strip()
    if raw in {"0,0", "(0,0)", "(0, 0)"}:
        return tuple()

    blocks: list[tuple[int, int]] = []
    for pair in raw.split(";"):
        p = pair.strip()
        if not p:
            continue
        if p.startswith("(") and p.endswith(")"):
            p = p[1:-1].strip()
        parts = [x.strip() for x in p.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid block spec pair: {pair!r}")
        i, j = int(parts[0]), int(parts[1])
        blocks.append((i, j))
    if not blocks:
        return tuple()
    return tuple(blocks)


def load_block_specs(path: Path) -> list[tuple[tuple[int, int], ...]]:
    specs: list[tuple[tuple[int, int], ...]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            if row.lower().startswith("layers:"):
                raise ValueError(
                    f"Seed rescore file must use block specs, not canonical layer lines: {row[:40]!r}"
                )
            specs.append(parse_block_spec(row))
    if not specs:
        raise ValueError(f"No block specs found in {path}")
    return specs


def validate_arbitrary_layer_scheme() -> None:
    # Canonical example from design notes.
    n = 9
    base = ((2, 7),)
    expected = [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8]
    got = expand_multi_block_config(n, base)
    if got != expected:
        raise RuntimeError(
            "Layer expansion validation failed for (2,7). "
            f"Expected {expected}, got {got}."
        )

    # Double-duplicate same block should add one more [2..6] segment.
    dbl = expand_multi_block_config(n, ((2, 7), (2, 7)))
    if len(dbl) != len(expected) + (7 - 2):
        raise RuntimeError(
            "Double-duplicate validation failed: unexpected length "
            f"{len(dbl)} (expected {len(expected) + (7 - 2)})."
        )
    for layer in range(2, 7):
        if dbl.count(layer) != 3:
            raise RuntimeError(
                "Double-duplicate validation failed: layer "
                f"{layer} appears {dbl.count(layer)} times (expected 3)."
            )


def build_seed_entries(
    *,
    num_layers: int,
    seed_math: dict[tuple[int, int], float],
    seed_eq: dict[tuple[int, int], float],
) -> tuple[dict[tuple[int, ...], dict[str, Any]], list[dict[str, Any]], ZStats]:
    common = sorted(set(seed_math) & set(seed_eq))
    if (0, 0) not in common:
        raise RuntimeError("Seed results are missing baseline key (0,0).")

    baseline_math = seed_math[(0, 0)]
    baseline_eq = seed_eq[(0, 0)]

    delta_math_vals: list[float] = []
    delta_eq_vals: list[float] = []
    single_rows: list[dict[str, Any]] = []

    for key in common:
        if key == (0, 0):
            continue
        math_score = seed_math[key]
        eq_score = seed_eq[key]
        d_math = float(math_score - baseline_math)
        d_eq = float(eq_score - baseline_eq)
        delta_math_vals.append(d_math)
        delta_eq_vals.append(d_eq)
        single_rows.append(
            {
                "block": (int(key[0]), int(key[1])),
                "math_score": float(math_score),
                "eq_score": float(eq_score),
                "math_delta": d_math,
                "eq_delta": d_eq,
            }
        )

    math_mean, math_std = safe_mean_std(delta_math_vals)
    eq_mean, eq_std = safe_mean_std(delta_eq_vals)
    stats = ZStats(
        baseline_math=float(baseline_math),
        baseline_eq=float(baseline_eq),
        math_delta_mean=math_mean,
        math_delta_std=math_std,
        eq_delta_mean=eq_mean,
        eq_delta_std=eq_std,
    )

    evaluated: dict[tuple[int, ...], dict[str, Any]] = {}

    # Baseline entry
    baseline_blocks: tuple[tuple[int, int], ...] = tuple()
    baseline_key = blocks_to_layer_key(num_layers, baseline_blocks)
    baseline_score, baseline_d_math, baseline_d_eq = score_method_b(
        baseline_math,
        baseline_eq,
        stats,
    )
    evaluated[baseline_key] = {
        "blocks": baseline_blocks,
        "block_spec": blocks_to_spec(baseline_blocks),
        "layer_key": baseline_key,
        "depth": 0,
        "math_score": float(baseline_math),
        "eq_score": float(baseline_eq),
        "math_delta": baseline_d_math,
        "eq_delta": baseline_d_eq,
        "method_score": baseline_score,
        "source": "seed",
    }

    # Single-block seed entries from existing results
    for row in single_rows:
        block = row["block"]
        blocks = (block,)
        layer_key = blocks_to_layer_key(num_layers, blocks)
        method_score, d_math, d_eq = score_method_b(row["math_score"], row["eq_score"], stats)
        evaluated[layer_key] = {
            "blocks": blocks,
            "block_spec": blocks_to_spec(blocks),
            "layer_key": layer_key,
            "depth": 1,
            "math_score": row["math_score"],
            "eq_score": row["eq_score"],
            "math_delta": d_math,
            "eq_delta": d_eq,
            "method_score": method_score,
            "source": "seed",
        }

    # Rank single blocks for pool/frontier construction
    ranked_single = sorted(
        [e for e in evaluated.values() if e["depth"] == 1],
        key=lambda e: (e["method_score"], e["eq_delta"], e["math_delta"]),
        reverse=True,
    )

    return evaluated, ranked_single, stats


def build_seed_entries_from_rescored_specs(
    *,
    num_layers: int,
    block_specs: list[tuple[tuple[int, int], ...]],
    math_layer_scores: dict[tuple[int, ...], float],
    eq_layer_scores: dict[tuple[int, ...], float],
    require_baseline: bool,
) -> tuple[dict[tuple[int, ...], dict[str, Any]], list[dict[str, Any]], ZStats]:
    keyed_rows: dict[tuple[int, ...], dict[str, Any]] = {}
    missing_rows = 0
    for blocks in block_specs:
        layer_key = blocks_to_layer_key(num_layers, blocks)
        if layer_key not in math_layer_scores or layer_key not in eq_layer_scores:
            missing_rows += 1
            continue
        keyed_rows[layer_key] = {
            "blocks": blocks,
            "layer_key": layer_key,
            "depth": 0 if not blocks else len(blocks),
            "math_score": float(math_layer_scores[layer_key]),
            "eq_score": float(eq_layer_scores[layer_key]),
        }

    if missing_rows:
        print(
            f"Seed rescore warning: {missing_rows} config(s) were missing math/eq scores and were skipped."
        )
    if not keyed_rows:
        raise RuntimeError("Seed rescore produced no usable entries.")

    baseline_key = tuple(range(num_layers))
    if require_baseline and baseline_key not in keyed_rows:
        raise RuntimeError(
            "Seed rescore entries are missing baseline layers; rerun seed rescore with (0,0) included."
        )
    if baseline_key not in keyed_rows:
        # Fallback for robustness if caller explicitly disabled strict baseline.
        # Pick the shortest layer-key as proxy baseline.
        baseline_key = min(keyed_rows.keys(), key=len)
        print(
            "Seed rescore note: canonical baseline missing; using shortest-layer-key proxy baseline "
            f"(len={len(baseline_key)})."
        )

    baseline_math = float(keyed_rows[baseline_key]["math_score"])
    baseline_eq = float(keyed_rows[baseline_key]["eq_score"])

    delta_math_vals: list[float] = []
    delta_eq_vals: list[float] = []
    for lk, row in keyed_rows.items():
        if lk == baseline_key:
            continue
        delta_math_vals.append(float(row["math_score"] - baseline_math))
        delta_eq_vals.append(float(row["eq_score"] - baseline_eq))

    math_mean, math_std = safe_mean_std(delta_math_vals)
    eq_mean, eq_std = safe_mean_std(delta_eq_vals)
    stats = ZStats(
        baseline_math=baseline_math,
        baseline_eq=baseline_eq,
        math_delta_mean=math_mean,
        math_delta_std=math_std,
        eq_delta_mean=eq_mean,
        eq_delta_std=eq_std,
    )

    evaluated: dict[tuple[int, ...], dict[str, Any]] = {}
    for lk, row in keyed_rows.items():
        m = float(row["math_score"])
        q = float(row["eq_score"])
        method_score, d_m, d_q = score_method_b(m, q, stats)
        blocks = tuple((int(b[0]), int(b[1])) for b in row["blocks"])
        entry = {
            "blocks": blocks,
            "block_spec": blocks_to_spec(blocks),
            "layer_key": lk,
            "depth": int(row["depth"]),
            "math_score": m,
            "eq_score": q,
            "math_delta": d_m,
            "eq_delta": d_q,
            "method_score": method_score,
            "source": "seed_rescore",
        }
        evaluated[lk] = entry

    ranked_single = sorted(
        [e for e in evaluated.values() if int(e.get("depth", -1)) == 1],
        key=lambda e: (e["method_score"], e["eq_delta"], e["math_delta"]),
        reverse=True,
    )
    if not ranked_single:
        raise RuntimeError("Seed rescore set has no depth-1 single-block entries.")

    return evaluated, ranked_single, stats


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def serialize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    out["blocks"] = [list(b) for b in entry["blocks"]]
    out["layer_key"] = list(entry["layer_key"])
    return out


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def upsert_registry(
    registry: dict[str, dict[str, Any]],
    *,
    block_spec: str,
    layer_key: tuple[int, ...],
    depth: int,
    status: str,
    source: str,
    method_score: float | None = None,
    math_score: float | None = None,
    eq_score: float | None = None,
    bump_counts: bool = True,
) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec = registry.get(block_spec)
    if rec is None:
        rec = {
            "block_spec": block_spec,
            "layer_key": list(layer_key),
            "first_seen": now,
            "proposed_count": 0,
            "evaluated_count": 0,
        }

    rec["last_seen"] = now
    rec["depth"] = int(depth)
    rec["layer_key"] = list(layer_key)
    rec["last_status"] = status
    rec["source"] = source

    if bump_counts and status in {"proposed", "planned"}:
        rec["proposed_count"] = int(rec.get("proposed_count", 0)) + 1
    if bump_counts and status == "evaluated":
        rec["evaluated_count"] = int(rec.get("evaluated_count", 0)) + 1

    if method_score is not None:
        rec["method_score"] = float(method_score)
        prev_best = rec.get("best_method_score")
        if prev_best is None or float(method_score) > float(prev_best):
            rec["best_method_score"] = float(method_score)
    if math_score is not None:
        rec["math_score"] = float(math_score)
    if eq_score is not None:
        rec["eq_score"] = float(eq_score)

    registry[block_spec] = rec


def run_worker(
    *,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    print(" ".join(cmd))
    if dry_run:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log_f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed with code {proc.returncode}. See log: {log_path}")


def run_workers_parallel(
    *,
    runs: list[tuple[list[str], Path]],
    cwd: Path,
    dry_run: bool,
) -> None:
    for cmd, _ in runs:
        print(" ".join(cmd))
    if dry_run:
        return

    handles = []
    procs: list[tuple[subprocess.Popen, Path]] = []
    try:
        for cmd, log_path in runs:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_f = log_path.open("a", encoding="utf-8")
            handles.append(log_f)
            log_f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)}\n")
            log_f.flush()
            proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=log_f, stderr=subprocess.STDOUT)
            procs.append((proc, log_path))

        failures: list[tuple[int, Path]] = []
        for proc, log_path in procs:
            rc = proc.wait()
            if rc != 0:
                failures.append((rc, log_path))
        if failures:
            details = ", ".join(f"code={rc} log={path}" for rc, path in failures)
            raise RuntimeError(f"Parallel workers failed: {details}")
    finally:
        for log_f in handles:
            try:
                log_f.close()
            except Exception:
                pass


def workers_share_device(args: argparse.Namespace) -> bool:
    """Return True when math and EQ workers target the same device map."""
    math_device = args.math_device_map or args.device_map
    eq_device = args.eq_device_map or args.device_map
    return str(math_device) == str(eq_device)


def run_workers(
    *,
    runs: list[tuple[list[str], Path]],
    cwd: Path,
    dry_run: bool,
    sequential: bool,
) -> None:
    if sequential:
        for cmd, log_path in runs:
            run_worker(cmd=cmd, cwd=cwd, log_path=log_path, dry_run=dry_run)
        return
    run_workers_parallel(runs=runs, cwd=cwd, dry_run=dry_run)


def build_math_worker_cmd(
    *,
    args: argparse.Namespace,
    config_file: Path | None,
    queue_file: Path | None,
    results_file: Path,
    depth: int,
    worker_suffix: str = "",
    device_override: str | None = None,
) -> list[str]:
    if bool(config_file) == bool(queue_file):
        raise ValueError("Exactly one of config_file or queue_file must be provided for math worker.")

    device_map = device_override or args.math_device_map or args.device_map
    worker_id = f"beam_math_d{depth}{worker_suffix}"
    cmd = [
        args.python_bin,
        "-m",
        "src.workers.math_worker",
        "--model-path",
        args.model_path,
        "--dataset-path",
        args.math_dataset_path,
        "--results-file",
        str(results_file),
        "--batch-size",
        str(args.math_batch_size),
        "--max-new",
        str(args.math_max_new),
        "--padding-mode",
        args.padding_mode,
        "--attention-impl",
        args.attention_impl,
        "--device-map",
        device_map,
        "--torch-dtype",
        args.torch_dtype,
        "--worker-id",
        worker_id,
    ]
    if config_file is not None:
        cmd.extend(["--config-file", str(config_file)])
    if queue_file is not None:
        cmd.extend(["--queue-file", str(queue_file)])
    if args.dataset_limit is not None:
        cmd.extend(["--dataset-limit", str(args.dataset_limit)])
    if args.skip_worker_preflight:
        cmd.append("--skip-preflight")
    if args.local_files_only:
        cmd.append("--local-files-only")
    else:
        cmd.append("--no-local-files-only")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    else:
        cmd.append("--no-trust-remote-code")
    return cmd


def build_eq_worker_cmd(
    *,
    args: argparse.Namespace,
    config_file: Path | None,
    queue_file: Path | None,
    results_file: Path,
    depth: int,
    worker_suffix: str = "",
    device_override: str | None = None,
) -> list[str]:
    if bool(config_file) == bool(queue_file):
        raise ValueError("Exactly one of config_file or queue_file must be provided for EQ worker.")

    device_map = device_override or args.eq_device_map or args.device_map
    worker_id = f"beam_eq_d{depth}{worker_suffix}"
    cmd = [
        args.python_bin,
        "-m",
        "src.workers.eq_worker",
        "--model-path",
        args.model_path,
        "--dataset-path",
        args.eq_dataset_path,
        "--results-file",
        str(results_file),
        "--batch-size",
        str(args.eq_batch_size),
        "--max-new",
        str(args.eq_max_new),
        "--padding-mode",
        args.padding_mode,
        "--attention-impl",
        args.attention_impl,
        "--device-map",
        device_map,
        "--torch-dtype",
        args.torch_dtype,
        "--worker-id",
        worker_id,
    ]
    if config_file is not None:
        cmd.extend(["--config-file", str(config_file)])
    if queue_file is not None:
        cmd.extend(["--queue-file", str(queue_file)])
    if args.dataset_limit is not None:
        cmd.extend(["--dataset-limit", str(args.dataset_limit)])
    if args.skip_worker_preflight:
        cmd.append("--skip-preflight")
    if args.local_files_only:
        cmd.append("--local-files-only")
    else:
        cmd.append("--no-local-files-only")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    else:
        cmd.append("--no-trust-remote-code")
    return cmd


def write_queue_file(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f)


def queue_remaining_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return 0
        data = json.loads(raw)
        if isinstance(data, list):
            return len(data)
    except Exception:
        pass
    return 0


def run_depth_workers_dynamic(
    *,
    args: argparse.Namespace,
    depth: int,
    work_dir: Path,
    math_queue_file: Path,
    eq_queue_file: Path,
    math_results_file: Path,
    eq_results_file: Path,
    dry_run: bool,
) -> None:
    math_device = args.math_device_map or args.device_map
    eq_device = args.eq_device_map or args.device_map

    if dry_run:
        print(f"[dry-run] dynamic depth {depth}: math_queue={math_queue_file} eq_queue={eq_queue_file}")
        return

    slots: list[dict[str, Any]] = [
        {
            "name": "slot0",
            "device": math_device,
            "preferred_metric": "math",
            "metric": "math",
            "proc": None,
            "log_handle": None,
        },
        {
            "name": "slot1",
            "device": eq_device,
            "preferred_metric": "eq",
            "metric": "eq",
            "proc": None,
            "log_handle": None,
        },
    ]

    worker_seq = {"math": 0, "eq": 0}

    def _metric_remaining(metric: str) -> int:
        return queue_remaining_count(math_queue_file if metric == "math" else eq_queue_file)

    def _spawn(slot: dict[str, Any], metric: str) -> None:
        worker_seq[metric] += 1
        suffix = f"_{slot['name']}_{worker_seq[metric]}"
        if metric == "math":
            cmd = build_math_worker_cmd(
                args=args,
                config_file=None,
                queue_file=math_queue_file,
                results_file=math_results_file,
                depth=depth,
                worker_suffix=suffix,
                device_override=str(slot["device"]),
            )
        else:
            cmd = build_eq_worker_cmd(
                args=args,
                config_file=None,
                queue_file=eq_queue_file,
                results_file=eq_results_file,
                depth=depth,
                worker_suffix=suffix,
                device_override=str(slot["device"]),
            )

        log_path = work_dir / f"beam_{metric}_worker_{slot['name']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("a", encoding="utf-8")
        log_f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)}\n")
        log_f.flush()
        print(" ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT)

        old_handle = slot.get("log_handle")
        if old_handle is not None:
            try:
                old_handle.close()
            except Exception:
                pass
        slot["metric"] = metric
        slot["proc"] = proc
        slot["log_handle"] = log_f

    def _cleanup_slot(slot: dict[str, Any]) -> None:
        handle = slot.get("log_handle")
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        slot["proc"] = None
        slot["log_handle"] = None

    # Initial assignment.
    for slot in slots:
        preferred = str(slot["preferred_metric"])
        if _metric_remaining(preferred) > 0:
            _spawn(slot, preferred)
            continue
        other = "eq" if preferred == "math" else "math"
        if args.allow_cross_metric_handoff and _metric_remaining(other) > 0:
            _spawn(slot, other)

    last_status_print = 0.0
    while True:
        # Reap exited workers.
        for slot in slots:
            proc = slot.get("proc")
            if proc is None:
                continue
            rc = proc.poll()
            if rc is None:
                continue
            if rc != 0:
                print(
                    f"WARNING: depth {depth} worker exited non-zero "
                    f"(slot={slot['name']} metric={slot['metric']} rc={rc}); "
                    "will attempt restart if queue has remaining work."
                )
            _cleanup_slot(slot)

        rem_math = _metric_remaining("math")
        rem_eq = _metric_remaining("eq")
        all_idle = all(slot.get("proc") is None for slot in slots)
        if rem_math == 0 and rem_eq == 0 and all_idle:
            break

        # Immediate handoff: keep every free slot assigned to any metric with remaining work.
        for slot in slots:
            if slot.get("proc") is not None:
                continue
            current_metric = str(slot.get("metric") or slot.get("preferred_metric"))
            if _metric_remaining(current_metric) > 0:
                _spawn(slot, current_metric)
                continue
            if args.allow_cross_metric_handoff:
                other = "eq" if current_metric == "math" else "math"
                if _metric_remaining(other) > 0:
                    _spawn(slot, other)

        now = time.time()
        if now - last_status_print >= max(5, args.monitor_interval_sec):
            live = [
                f"{slot['name']}:{slot.get('metric')}:{'up' if slot.get('proc') is not None else 'idle'}"
                for slot in slots
            ]
            print(
                f"Depth {depth} dynamic status: rem_math={rem_math} rem_eq={rem_eq} "
                f"slots=[{', '.join(live)}]"
            )
            last_status_print = now

        time.sleep(max(1, args.monitor_interval_sec))

    for slot in slots:
        _cleanup_slot(slot)


def main() -> None:
    args = parse_args()

    if args.max_depth < 1:
        raise ValueError("--max-depth must be >= 1")
    if args.start_depth < 2:
        raise ValueError("--start-depth must be >= 2")
    if args.start_depth > args.max_depth:
        raise ValueError("--start-depth must be <= --max-depth")
    if args.beam_width < 1:
        raise ValueError("--beam-width must be >= 1")
    if args.pool_size < 1:
        raise ValueError("--pool-size must be >= 1")
    if args.seed_top_k < 1:
        raise ValueError("--seed-top-k must be >= 1")
    if args.expand_per_node < 1:
        raise ValueError("--expand-per-node must be >= 1")
    if args.max_candidates_per_depth < 1:
        raise ValueError("--max-candidates-per-depth must be >= 1")
    if args.min_beam_hours < 0:
        raise ValueError("--min-beam-hours must be >= 0")
    if args.max_beam_hours is not None and args.max_beam_hours <= 0:
        raise ValueError("--max-beam-hours must be > 0 when provided")
    if args.plateau_streak < 0:
        raise ValueError("--plateau-streak must be >= 0")
    if args.plateau_no_replace_streak < 0:
        raise ValueError("--plateau-no-replace-streak must be >= 0")
    if args.monitor_interval_sec < 1:
        raise ValueError("--monitor-interval-sec must be >= 1")
    if args.overhead_penalty_lambda < 0:
        raise ValueError("--overhead-penalty-lambda must be >= 0")
    if args.dataset_limit is not None and args.dataset_limit < 1:
        raise ValueError("--dataset-limit must be >= 1")

    shared_device = workers_share_device(args)
    if shared_device and args.dynamic_split:
        print(
            "Math and EQ workers share one device map; disabling dynamic split "
            "to avoid concurrent workers on the same device."
        )
        args.dynamic_split = False

    validate_arbitrary_layer_scheme()
    print("Arbitrary layer expansion validation: OK")

    work_dir = (ROOT / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    seed_math_path = (ROOT / args.seed_math_results).resolve() if not Path(args.seed_math_results).is_absolute() else Path(args.seed_math_results)
    seed_eq_path = (ROOT / args.seed_eq_results).resolve() if not Path(args.seed_eq_results).is_absolute() else Path(args.seed_eq_results)

    if not seed_math_path.exists() or not seed_eq_path.exists():
        raise FileNotFoundError(
            "Seed results missing. "
            f"math={seed_math_path} exists={seed_math_path.exists()} | "
            f"eq={seed_eq_path} exists={seed_eq_path.exists()}"
        )

    seed_math = load_pair_score_map(seed_math_path)
    seed_eq = load_pair_score_map(seed_eq_path)
    print(
        f"Loaded seed results: math={len(seed_math)} keys, eq={len(seed_eq)} keys, "
        f"common={len(set(seed_math) & set(seed_eq))}"
    )

    seed_rescore_config_path: Path | None = None
    seed_rescore_math_path: Path | None = None
    seed_rescore_eq_path: Path | None = None
    if args.seed_rescore_config_file:
        seed_rescore_config_path = (
            (ROOT / args.seed_rescore_config_file).resolve()
            if not Path(args.seed_rescore_config_file).is_absolute()
            else Path(args.seed_rescore_config_file)
        )
        if not seed_rescore_config_path.exists():
            raise FileNotFoundError(f"Seed-rescore config file not found: {seed_rescore_config_path}")

        seed_rescore_math_path = (
            (ROOT / args.seed_rescore_math_results).resolve()
            if (args.seed_rescore_math_results and not Path(args.seed_rescore_math_results).is_absolute())
            else Path(args.seed_rescore_math_results)
            if args.seed_rescore_math_results
            else work_dir / "seed_rescore_math.pkl"
        )
        seed_rescore_eq_path = (
            (ROOT / args.seed_rescore_eq_results).resolve()
            if (args.seed_rescore_eq_results and not Path(args.seed_rescore_eq_results).is_absolute())
            else Path(args.seed_rescore_eq_results)
            if args.seed_rescore_eq_results
            else work_dir / "seed_rescore_eq.pkl"
        )

        seed_specs: list[tuple[tuple[int, int], ...]] | None = None
        if args.seed_rescore_reuse_existing and seed_rescore_math_path.exists() and seed_rescore_eq_path.exists():
            # Reuse only when existing seed-rescore outputs fully cover the requested config set.
            # This avoids silently accepting partial/interrupted seed passes.
            seed_specs = load_block_specs(seed_rescore_config_path)
            seed_math_layer_scores = load_layer_score_map(seed_rescore_math_path)
            seed_eq_layer_scores = load_layer_score_map(seed_rescore_eq_path)
            expected_keys = {
                tuple(blocks_to_layer_key(args.num_layers, spec))
                for spec in seed_specs
            }
            common_existing = set(seed_math_layer_scores) & set(seed_eq_layer_scores)
            missing_existing = expected_keys - common_existing
            needs_seed_run = bool(missing_existing)
            if needs_seed_run:
                print(
                    "Seed-rescore reuse disabled: partial coverage detected "
                    f"(expected={len(expected_keys)} common_existing={len(common_existing)} "
                    f"missing={len(missing_existing)})."
                )
        else:
            needs_seed_run = True

        if needs_seed_run:
            print(f"Running seed-rescore pass from {seed_rescore_config_path}")
            math_cmd = build_math_worker_cmd(
                args=args,
                config_file=seed_rescore_config_path,
                queue_file=None,
                results_file=seed_rescore_math_path,
                depth=0,
            )
            eq_cmd = build_eq_worker_cmd(
                args=args,
                config_file=seed_rescore_config_path,
                queue_file=None,
                results_file=seed_rescore_eq_path,
                depth=0,
            )
            run_workers(
                runs=[
                    (math_cmd, work_dir / "beam_math_worker.log"),
                    (eq_cmd, work_dir / "beam_eq_worker.log"),
                ],
                cwd=ROOT,
                dry_run=args.dry_run,
                sequential=shared_device,
            )
        else:
            print(
                "Seed-rescore reuse enabled; using existing files: "
                f"{seed_rescore_math_path} and {seed_rescore_eq_path}"
            )

        if not args.dry_run:
            if seed_specs is None:
                seed_specs = load_block_specs(seed_rescore_config_path)
            seed_math_layer_scores = load_layer_score_map(seed_rescore_math_path)
            seed_eq_layer_scores = load_layer_score_map(seed_rescore_eq_path)
            evaluated, ranked_single, stats = build_seed_entries_from_rescored_specs(
                num_layers=args.num_layers,
                block_specs=seed_specs,
                math_layer_scores=seed_math_layer_scores,
                eq_layer_scores=seed_eq_layer_scores,
                require_baseline=args.seed_rescore_require_baseline,
            )
            print(
                f"Seed bootstrap from rescore: specs={len(seed_specs)} "
                f"usable={len(evaluated)} singles={len(ranked_single)}"
            )
        else:
            evaluated, ranked_single, stats = build_seed_entries(
                num_layers=args.num_layers,
                seed_math=seed_math,
                seed_eq=seed_eq,
            )
    else:
        evaluated, ranked_single, stats = build_seed_entries(
            num_layers=args.num_layers,
            seed_math=seed_math,
            seed_eq=seed_eq,
        )

    print(
        "Seed stats: "
        f"baseline_math={stats.baseline_math:.4f}, baseline_eq={stats.baseline_eq:.4f}, "
        f"math_delta_mean={stats.math_delta_mean:.6f}, math_delta_std={stats.math_delta_std:.6f}, "
        f"eq_delta_mean={stats.eq_delta_mean:.6f}, eq_delta_std={stats.eq_delta_std:.6f}"
    )

    evaluated_path = work_dir / "beam_evaluated.pkl"
    math_beam_results_path = work_dir / "beam_math_results.pkl"
    eq_beam_results_path = work_dir / "beam_eq_results.pkl"
    summary_path = work_dir / "beam_summary.json"
    registry_path = work_dir / "tried_registry.json"
    registry = load_registry(registry_path)

    if evaluated_path.exists():
        with evaluated_path.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            # Merge/refresh from loaded state; keep seed entries if missing.
            for key, entry in loaded.items():
                try:
                    layer_key = tuple(int(x) for x in key)
                except Exception:
                    continue
                evaluated[layer_key] = entry
            # Re-score loaded entries with current seed stats for consistency.
            for entry in evaluated.values():
                m = float(entry["math_score"])
                q = float(entry["eq_score"])
                method_score, d_m, d_q = score_method_b(m, q, stats)
                entry["math_delta"] = d_m
                entry["eq_delta"] = d_q
                entry["method_score"] = method_score
        print(f"Resumed evaluated cache: {len(evaluated)} entries from {evaluated_path}")
    else:
        with evaluated_path.open("wb") as f:
            pickle.dump(evaluated, f)
        print(f"Initialized evaluated cache with seed entries: {len(evaluated)}")

    for entry in evaluated.values():
        apply_efficiency_fields(
            entry,
            num_layers=args.num_layers,
            penalty_lambda=args.overhead_penalty_lambda,
        )

    # Ensure registry has a complete view of already known/evaluated configs.
    for entry in evaluated.values():
        blocks_raw = entry.get("blocks", ())
        blocks: tuple[tuple[int, int], ...] = tuple(
            (int(b[0]), int(b[1])) for b in blocks_raw
        )
        layer_key = tuple(int(x) for x in entry.get("layer_key", ()))
        if not layer_key:
            layer_key = blocks_to_layer_key(args.num_layers, blocks)
            entry["layer_key"] = layer_key
        block_spec = str(entry.get("block_spec") or blocks_to_spec(blocks))
        entry["block_spec"] = block_spec
        upsert_registry(
            registry,
            block_spec=block_spec,
            layer_key=layer_key,
            depth=int(entry.get("depth", len(blocks))),
            status="evaluated",
            source=str(entry.get("source", "seed")),
            method_score=float(entry["method_score"]),
            math_score=float(entry["math_score"]),
            eq_score=float(entry["eq_score"]),
            bump_counts=False,
        )
    save_json(registry_path, registry)

    pool_entries = ranked_single[: min(args.pool_size, len(ranked_single))]
    pool_blocks = [tuple(int(x) for x in e["blocks"][0]) for e in pool_entries]
    block_seed_score = {tuple(int(x) for x in e["blocks"][0]): float(e["method_score"]) for e in pool_entries}

    if not pool_blocks:
        raise RuntimeError("No single-block pool entries found from seed results.")

    depth1_frontier = pool_entries[: min(args.seed_top_k, len(pool_entries))]
    save_json(
        work_dir / "depth_1_frontier.json",
        [serialize_entry(e) for e in depth1_frontier],
    )
    print(f"Depth 1 frontier size: {len(depth1_frontier)}")

    global_best = max(
        evaluated.values(),
        key=lambda e: (e["method_score"], e["eq_delta"], e["math_delta"]),
    )
    global_best_score = float(global_best["method_score"])
    global_best_spec = str(global_best["block_spec"])
    plateau_low_improve_streak = 0
    plateau_no_replace_streak = 0
    completed_depths: list[int] = []
    stop_reason = "max_depth_reached"
    beam_started_at = time.time()

    # Beam expansion for depth >= 2
    for depth in range(args.start_depth, args.max_depth + 1):
        prev_frontier_path = work_dir / f"depth_{depth - 1}_frontier.json"
        if prev_frontier_path.exists():
            prev_frontier_raw = json.loads(prev_frontier_path.read_text())
            prev_frontier = []
            for row in prev_frontier_raw:
                blocks = tuple((int(b[0]), int(b[1])) for b in row["blocks"])
                layer_key = tuple(int(x) for x in row["layer_key"])
                prev_frontier.append(
                    {
                        "blocks": blocks,
                        "layer_key": layer_key,
                        "method_score": float(row["method_score"]),
                    }
                )
        else:
            prev_entries = [
                e for e in evaluated.values() if int(e.get("depth", -1)) == depth - 1
            ]
            prev_entries.sort(key=lambda e: (e["method_score"], e["eq_delta"], e["math_delta"]), reverse=True)
            prev_frontier = [
                {
                    "blocks": tuple((int(b[0]), int(b[1])) for b in e["blocks"]),
                    "layer_key": tuple(int(x) for x in e["layer_key"]),
                    "method_score": float(e["method_score"]),
                }
                for e in prev_entries[: args.beam_width]
            ]

        if not prev_frontier:
            print(f"Depth {depth}: no frontier from depth {depth - 1}; stopping.")
            break

        pool_for_expansion = pool_blocks[: min(args.expand_per_node, len(pool_blocks))]

        proposals: list[dict[str, Any]] = []
        for parent in prev_frontier:
            parent_blocks = parent["blocks"]
            parent_score = float(parent["method_score"])
            for block in pool_for_expansion:
                cand_blocks = parent_blocks + (block,)
                cand_layer_key = blocks_to_layer_key(args.num_layers, cand_blocks)

                extra_layers = len(cand_layer_key) - args.num_layers
                if args.max_extra_layers is not None and extra_layers > args.max_extra_layers:
                    continue

                heuristic = parent_score + block_seed_score.get(block, 0.0)
                proposals.append(
                    {
                        "blocks": cand_blocks,
                        "layer_key": cand_layer_key,
                        "heuristic": heuristic,
                    }
                )

        # Highest-heuristic first, then dedupe by layer key and skip evaluated.
        proposals.sort(key=lambda p: p["heuristic"], reverse=True)
        candidates: list[dict[str, Any]] = []
        seen_layer_keys: set[tuple[int, ...]] = set()

        for p in proposals:
            lk = p["layer_key"]
            if lk in seen_layer_keys:
                continue
            if lk in evaluated:
                continue
            seen_layer_keys.add(lk)
            candidates.append(p)
            if len(candidates) >= args.max_candidates_per_depth:
                break

        print(
            f"Depth {depth}: frontier={len(prev_frontier)}, proposals={len(proposals)}, "
            f"new_candidates={len(candidates)}"
        )

        if not candidates:
            existing_depth_frontier = work_dir / f"depth_{depth}_frontier.json"
            if existing_depth_frontier.exists():
                print(
                    f"Depth {depth}: no new candidates but frontier file exists; "
                    "treating as completed depth and continuing."
                )
                completed_depths.append(depth)
                continue
            print(f"Depth {depth}: nothing new to evaluate; stopping.")
            stop_reason = f"candidate_exhausted_depth_{depth}"
            break

        candidate_file = work_dir / f"depth_{depth}_candidates.txt"
        with candidate_file.open("w") as f:
            for c in candidates:
                f.write(blocks_to_spec(c["blocks"]) + "\n")
                upsert_registry(
                    registry,
                    block_spec=blocks_to_spec(c["blocks"]),
                    layer_key=c["layer_key"],
                    depth=depth,
                    status="planned",
                    source="beam_candidate",
                    method_score=float(c["heuristic"]),
                )
        save_json(registry_path, registry)

        if args.dry_run:
            planned = sorted(candidates, key=lambda c: c["heuristic"], reverse=True)
            save_json(
                work_dir / f"depth_{depth}_planned_candidates.json",
                [
                    {
                        "block_spec": blocks_to_spec(c["blocks"]),
                        "blocks": [list(b) for b in c["blocks"]],
                        "layer_key": list(c["layer_key"]),
                        "heuristic": c["heuristic"],
                    }
                    for c in planned
                ],
            )
            depth_frontier = planned[: args.beam_width]
            save_json(
                work_dir / f"depth_{depth}_frontier.json",
                [
                    {
                        "blocks": [list(b) for b in c["blocks"]],
                        "layer_key": list(c["layer_key"]),
                        "method_score": c["heuristic"],
                    }
                    for c in depth_frontier
                ],
            )
            print(f"Depth {depth}: dry-run planned frontier size={len(depth_frontier)}")
            continue

        # Resume-aware missing-only queue creation for this depth.
        existing_math_scores = load_layer_score_map(math_beam_results_path)
        existing_eq_scores = load_layer_score_map(eq_beam_results_path)

        math_queue_entries: list[dict[str, Any]] = []
        eq_queue_entries: list[dict[str, Any]] = []
        for idx, c in enumerate(candidates):
            entry = {
                "idx": idx,
                "spec": blocks_to_spec(c["blocks"]),
                "layers": list(c["layer_key"]),
            }
            if c["layer_key"] not in existing_math_scores:
                math_queue_entries.append(entry)
            if c["layer_key"] not in existing_eq_scores:
                eq_queue_entries.append(entry)

        math_queue_file = work_dir / f"depth_{depth}_math_queue.json"
        eq_queue_file = work_dir / f"depth_{depth}_eq_queue.json"
        write_queue_file(math_queue_file, math_queue_entries)
        write_queue_file(eq_queue_file, eq_queue_entries)

        print(
            f"Depth {depth} queue prep: math_missing={len(math_queue_entries)} "
            f"eq_missing={len(eq_queue_entries)}"
        )

        if args.dynamic_split:
            run_depth_workers_dynamic(
                args=args,
                depth=depth,
                work_dir=work_dir,
                math_queue_file=math_queue_file,
                eq_queue_file=eq_queue_file,
                math_results_file=math_beam_results_path,
                eq_results_file=eq_beam_results_path,
                dry_run=args.dry_run,
            )
        else:
            # Compatibility path: one worker per metric in queue mode.
            runs: list[tuple[list[str], Path]] = []
            if len(math_queue_entries) > 0:
                math_cmd = build_math_worker_cmd(
                    args=args,
                    config_file=None,
                    queue_file=math_queue_file,
                    results_file=math_beam_results_path,
                    depth=depth,
                )
                runs.append((math_cmd, work_dir / "beam_math_worker.log"))
            if len(eq_queue_entries) > 0:
                eq_cmd = build_eq_worker_cmd(
                    args=args,
                    config_file=None,
                    queue_file=eq_queue_file,
                    results_file=eq_beam_results_path,
                    depth=depth,
                )
                runs.append((eq_cmd, work_dir / "beam_eq_worker.log"))
            if runs:
                run_workers(
                    runs=runs,
                    cwd=ROOT,
                    dry_run=args.dry_run,
                    sequential=shared_device,
                )

        math_layer_scores = load_layer_score_map(math_beam_results_path)
        eq_layer_scores = load_layer_score_map(eq_beam_results_path)

        newly_added: list[dict[str, Any]] = []
        for c in candidates:
            lk = c["layer_key"]
            cand_spec = blocks_to_spec(c["blocks"])
            if lk not in math_layer_scores or lk not in eq_layer_scores:
                print(f"WARNING: Missing worker scores for candidate {cand_spec}")
                upsert_registry(
                    registry,
                    block_spec=cand_spec,
                    layer_key=lk,
                    depth=depth,
                    status="missing_scores",
                    source="beam_eval",
                    method_score=float(c["heuristic"]),
                )
                continue

            m = float(math_layer_scores[lk])
            q = float(eq_layer_scores[lk])
            method_score, d_m, d_q = score_method_b(m, q, stats)
            entry = {
                "blocks": c["blocks"],
                "block_spec": cand_spec,
                "layer_key": lk,
                "depth": depth,
                "math_score": m,
                "eq_score": q,
                "math_delta": d_m,
                "eq_delta": d_q,
                "method_score": method_score,
                "source": "beam_eval",
            }
            apply_efficiency_fields(
                entry,
                num_layers=args.num_layers,
                penalty_lambda=args.overhead_penalty_lambda,
            )
            evaluated[lk] = entry
            newly_added.append(entry)
            upsert_registry(
                registry,
                block_spec=cand_spec,
                layer_key=lk,
                depth=depth,
                status="evaluated",
                source="beam_eval",
                method_score=method_score,
                math_score=m,
                eq_score=q,
            )

        with evaluated_path.open("wb") as f:
            pickle.dump(evaluated, f)
        save_json(registry_path, registry)

        depth_entries = [
            e for e in evaluated.values() if int(e.get("depth", -1)) == depth
        ]
        depth_entries.sort(
            key=lambda e: (e["method_score"], e["eq_delta"], e["math_delta"]),
            reverse=True,
        )
        depth_frontier = depth_entries[: args.beam_width]
        save_json(
            work_dir / f"depth_{depth}_frontier.json",
            [serialize_entry(e) for e in depth_frontier],
        )
        save_json(
            work_dir / f"depth_{depth}_newly_added.json",
            [serialize_entry(e) for e in newly_added],
        )

        if depth_frontier:
            best = depth_frontier[0]
            print(
                f"Depth {depth} best: spec={best['block_spec']} "
                f"method={best['method_score']:.4f} "
                f"math={best['math_score']:.4f} eq={best['eq_score']:.4f}"
            )
        completed_depths.append(depth)

        current_best = max(
            evaluated.values(),
            key=lambda e: (e["method_score"], e["eq_delta"], e["math_delta"]),
        )
        current_best_score = float(current_best["method_score"])
        current_best_spec = str(current_best["block_spec"])
        improvement = current_best_score - global_best_score
        replaced = (current_best_spec != global_best_spec) and (improvement > 1e-12)

        if args.plateau_min_improvement > 0:
            if improvement < args.plateau_min_improvement:
                plateau_low_improve_streak += 1
            else:
                plateau_low_improve_streak = 0
        else:
            plateau_low_improve_streak = 0

        if replaced:
            plateau_no_replace_streak = 0
        else:
            plateau_no_replace_streak += 1

        global_best_score = max(global_best_score, current_best_score)
        if replaced:
            global_best_spec = current_best_spec

        elapsed_hours = (time.time() - beam_started_at) / 3600.0
        if args.max_beam_hours is not None and elapsed_hours >= args.max_beam_hours:
            stop_reason = f"max_beam_hours_reached_{args.max_beam_hours:.2f}h"
            print(f"Stopping after depth {depth}: {stop_reason}")
            break

        plateau_enabled = (
            args.plateau_min_improvement > 0
            and args.plateau_streak > 0
            and args.plateau_no_replace_streak > 0
        )
        if (
            plateau_enabled
            and elapsed_hours >= args.min_beam_hours
            and plateau_low_improve_streak >= args.plateau_streak
            and plateau_no_replace_streak >= args.plateau_no_replace_streak
        ):
            stop_reason = (
                "plateau_stop:"
                f"low_improve={plateau_low_improve_streak}/"
                f"{args.plateau_streak},"
                f"no_replace={plateau_no_replace_streak}/"
                f"{args.plateau_no_replace_streak}"
            )
            print(f"Stopping after depth {depth}: {stop_reason}")
            break

    # Final overall summary
    use_final_ranking = args.overhead_penalty_lambda > 0
    all_entries = sorted(
        evaluated.values(),
        key=lambda e: rank_key(e, use_final_score=use_final_ranking),
        reverse=True,
    )
    top_overall = all_entries[:50]
    save_json(work_dir / "top_overall.json", [serialize_entry(e) for e in top_overall])

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_path": args.model_path,
        "num_layers": args.num_layers,
        "seed_math_results": str(seed_math_path),
        "seed_eq_results": str(seed_eq_path),
        "seed_rescore_config_file": str(seed_rescore_config_path) if seed_rescore_config_path else None,
        "seed_rescore_math_results": str(seed_rescore_math_path) if seed_rescore_math_path else None,
        "seed_rescore_eq_results": str(seed_rescore_eq_path) if seed_rescore_eq_path else None,
        "work_dir": str(work_dir),
        "beam_width": args.beam_width,
        "start_depth": args.start_depth,
        "max_depth": args.max_depth,
        "seed_top_k": args.seed_top_k,
        "pool_size": args.pool_size,
        "expand_per_node": args.expand_per_node,
        "max_candidates_per_depth": args.max_candidates_per_depth,
        "max_extra_layers": args.max_extra_layers,
        "min_beam_hours": args.min_beam_hours,
        "max_beam_hours": args.max_beam_hours,
        "plateau_min_improvement": args.plateau_min_improvement,
        "plateau_streak": args.plateau_streak,
        "plateau_no_replace_streak": args.plateau_no_replace_streak,
        "overhead_penalty_lambda": args.overhead_penalty_lambda,
        "final_ranking_enabled": use_final_ranking,
        "stop_reason": stop_reason,
        "completed_depths": completed_depths,
        "evaluated_total": len(evaluated),
        "tried_registry": str(registry_path),
        "tried_total": len(registry),
        "tried_status_counts": {
            status: sum(1 for rec in registry.values() if rec.get("last_status") == status)
            for status in sorted({str(rec.get("last_status", "unknown")) for rec in registry.values()})
        },
        "top_overall": [serialize_entry(e) for e in top_overall[:10]],
        "z_stats": {
            "baseline_math": stats.baseline_math,
            "baseline_eq": stats.baseline_eq,
            "math_delta_mean": stats.math_delta_mean,
            "math_delta_std": stats.math_delta_std,
            "eq_delta_mean": stats.eq_delta_mean,
            "eq_delta_std": stats.eq_delta_std,
        },
    }
    save_json(summary_path, summary)

    print(f"Wrote summary: {summary_path}")
    if top_overall:
        best = top_overall[0]
        print(
            f"Best overall: spec={best['block_spec']} method={best['method_score']:.4f} "
            f"final={best['final_score']:.4f} math={best['math_score']:.4f} "
            f"eq={best['eq_score']:.4f} rel_overhead={best['relative_overhead']:.4f}"
        )


if __name__ == "__main__":
    main()
