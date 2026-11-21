#!/usr/bin/env python3
"""Compare paired mean absolute UU norm changes from the existing primary18 run."""

import argparse
from collections import Counter
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gcg-norm-change")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.artifact_io import read_json, read_jsonl, sha256_file, utc_now
from utils.gcg_shuffled_common import DATASET, SCORES, region_layout, write_csv, write_json

PHASE = ROOT / "data/outputs/gcg_vs_shuffled/legacy_selection20"
FOUR = PHASE / "10_existing_attention/four_metrics"
REGION = PHASE / "20_region_scores/rms_and_frobenius"
CONDITIONS = ["gcg", "shuffled"]
KEYS = ["dataset_index", "condition", "layer", "head"]
NORMS = ["F", "F_sym", "F_asym"]
TOL = 1e-12


def verify_inputs():
    hashes = {}

    def check(path, expected):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        hashes[str(path)] = actual

    for directory in (FOUR, REGION):
        ledger = directory / "output_hashes.json"
        hashes[str(ledger)] = sha256_file(ledger)
        for name in ("manifest.json", "metrics.csv"):
            check(directory / name, read_json(ledger)[name])
    four = read_json(FOUR / "manifest.json")
    region = read_json(REGION / "manifest.json")
    plan = read_json(SCORES / "plan.json")
    for path in (DATASET, SCORES / "plan.json", SCORES / "manifest.jsonl"):
        check(path, four["source_hashes"][str(path)])
    check(DATASET, plan["dataset_sha256"])
    if (plan["model_name"], plan["score_dtype"]) != ("meta-llama/Llama-2-7b-chat-hf", "float32"):
        raise ValueError("unexpected Score source model or dtype")

    dataset = {row["dataset_index"]: row["conditions"] for row in read_jsonl(DATASET)}
    primary = sorted(i for i, c in dataset.items()
                     if [c[k]["evaluation"]["success"] for k in ("original", *CONDITIONS)]
                     == [False, True, False])
    if len(primary) != 18 or primary != four["primary_indices"] or primary != region["primary_indices"]:
        raise ValueError("primary18 membership mismatch")
    grid = pd.MultiIndex.from_product([primary, CONDITIONS, range(32), range(32)], names=KEYS)
    frames = []
    for directory, flag in ((FOUR, "primary_18"), (REGION, "primary18")):
        frame = pd.read_csv(directory / "metrics.csv", float_precision="round_trip")
        if not frame[flag].equals(frame.dataset_index.isin(primary)):
            raise ValueError(f"primary flags mismatch: {directory}")
        frame = frame.loc[frame[flag]].set_index(KEYS, verify_integrity=True).sort_index()
        if not frame.index.equals(grid):
            raise ValueError(f"incomplete primary18 grid: {directory}")
        frames.append(frame)
    metrics, regions = frames
    np.testing.assert_allclose(metrics.F, regions.F_UU_full, rtol=TOL, atol=TOL)
    if four["score_references"] != region["score_references"]:
        raise ValueError("source Score references disagree")

    refs = {(r["dataset_index"], r["condition"]): r for r in four["score_references"]}
    source_keys = {(r["dataset_index"], r["condition"]): r
                   for r in read_jsonl(SCORES / "manifest.jsonl")}
    max_error = 0.0
    positions = {}
    for i in primary:
        g, s = (dataset[i][c] for c in CONDITIONS)
        u0, u1 = g["token_ranges"]["suffix"]
        if u1-u0 != 20 or g["token_ranges"] != s["token_ranges"]:
            raise ValueError(f"suffix position mismatch: {i}")
        gi, si = g["input_ids"], s["input_ids"]
        if gi[:u0] != si[:u0] or gi[u1:] != si[u1:] or Counter(gi[u0:u1]) != Counter(si[u0:u1]):
            raise ValueError(f"paired tokens mismatch: {i}")
        positions[i] = [u0, u1]
        for c in CONDITIONS:
            ref = refs[i, c]
            score = Path(ref["path"])
            metadata_path = Path(ref["metadata_path"])
            check(metadata_path, ref["metadata_sha256"])
            metadata = read_json(metadata_path)
            check(score, ref["sha256"])
            if (score != SCORES / source_keys[i, c]["score_file"]
                    or metadata["score_sha256"] != ref["sha256"]
                    or metadata["plan_sha256"] != hashes[str(SCORES / "plan.json")]
                    or metadata["input"] != ref["input"]):
                raise ValueError(f"Score provenance mismatch: {i}/{c}")
            item = metadata["input"]
            extraction = metadata["extraction"]
            if (item["dataset_index"], item["condition"]) != (i, c) or any(
                item[k] != dataset[i][c][k] for k in ("input_ids", "token_ranges")
            ) or (extraction["causal_mask_applied_to_saved_scores"]
                  or extraction["softmax_applied"]
                  or not extraction["future_direction_entries_preserved"]
                  or not extraction["computation"].startswith("post-RoPE")):
                raise ValueError(f"Score conditions mismatch: {i}/{c}")
            values = np.load(score, mmap_mode="r", allow_pickle=False)
            if values.dtype != np.float32 or values.shape != (32, 32, len(gi), len(gi)):
                raise ValueError(f"Score shape/dtype mismatch: {i}/{c}")
            block = region_layout(values, item["token_ranges"])[0]["UU"]
            transpose = block.swapaxes(-1, -2)
            direct = np.stack([np.linalg.norm(x, axis=(-2, -1)).ravel()
                               for x in (block, (block+transpose)/2, (block-transpose)/2)], axis=1)
            saved = metrics.loc[(i, c), NORMS].to_numpy()
            np.testing.assert_allclose(saved, direct, atol=TOL, rtol=TOL)
            max_error = max(max_error, float(np.max(np.abs(saved-direct))))
    return metrics, primary, positions, hashes, plan, four, region, max_error


def compare(metrics):
    norms = metrics[NORMS].to_numpy().reshape(18, 2, 1024, 3)
    if not np.isfinite(norms).all() or np.any(norms < 0) or np.any(norms[..., 0] <= 0):
        raise ValueError("nonfinite, negative or undefined Norm ratio")
    ratios = norms[..., 1:] / norms[..., :1]
    np.testing.assert_allclose((ratios**2).sum(axis=-1), 1, atol=TOL, rtol=0)
    np.testing.assert_allclose(ratios[..., 1].ravel()**2, metrics.R_asym, atol=TOL, rtol=0)
    delta = ratios[:, 0] - ratios[:, 1]
    absolute = np.abs(delta)
    sums = ratios.sum(axis=1)
    gap = absolute[..., 1] - absolute[..., 0]
    # Since |ΔR_asym| sum_asym = |ΔR_sym| sum_sym, avoid dividing by small changes.
    residual = absolute[..., 1]*sums[..., 1] - absolute[..., 0]*sums[..., 0]
    predicted = absolute[..., 1]*(sums[..., 0]-sums[..., 1])/sums[..., 0]
    np.testing.assert_allclose(gap, predicted, atol=TOL, rtol=0)
    np.testing.assert_allclose(residual, 0, atol=TOL, rtol=0)
    compare_sign = lambda x: np.sign(np.where(np.abs(x) <= TOL, 0, x))
    mismatch = int(np.count_nonzero(compare_sign(gap) != compare_sign(predicted)))
    if mismatch:
        raise ValueError("observed and predicted comparisons disagree")

    layer, head = np.indices((32, 32))
    heads = pd.DataFrame({"layer": layer.ravel(), "head": head.ravel(), "n_samples": 18})
    for j, component in enumerate(("sym", "asym")):
        for k, condition in enumerate(CONDITIONS):
            heads[f"mean_R_{component}_{condition}"] = ratios[:, k, :, j].mean(axis=0)
    heads["mean_R_sym_minus_R_asym_midpoint"] = (sums[..., 0]-sums[..., 1]).mean(axis=0)/2

    summaries = {}
    for name, changes, prefix in (("ratio", absolute, "A"), ("raw_norm", np.abs(
            norms[:, 0, :, 1:] - norms[:, 1, :, 1:]), "A_raw")):
        avg = changes.mean(axis=0)
        diff = avg[:, 1]-avg[:, 0]
        wins = (changes[..., 1] > changes[..., 0]).sum(axis=0)
        heads[f"{prefix}_sym"] = avg[:, 0]
        heads[f"{prefix}_asym"] = avg[:, 1]
        heads[f"{prefix}_asym_minus_sym"] = diff
        heads[f"{name}_asym_gt_sym_sample_count"] = wins
        summaries[name] = dict(
            mean_A_sym=float(avg[:, 0].mean()), mean_A_asym=float(avg[:, 1].mean()),
            mean_gap=float(diff.mean()), heads_asym_gt_sym=int((diff > 0).sum()),
            fraction_heads_asym_gt_sym=float((diff > 0).mean()),
            heads_asym_lt_sym=int((diff < 0).sum()), heads_equal=int((diff == 0).sum()),
            pair_asym_gt_sym=int(wins.sum()),
            pair_asym_lt_sym=int((changes[..., 1] < changes[..., 0]).sum()),
            pair_equal=int((changes[..., 1] == changes[..., 0]).sum()),
            heads_by_sample_count=np.bincount(wins, minlength=19).tolist(),
        )
    summaries["ratio"].update(
        mean_R_by_condition=dict(zip(CONDITIONS, ratios.mean(axis=(0, 2)).tolist())),
        pearson_head_mean_ratio_gap_vs_change_gap=float(heads[
            "mean_R_sym_minus_R_asym_midpoint"].corr(heads["A_asym_minus_sym"])),
        pairs_sum_sym_gt_asym=int((sums[..., 0] > sums[..., 1]).sum()),
        pairs_sum_sym_lt_asym=int((sums[..., 0] < sums[..., 1]).sum()),
        gap_range=[float(gap.min()), float(gap.max())],
    )
    pairs = metrics.index.droplevel("condition").unique().to_frame(index=False)
    for j, component in enumerate(("sym", "asym")):
        for k, condition in enumerate(CONDITIONS):
            pairs[f"R_{component}_{condition}"] = ratios[:, k, :, j].ravel()
        pairs[f"abs_delta_R_{component}"] = absolute[..., j].ravel()
    pairs["predicted_abs_delta_gap"] = predicted.ravel()
    validation = dict(
        ratio_identity_max_error=float(np.max(np.abs((ratios**2).sum(axis=-1)-1))),
        squared_ratio_difference_max=float(np.max(np.abs(ratios[..., 1].ravel()**2-metrics.R_asym))),
        weighted_change_identity_max_error=float(np.max(np.abs(residual))),
        predicted_gap_max_error=float(np.max(np.abs(gap-predicted))),
        strict_comparison_mismatches=int(np.count_nonzero(np.sign(gap) != np.sign(predicted))),
        comparison_mismatches_at_1e_12=mismatch,
        pairs_near_tie_at_1e_12=int((np.abs(gap) <= TOL).sum()),
    )
    return heads, pairs, summaries, validation


def save_figures(directory, heads):
    grid = heads.A_asym_minus_sym.to_numpy().reshape(32, 32)
    limit = np.max(np.abs(grid))
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    plot = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set(xlabel="Head", ylabel="Layer", title="Mean absolute Norm-ratio change | primary18")
    fig.colorbar(plot, ax=ax, label="A_asym − A_sym")
    fig.savefig(directory / "heatmap.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.scatter(heads.mean_R_sym_minus_R_asym_midpoint, heads.A_asym_minus_sym,
               s=10, alpha=.45, edgecolors="none")
    ax.axhline(0, color="0.4", linewidth=.8)
    ax.axvline(0, color="0.4", linewidth=.8)
    ax.set(xlabel="Mean R_sym − R_asym (both conditions)", ylabel="A_asym − A_sym",
           title="Head-level ratio balance and change gap | primary18")
    fig.savefig(directory / "ratio_balance.png", dpi=200)
    plt.close(fig)


def report(summaries, validation, primary):
    ratio, raw = (summaries[k] for k in ("ratio", "raw_norm"))
    g, s = (ratio["mean_R_by_condition"][k] for k in CONDITIONS)
    rows = []
    for label, item in (("Norm比", ratio), ("生のFrobenius Norm", raw)):
        rows.append(
            f"| {label} | {item['mean_A_sym']:.8f} | {item['mean_A_asym']:.8f} | "
            f"{item['mean_gap']:+.8f} | {item['heads_asym_gt_sym']:,}/1,024 "
            f"({item['fraction_heads_asym_gt_sym']:.2%}) | {item['pair_asym_gt_sym']:,}/18,432 |"
        )
    return f"""# GCG vs ShuffledのUU対称・反対称成分の変化量

既存primary18のUUでは、**{ratio['heads_asym_gt_sym']:,}/1,024 Head
（{ratio['fraction_heads_asym_gt_sym']:.2%}）で反対称成分のNorm比の平均絶対変化量が大きい**。
生Normでは **{raw['heads_asym_gt_sym']:,}/1,024 Head
（{raw['fraction_heads_asym_gt_sym']:.2%}）**。

| 指標 | 全Head平均 A_sym | 全Head平均 A_asym | 平均差 A_asym−A_sym | A_asym > A_sym | ペアで反対称の変化が大きい件数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 条件と集計

Llama-2-7b-chat-hf。既存primary18のindexは {primary}。
既存の自動判定でOriginal失敗・GCG成功・Shuffled失敗に該当した探索用の18件。
RoPE後・Mask前・Softmax前のFull Scoreから同じSuffix位置の20×20 UUを使い、
全32 Layer×32 Headを解析した。対角と未来方向の要素も含む。
保存済みの生Norm F, F_sym, F_asymから R_sym = F_sym/F、R_asym = F_asym/F をFP64で計算した。
既存CSVの `R_asym` 列は二乗比であり、今回のR_asymの二乗と照合した。
各HeadのAは、サンプルごとにGCG−Shuffledの差の絶対値を取り、18件で平均した値。
平均差の絶対値とは異なる。件数は厳密な > で集計し、検証には絶対許容差1e-12を使用した。
生Normの行は F_sym と F_asym に同じ差分・絶対値・平均を適用した。

## 観測と恒等式

- 18,432ペアのうち反対称比率の絶対変化が大きいのは {ratio['pair_asym_gt_sym']:,} 件、
  小さいのは {ratio['pair_asym_lt_sym']:,} 件、等しいのは {ratio['pair_equal']:,} 件。
  全18件で反対称の変化が大きいHeadは {ratio['heads_by_sample_count'][18]:,}。
- 平均 (R_sym, R_asym) はGCG ({g[0]:.6f}, {g[1]:.6f})、Shuffled ({s[0]:.6f}, {s[1]:.6f})。
  両条件のR_symの和がR_asymの和を上回るのは {ratio['pairs_sum_sym_gt_asym']:,}/18,432ペア。
  各Headの両条件・18件平均の比率差と A_asym−A_sym のPearson相関は
  {ratio['pearson_head_mean_ratio_gap_vs_change_gap']:+.4f}（記述値）。
- Frobenius内積での直交性から R_sym² + R_asym² = 1。各成分のGCGとShuffledでの
  比率の和を Σ_sym, Σ_asym とすると
  `|ΔR_sym| Σ_sym = |ΔR_asym| Σ_asym`。
  変化があるペアでは、両条件での比率の和が小さい成分ほど絶対変化が大きい。
  Head平均の差はペアごとの大小と変化の大きさを両方反映する。
- 生Normには全体Normの条件間変化も入るため、上記の大小関係はそのまま適用できない。
  6月のOriginal vs AutoDANに似たNorm比の傾向は確認できるが、この傾向だけから
  Jailbreak特有の機構・因果効果は判断できない。データ・窓・集計単位も異なる。

## 数値検証と出典

- R_sym² + R_asym² の最大絶対誤差 {validation['ratio_identity_max_error']:.3g}。
  恒等式による大小の予測と実測の不一致は厳密比較で {validation['strict_comparison_mismatches']} 件、
  許容差1e-12で {validation['comparison_mismatches_at_1e_12']} 件。
- primary18、Suffixの半開区間、完全なLayer/Head対応を既存manifest・CSVと照合した。
  保存Scoreから36入力のUU生Normを検算した最大絶対差は
  {validation['saved_vs_raw_norm_max_error']:.3g}。ゼロ分母、欠損、非有限値は0件。
- 数値入力は既存four_metricsのmetrics.csv。既存Region CSVの F_UU_full および
  保存Scoreで検算した。入力パス・ハッシュ、取得条件、元Commit、分析コードのハッシュは
  [manifest.json](manifest.json)、数値検証は [validation.json](validation.json)。
- [Head別CSV](heads.csv)・[ペア別比率と大小予測](pairs.csv)・[全体集計](summary.json)・
  [32×32ヒートマップ](heatmap.png)・[元の比率差との対応](ratio_balance.png)。
"""


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    directory = parser.parse_args().output_dir
    if directory.exists():
        raise FileExistsError(directory)
    metrics, primary, positions, hashes, plan, four, region, max_error = verify_inputs()
    heads, pairs, summaries, validation = compare(metrics)
    if any(sha256_file(path) != expected for path, expected in hashes.items()):
        raise ValueError("input changed during analysis")
    validation.update(status="passed", saved_vs_raw_norm_max_error=max_error,
                      condition_rows=len(metrics), pair_rows=len(pairs), head_rows=len(heads))
    code_paths = [Path(__file__).resolve(), ROOT / "utils/gcg_shuffled_common.py",
                  ROOT / "utils/artifact_io.py"]
    provenance = dict(
        created_at=utc_now(), git_commit=git("rev-parse", "HEAD"), git_branch=git("branch", "--show-current"),
        git_status=git("status", "--short"), source_commits=dict(
            score_extraction=plan["git_commit"], four_metrics=four["git_commit"], region=region["git_commit"]),
        code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in code_paths},
        input_sha256=hashes, model=plan["model"], model_name=plan["model_name"],
        model_revision=plan["model_revision"], tokenizer=plan["tokenizer_info"],
        score_computation=plan["computation"], score_dtype="float32", analysis_dtype="float64",
        device="CPU", model_loaded=False, model_forward=False,
        primary_indices=primary, suffix_ranges_half_open=positions, conditions=CONDITIONS,
        delta="GCG - Shuffled", heads=[32, 32], region="UU full unmasked suffix 20×20",
        aggregation="mean of paired absolute deltas per Head, then equal-weight Head mean",
        ratio="component Frobenius norm / total Frobenius norm", tolerance=TOL,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        matplotlib=matplotlib.__version__, random_seed=None,
    )
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / "heads.csv", heads)
    write_csv(directory / "pairs.csv", pairs)
    write_json(directory / "summary.json", summaries)
    write_json(directory / "validation.json", validation)
    write_json(directory / "manifest.json", provenance)
    save_figures(directory, heads)
    with (directory / "report.md").open("x", encoding="utf-8") as output:
        output.write(report(summaries, validation, primary))
    write_json(directory / "output_hashes.json",
               {p.name: sha256_file(p) for p in sorted(directory.iterdir())})


if __name__ == "__main__":
    main()
