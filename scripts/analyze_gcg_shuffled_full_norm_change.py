#!/usr/bin/env python3
"""CPU analysis of paired signed Norm-ratio changes on saved full prompt scores."""

import argparse
from collections import Counter
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gcg-full-norm-change")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.artifact_io import read_json, sha256_file, utc_now
from utils.gcg_shuffled_common import DATASET, SCORES, inspect_inputs, write_csv, write_json

PAPER_HEADS = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
CONDITIONS = ("gcg", "shuffled")
KEYS = ["dataset_index", "layer", "head"]
COMPONENTS = ("sym", "asym")
TOL = 1e-12
JUNE_COMMIT = "d5be824"


def decompose(score):
    s = np.asarray(score, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError("nonfinite Full Score")
    transpose = s.swapaxes(-1, -2)
    norm = np.linalg.norm(s, axis=(-2, -1))
    if (norm == 0).any():
        raise ValueError("undefined Norm ratio: zero Score norm")
    return np.stack([
        np.linalg.norm((s + transpose)/2, axis=(-2, -1))/norm,
        np.linalg.norm((s - transpose)/2, axis=(-2, -1))/norm,
    ], axis=-1)


def read_inputs(uu_dir):
    _, primary, references = inspect_inputs()
    primary = sorted(primary)
    uu_ledger = read_json(uu_dir/"output_hashes.json")
    source_paths = [DATASET, DATASET.parent/"plan.json", SCORES/"plan.json",
                    SCORES/"manifest.jsonl", SCORES/"validation_final.json",
                    uu_dir/"output_hashes.json"]
    hashes = {str(p): sha256_file(p) for p in source_paths}
    for name in ("manifest.json", "pairs.csv", "summary.json"):
        path = uu_dir/name
        hashes[str(path)] = sha256_file(path)
        if hashes[str(path)] != uu_ledger[name]:
            raise ValueError(f"UU artifact hash mismatch: {path}")
    uu_manifest = read_json(uu_dir/"manifest.json")
    source_plan = read_json(SCORES/"plan.json")
    if (uu_manifest["primary_indices"] != primary
            or source_plan["model_name"] != "meta-llama/Llama-2-7b-chat-hf"
            or source_plan["score_dtype"] != "float32"
            or not source_plan["computation"].startswith("post-RoPE")):
        raise ValueError("unexpected model, Score extraction or primary18")
    refs = {(r["dataset_index"], r["condition"]): r for r in references}
    grid = pd.MultiIndex.from_product([primary, range(32), range(32)], names=KEYS)
    uu = pd.read_csv(uu_dir/"pairs.csv", float_precision="round_trip").set_index(
        KEYS, verify_integrity=True).sort_index()
    if not uu.index.equals(grid):
        raise ValueError("Full and UU sample/Head grids differ")

    lengths, ratios = [], []
    for i in primary:
        pair = [refs[i, c] for c in CONDITIONS]
        sizes = [r["input"]["sequence_length"] for r in pair]
        if sizes[0] != sizes[1]:
            raise ValueError(f"actual sequence lengths differ: {i}, {sizes}")
        lengths.append(sizes[0])
        values = []
        for r in pair:
            path = r["path"]
            hashes[path] = r["sha256"]
            hashes[r["metadata_path"]] = r["metadata_sha256"]
            if any(uu_manifest["input_sha256"].get(r[k]) != r[digest]
                   for k, digest in (("path", "sha256"), ("metadata_path", "metadata_sha256"))):
                raise ValueError(f"Full and UU source references differ: {path}")
            score = np.load(path, mmap_mode="r", allow_pickle=False)
            values.append(decompose(score))
        ratios.append(values)
    return grid, np.asarray(lengths), np.asarray(ratios), uu, hashes, source_plan


def signed_summary(delta):
    avg = delta.mean(axis=0).reshape(1024, 2)
    gap = np.abs(avg[:, 1])-np.abs(avg[:, 0])
    opposite = avg[:, 0]*avg[:, 1] < 0
    clear = (np.abs(avg) > TOL).all(axis=1)
    return dict(
        mean_delta_sym=float(avg[:, 0].mean()), mean_delta_asym=float(avg[:, 1].mean()),
        heads_abs_mean_asym_gt_sym=int((gap > 0).sum()),
        fraction_abs_mean_asym_gt_sym=float((gap > 0).mean()),
        heads_abs_mean_asym_lt_sym=int((gap < 0).sum()), heads_abs_mean_tie=int((gap == 0).sum()),
        heads_opposite_sign=int(opposite.sum()), fraction_opposite_sign=float(opposite.mean()),
        heads_abs_mean_asym_gt_sym_at_tol=int((gap > TOL).sum()),
        heads_opposite_sign_at_tol=int((opposite & clear).sum()),
        heads_abs_gap_near_zero=int((np.abs(gap) <= TOL).sum()),
        heads_any_mean_near_zero=int((~clear).sum()),
    )


def summarize(grid, lengths, ratios, uu):
    delta = ratios[:, 0]-ratios[:, 1]
    mean_delta = delta.mean(axis=0).reshape(1024, 2)
    np.testing.assert_allclose(mean_delta, [
        [mean(column) for column in head] for head in delta.reshape(18, 1024, 2).transpose(1, 2, 0)
    ], atol=TOL, rtol=0)
    error = np.abs(np.sum(ratios**2, axis=-1)-1)
    np.testing.assert_allclose(error, 0, atol=TOL, rtol=0)
    ratio_sums = ratios.sum(axis=1)
    residual = np.abs((delta*ratio_sums).sum(axis=-1))
    np.testing.assert_allclose(residual, 0, atol=TOL, rtol=0)

    pair_rows = grid.to_frame(index=False)
    pair_rows["token_count"] = np.repeat(lengths, 1024)
    head_rows = grid.droplevel("dataset_index").unique().to_frame(index=False)
    head_rows["n_samples"] = len(lengths)
    head_rows["mean_token_count"] = lengths.mean()
    length_centered = lengths-lengths.mean()
    uu_delta = []
    for j, component in enumerate(COMPONENTS):
        y = delta[..., j].reshape(18, 1024)
        for k, condition in enumerate(CONDITIONS):
            pair_rows[f"R_{component}_{condition}"] = ratios[:, k, :, :, j].ravel()
        pair_rows[f"delta_R_{component}"] = y.ravel()
        head_rows[f"mean_delta_R_{component}"] = mean_delta[:, j]
        centered = y-y.mean(axis=0)
        denominator = np.linalg.norm(length_centered)*np.linalg.norm(centered, axis=0)
        head_rows[f"pearson_length_delta_{component}"] = np.divide(
            length_centered @ centered, denominator, out=np.full(1024, np.nan),
            where=denominator > 0)
        uu_delta.append((uu[f"R_{component}_gcg"]-uu[f"R_{component}_shuffled"]).to_numpy())
    head_rows["abs_mean_asym_minus_sym"] = abs(mean_delta[:, 1])-abs(mean_delta[:, 0])
    per_sample = pair_rows.groupby(["dataset_index", "token_count"], as_index=False)[
        ["delta_R_sym", "delta_R_asym"]].mean()
    full = signed_summary(delta)
    full["length_pearson_by_component"] = {
        c: dict(defined=int(head_rows[f"pearson_length_delta_{c}"].count()),
                median=float(head_rows[f"pearson_length_delta_{c}"].median()),
                minimum=float(head_rows[f"pearson_length_delta_{c}"].min()),
                maximum=float(head_rows[f"pearson_length_delta_{c}"].max()))
        for c in COMPONENTS
    }
    length_info = dict(min=int(lengths.min()), max=int(lengths.max()), mean=float(lengths.mean()),
                       median=float(np.median(lengths)),
                       counts={int(k): v for k, v in sorted(Counter(lengths).items())})
    validation = dict(
        status="passed", primary18_head_grid_matches_uu=True, token_length_mismatches=0,
        ratio_identity_max_error=float(error.max()), signed_change_identity_max_error=float(residual.max()),
        june_signed_mean_verified=True, absolute_tolerance=TOL,
        pair_opposite_sign_count=int((delta[..., 0]*delta[..., 1] < 0).sum()),
        pair_rows=len(pair_rows), head_rows=len(head_rows),
    )
    summary = dict(full=full, uu_same_signed_method=signed_summary(np.stack(uu_delta, axis=-1)
                   .reshape(18, 1024, 2)), token_lengths=length_info)
    return head_rows, pair_rows, per_sample, summary, validation


def figures(out, heads, paper):
    columns = ["mean_delta_R_sym", "mean_delta_R_asym", "abs_mean_asym_minus_sym"]
    limit = heads[columns[:2]].abs().to_numpy().max()
    for column, label in zip(columns, ("Mean ΔR_sym", "Mean ΔR_asym", "|Mean ΔR_asym| − |Mean ΔR_sym|")):
        bound = heads[column].abs().max() if column == columns[2] else limit
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        img = ax.imshow(heads[column].to_numpy().reshape(32, 32), cmap="RdBu_r",
                        vmin=-bound, vmax=bound, interpolation="nearest")
        ax.set(xlabel="Head (0–31)", ylabel="Layer (0–31)", title=f"Full prompt | {label} | primary18")
        fig.colorbar(img, ax=ax, label=label)
        fig.savefig(out/f"{column}.png", dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    x = np.arange(len(paper))
    ax.bar(x-.18, paper[columns[0]], .36, label="Symmetric", color="#4477AA")
    ax.bar(x+.18, paper[columns[1]], .36, label="Antisymmetric", color="#CC6677")
    ax.axhline(0, color="0.25", linewidth=.8)
    ax.set(xticks=x, xticklabels=[f"L{l} H{h}" for l,h in PAPER_HEADS],
           ylabel="Mean ΔR (GCG − Shuffled)", title="June paper heads | Full prompt | primary18")
    ax.legend()
    fig.savefig(out/"paper_heads.png", dpi=200)
    plt.close(fig)


def report(summary, validation, paper, primary):
    full = summary["full"]
    uu = summary["uu_same_signed_method"]
    prior = summary["uu_previous_absolute_method"]
    lengths = summary["token_lengths"]
    paper_rows = "\n".join(
        f"| L{r.layer} H{r.head} | {r.mean_delta_R_sym:+.8f} | {r.mean_delta_R_asym:+.8f} | "
        f"{r.abs_mean_asym_minus_sym:+.8f} |" for r in paper.itertuples(index=False))
    length_rows = "\n".join(
        f"| {c} | {d['median']:+.4f} | {d['minimum']:+.4f} ～ {d['maximum']:+.4f} | {d['defined']} |"
        for c, d in full["length_pearson_by_component"].items())
    return f"""# GCG vs Shuffled primary18のFull Score Norm比

全1,024 Headの等重み平均は対称 **{full['mean_delta_sym']:+.8f}**、
反対称 **{full['mean_delta_asym']:+.8f}**。
反対称の符号付き平均差の絶対値が大きいのは
**{full['heads_abs_mean_asym_gt_sym']}/1,024 Head（{full['fraction_abs_mean_asym_gt_sym']:.2%}）**。
平均差の符号が逆なのは **{full['heads_opposite_sign']}/1,024 Head（{full['fraction_opposite_sign']:.2%}）**。

## 条件と集計

Llama-2-7b-chat-hfの既存primary18。dataset indexは {primary}。
保存済みFP32 ScoreをCPUのFP64演算で扱い、チャットテンプレートを含む入力全体の
L×L行列を分解した。RoPE後・Mask前・Softmax前なので未来Token側のScoreも含む。
Frobenius Normの比を各サンプル・各Headで求め、GCG−Shuffledの符号付き差を18件で平均した。
6月の集計コード（Git {JUNE_COMMIT}）もペア差を先に取り、符号付きのまま算術平均する。
今回は指定どおり分母は全体Normそのもので、6月実装の1e-12加算と追加の系列長除算は適用していない。

## 固定paper_heads

| Head | 平均ΔR_sym | 平均ΔR_asym | 絶対値の差（asym−sym） |
|---|---:|---:|---:|
{paper_rows}

## UUとの比較

| 領域 | 全Head平均ΔR_sym | 全Head平均ΔR_asym | 反対称の平均差の絶対値が大きいHead | 平均差が逆符号のHead |
|---|---:|---:|---:|---:|
| Full L×L | {full['mean_delta_sym']:+.8f} | {full['mean_delta_asym']:+.8f} | {full['heads_abs_mean_asym_gt_sym']} | {full['heads_opposite_sign']} |
| UU 20×20、同じ符号付き集計 | {uu['mean_delta_sym']:+.8f} | {uu['mean_delta_asym']:+.8f} | {uu['heads_abs_mean_asym_gt_sym']} | {uu['heads_opposite_sign']} |

前回のUUで報告した {prior['heads_asym_gt_sym']}/1,024 Headは、各サンプルで絶対値を取ってから
平均する mean(|ΔR|) の結果。上表のUU行は前回のペア別CSVを使い、今回と同じ mean(ΔR)
を適用した。絶対値を取る順序によってペア間の相殺の有無が変わる。

## プロンプト長

GCGとShuffledは全18ペアでToken数一致。同じ18件を全Headが観測するため、Headごとの
平均Token数はすべて {lengths['mean']:.3f}。長さとの関係は各Head内で18件の実Token数と
符号付きΔRのPearson相関を求めた。長さは {lengths['min']}～{lengths['max']} Token、
中央値 {lengths['median']:g}。各長さの件数は {lengths['counts']}。

| 成分 | Head別相関の中央値 | 範囲 | 定義されたHead数 |
|---|---:|---:|---:|
{length_rows}

相関は同じ18件における記述値。サンプルの内容と長さを分離した効果推定ではない。

## 数学的性質と検証

各行列で R_sym² + R_asym² = 1。各ペアでは、両条件での比率の和を成分ごとに掛けた
符号付き差が互いに打ち消し合う。差が非零ならペア内の差分方向は逆になる。
この重みはサンプルごとに異なるため、18件を平均した後の符号や変化幅の大小は
ペア単位の恒等式だけでは決まらない。上のHead数やpaper_headsの値は観測結果で、
この比較だけからJailbreakに固有の効果は特定できない。

- 二乗和の最大絶対誤差は {validation['ratio_identity_max_error']:.3g}。
  ペアの符号付き差の恒等式の最大残差は {validation['signed_change_identity_max_error']:.3g}。
- Head数は厳密な符号と不等号で算出。絶対許容差1e-12を使うと、反対称優位は
  {full['heads_abs_mean_asym_gt_sym_at_tol']} Head、逆符号は {full['heads_opposite_sign_at_tol']} Head。
- [Head別CSV](heads.csv)・[ペア別CSV](pairs.csv)・[サンプル別の長さと全Head平均差](samples.csv)・
  [全体集計](summary.json)・[検証](validation.json)・[入力と設定](manifest.json)。
- [対称ヒートマップ](mean_delta_R_sym.png)・[反対称ヒートマップ](mean_delta_R_asym.png)は
  0中心・共通スケール。[平均差の絶対値の差](abs_mean_asym_minus_sym.png)・[paper heads](paper_heads.png)。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uu-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)

    grid, lengths, ratios, uu, hashes, plan = read_inputs(args.uu_dir.resolve())
    heads, pairs, samples, summary, validation = summarize(grid, lengths, ratios, uu)
    summary["uu_previous_absolute_method"] = read_json(args.uu_dir/"summary.json")["ratio"]
    paper = heads.set_index(["layer", "head"]).loc[PAPER_HEADS].reset_index()
    summary["paper_heads"] = paper.to_dict("records")
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError("input hash changed during analysis")

    def git(*command):
        return subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()

    primary = grid.get_level_values("dataset_index").unique().tolist()
    code = [Path(__file__).resolve(), ROOT/"utils/gcg_shuffled_common.py", ROOT/"utils/artifact_io.py"]
    manifest = dict(
        created_at=utc_now(), git_commit=git("rev-parse", "HEAD"),
        git_branch=git("branch", "--show-current"), git_status=git("status", "--short"),
        june_aggregation_commit=git("rev-parse", JUNE_COMMIT), score_extraction_commit=plan["git_commit"],
        code_sha256={str(p.relative_to(ROOT)): sha256_file(p) for p in code},
        input_sha256=hashes, primary_indices=primary, model=plan["model"],
        model_revision=plan["model_revision"], tokenizer=plan["tokenizer_info"],
        score_computation=plan["computation"], score_dtype="float32", analysis_dtype="float64",
        device="CPU", model_forward=False, sequence_scope="full prompt including template tokens",
        heads=[32, 32], paper_heads=PAPER_HEADS, delta="GCG - Shuffled",
        norm_ratio="component Frobenius norm / full Frobenius norm, no epsilon or length division",
        aggregation="paired signed differences, then sample mean per head, then equal-weight head mean",
        numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
    )
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in (("heads", heads), ("pairs", pairs), ("samples", samples), ("paper_heads", paper)):
        write_csv(out/f"{name}.csv", frame)
    for name, obj in (("summary", summary), ("validation", validation), ("manifest", manifest)):
        write_json(out/f"{name}.json", obj)
    figures(out, heads, paper)
    with (out/"report.md").open("x", encoding="utf-8") as handle:
        handle.write(report(summary, validation, paper, primary))
    write_json(out/"output_hashes.json", {p.name: sha256_file(p) for p in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
