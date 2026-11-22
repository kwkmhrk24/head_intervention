#!/usr/bin/env python3
"""Compute component RMS on saved Full Score matrices for legacy primary18."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gcg-full-component-rms")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from utils.artifact_io import read_json, sha256_file, utc_now
from utils.gcg_shuffled_common import DATASET, SCORES, inspect_inputs, paired_stats, write_csv, write_json
from utils.gcg_shuffled_metrics import TIE_ATOL, TIE_RTOL

PHASE = ROOT/"data/outputs/gcg_vs_shuffled/legacy_selection20/10_existing_attention"
PRIOR = PHASE/"full_norm_change_primary18/20261121"
KEYS = ["dataset_index", "layer", "head"]
COMPONENTS = ("sym", "asym")
PAPER = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
TOL = 1e-12


def component_rms(score):
    s = np.asarray(score, dtype=np.float64)
    if not np.isfinite(s).all():
        raise ValueError("nonfinite Score")
    t = s.swapaxes(-1, -2)
    sym, asym = (s+t)/2, (s-t)/2
    norms = np.stack([np.linalg.norm(x, axis=(-2, -1)) for x in (s, sym, asym)], axis=-1)
    rec = float(np.abs((sym+asym)-s).max())
    energy = norms[..., 0]**2
    remaining = np.abs(energy-(norms[..., 1:]**2).sum(axis=-1))
    relative = float((remaining/np.maximum(energy, np.finfo(float).tiny)).max())
    if rec > TOL or relative > TOL:
        raise ValueError(f"decomposition failed reconstruction or energy check: {rec}, {relative}")
    return norms/s.shape[-1], rec, relative


def from_scores(references, indices, conditions, hashes):
    refs = {(r["dataset_index"], r["condition"]): r for r in references}
    arrays, lengths, errors = [], [], []
    for index in indices:
        parts, sizes = [], []
        for c in conditions:
            ref = refs[index, c]
            hashes[ref["path"]] = ref["sha256"]
            hashes[ref["metadata_path"]] = ref["metadata_sha256"]
            extraction = read_json(ref["metadata_path"])["extraction"]
            if not extraction["computation"].startswith("post-RoPE"):
                raise ValueError(f"not post-RoPE: {index}/{c}")
            score = np.load(ref["path"], mmap_mode="r", allow_pickle=False)
            rms, rec, energy = component_rms(score)
            parts.append(rms.reshape(1024, 3))
            sizes.append(score.shape[-1])
            errors.append(dict(dataset_index=index, condition=c, reconstruction_max_abs=rec,
                               energy_max_relative=energy))
        arrays.append(parts)
        lengths.append(sizes)
    return np.asarray(arrays), np.asarray(lengths), errors


def summarize(values):
    stats = paired_stats(values[..., 1:], list(range(len(values))))
    signed, absolute = stats["delta_mean"], stats["delta_mean_absolute"]
    if not (stats["pair_valid_count"] == 18).all():
        raise ValueError("incomplete Head pairs")
    if (np.abs(signed) > absolute+TOL).any():
        raise ValueError("absolute mean exceeds mean absolute change")
    np.testing.assert_array_equal(stats["gcg_larger_count"]+stats["shuffled_larger_count"]
                                  +stats["equal_count"], 18)
    layer, head = np.indices((32, 32))
    frame = pd.DataFrame(dict(layer=layer.ravel(), head=head.ravel(), n_pairs=18))
    summary = {}
    signs = np.sign(np.where(np.abs(signed) > TOL, signed, 0))
    for j, part in enumerate(COMPONENTS):
        frame[f"mean_delta_N_{part}"] = signed[:, j]
        frame[f"mean_abs_delta_N_{part}"] = absolute[:, j]
        for label, key in (("increase", "gcg_larger_count"), ("decrease", "shuffled_larger_count"),
                           ("tie", "equal_count")):
            frame[f"{part}_{label}_pairs"] = stats[key][:, j]
        direction_count = np.where(signs[:, j] > 0, stats["gcg_larger_count"][:, j],
                            np.where(signs[:, j] < 0, stats["shuffled_larger_count"][:, j],
                                     stats["equal_count"][:, j]))
        frame[f"{part}_pairs_matching_mean_direction"] = direction_count
        summary[part] = dict(
            mean_signed=float(signed[:, j].mean()), mean_absolute=float(absolute[:, j].mean()),
            increase=int((signs[:, j] > 0).sum()), decrease=int((signs[:, j] < 0).sum()),
            tie=int((signs[:, j] == 0).sum()),
            all_18_increase=int((stats["gcg_larger_count"][:, j] == 18).sum()),
            all_18_decrease=int((stats["shuffled_larger_count"][:, j] == 18).sum()),
            median_pairs_matching_mean_direction=float(np.median(direction_count)),
        )
    for label, magnitudes in (("signed", np.abs(signed)), ("absolute", absolute)):
        gap = magnitudes[:, 1]-magnitudes[:, 0]
        category = np.sign(np.where(np.abs(gap) > TOL, gap, 0)).astype(int)
        frame[f"{label}_asym_minus_sym"] = gap
        frame[f"{label}_dominance"] = category
        summary[f"{label}_dominance"] = dict(sym=int((category < 0).sum()),
                                              asym=int((category > 0).sum()),
                                              tie=int((category == 0).sum()))
    product = signs[:, 0]*signs[:, 1]
    summary["direction_agreement"] = dict(same=int((product > 0).sum()), opposite=int((product < 0).sum()),
                                          either_tie=int((product == 0).sum()))
    return frame, summary


def figures(out, heads, paper):
    signed = [f"mean_delta_N_{c}" for c in COMPONENTS]
    absolute = [f"mean_abs_delta_N_{c}" for c in COMPONENTS]
    limit = heads[signed].abs().to_numpy().max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    for ax, column, label in zip(axes, signed, COMPONENTS):
        image = ax.imshow(heads[column].to_numpy().reshape(32,32), cmap="RdBu_r",
                          vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set(title=f"Mean ΔN_{label}", xlabel="Head", ylabel="Layer")
    fig.colorbar(image, ax=axes, label="GCG − Shuffled")
    fig.savefig(out/"signed_heatmaps.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    upper = heads[absolute].to_numpy().max()*1.04
    ax.scatter(heads[absolute[0]], heads[absolute[1]], s=10, alpha=.45, edgecolors="none")
    ax.plot([0,upper], [0,upper], color="0.4", linewidth=.8)
    ax.set(xlim=(0,upper), ylim=(0,upper), aspect="equal",
           xlabel="Mean |ΔN_sym|", ylabel="Mean |ΔN_asym|", title="Full component RMS | 1,024 Heads")
    fig.savefig(out/"mean_absolute_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    cmap = ListedColormap(["#4477AA", "#e6e6e6", "#CC6677"])
    norm = BoundaryNorm([-1.5,-.5,.5,1.5], cmap.N)
    for ax, (column, label) in zip(axes, (("signed_dominance", "|Mean ΔN|"),
                                         ("absolute_dominance", "Mean |ΔN|"))):
        image = ax.imshow(heads[column].to_numpy().reshape(32,32), cmap=cmap, norm=norm,
                          interpolation="nearest")
        ax.set(title=label, xlabel="Head", ylabel="Layer")
    bar = fig.colorbar(image, ax=axes, ticks=[-1,0,1])
    bar.ax.set_yticklabels(["Sym dominant", "Tie", "Asym dominant"])
    fig.savefig(out/"dominance_maps.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    x = np.arange(5)
    for ax, columns, label in zip(axes, (signed, absolute), ("Mean ΔN", "Mean |ΔN|")):
        ax.bar(x-.18, paper[columns[0]], .36, label="Sym", color="#4477AA")
        ax.bar(x+.18, paper[columns[1]], .36, label="Asym", color="#CC6677")
        ax.axhline(0, color=".3", linewidth=.8)
        ax.set(xticks=x, xticklabels=[f"L{l} H{h}" for l,h in PAPER], ylabel=label)
    axes[0].legend()
    fig.savefig(out/"paper_heads.png", dpi=200)
    plt.close(fig)


def report(summary, validation, paper):
    s, a = (summary[c] for c in COMPONENTS)
    signed, absolute = (summary[f"{c}_dominance"] for c in ("signed", "absolute"))
    direction = summary["direction_agreement"]
    fixed_rows = "\n".join(
        f"| L{r.layer} H{r.head} | {r.mean_delta_N_sym:+.6f} | {r.mean_delta_N_asym:+.6f} | "
        f"{r.mean_abs_delta_N_sym:.6f} | {r.mean_abs_delta_N_asym:.6f} |"
        for r in paper.itertuples(index=False))
    original = ("同じ18件のOriginalのFull Scoreも、元プロンプト対応・入力Token ID・領域・保存条件を"
                "照合して `sample_heads.csv` に保存した。"
                if validation["original"]["available"] else
                f"OriginalのFull Scoreを利用できなかった。理由は `{validation['original']['reason']}`。")
    return f"""# GCG vs Shuffled primary18のFull Score成分RMS

| 指標 | sym | asym |
|---|---:|---:|
| 符号付き平均差の全Head平均 | {s['mean_signed']:+.8f} | {a['mean_signed']:+.8f} |
| 平均絶対差の全Head平均 | {s['mean_absolute']:.8f} | {a['mean_absolute']:.8f} |
| 増加Head数 | {s['increase']} | {a['increase']} |
| 減少Head数 | {s['decrease']} | {a['decrease']} |

## 優勢と変化方向

| 優勢の基準 | sym優勢 | asym優勢 | 同等 |
|---|---:|---:|---:|
| 符号付き平均差の絶対値 | {signed['sym']} | {signed['asym']}（{signed['asym']/1024:.2%}） | {signed['tie']} |
| 平均絶対差 | {absolute['sym']} | {absolute['asym']}（{absolute['asym']/1024:.2%}） | {absolute['tie']} |

符号付き平均差が同方向のHeadは {direction['same']}、逆方向は {direction['opposite']}、
少なくとも片方が同等は {direction['either_tie']}。成分ごとの増減で同等は sym {s['tie']}、asym {a['tie']}。
18件すべてが増加／減少したHead数は sym {s['all_18_increase']}／{s['all_18_decrease']}、
asym {a['all_18_increase']}／{a['all_18_decrease']}。18件のうちHeadの平均差と
同じ方向だった件数の中央値は sym {s['median_pairs_matching_mean_direction']:g}、asym
{a['median_pairs_matching_mean_direction']:g}。各Headの増加・減少・同等件数は [heads.csv](heads.csv)。

## 条件と検証

Llama-2-7b-chat-hfの既存primary18、全32 Layer×32 Head。既存自動判定で成功のGCGから
失敗のShuffledを引いた。保存済みのRoPE後・Causal Mask前・Softmax前のFull Scoreを使用し、
未来位置の要素と対角も含めた。実系列長は各ペアで一致し、範囲は {validation['length_min']}～{validation['length_max']} Token。
全18×1,024キーを既存Full比率分析と照合した。{original}

各行列での N は成分のFrobenius Normを実系列長Lで割った要素RMS。
ペア差をそのまま平均すると符号の相殺が生じる。ペア差の絶対値を先に取る平均絶対差は変化量を表す。
Headの増減・優勢で絶対差1e-12以内、ペアの増減では絶対差1e-12＋相対差1e-12を同等とした。

再構成の最大絶対誤差は {validation['reconstruction_max_abs']:.3g}、
エネルギー恒等式の最大相対誤差は {validation['energy_max_relative']:.3g}。
照合のためにだけ成分RMSを全体RMSで割った比率を作り、既存Full比率との最大差
{validation['prior_ratio_max_abs_error']:.3g} を確認した。主集計には全体Normを分母に使わず、
既存UUの生Norm集計は再利用していない。処理はCPUのみ。

## 固定paper_heads

| Head | mean ΔN_sym | mean ΔN_asym | mean abs(ΔN_sym) | mean abs(ΔN_asym) |
|---|---:|---:|---:|---:|
{fixed_rows}

## 解釈

既存の同じFull ScoreのNorm比では、符号付き平均差の絶対値で976/1,024 Head（95.31%）が
asym優勢だった。長さ正規化Normでは同じ基準で {signed['asym']}/1,024 Headとなった。
長さ正規化Normでも N_sym² + N_asym² = (||S||_F/L)² が成り立つが、
右辺は入力によって変わる。比率の二乗和が1という制約と異なり、成分の同方向の変化も可能。
増減・優勢の件数は実際の観測結果である。自動判定で選んだ18ペアの記述分析であり、
依存する1,024 Headを独立標本として一般化したり、Jailbreak成功への因果的寄与を判断したりできない。

[符号付きヒートマップ](signed_heatmaps.png) · [平均絶対差の比較](mean_absolute_comparison.png) ·
[優勢分類](dominance_maps.png) · [固定5 Headの棒グラフ](paper_heads.png)

[サンプル×Head指標](sample_heads.csv) · [Head別集計](heads.csv) ·
[全体サマリーCSV](summary.csv) · [JSON](summary.json) ·
[検証](validation.json) · [入力と設定](manifest.json)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    plan = read_json(SCORES/"plan.json")
    if plan["model_name"] != "meta-llama/Llama-2-7b-chat-hf":
        raise ValueError("unexpected model")
    dependencies = [DATASET, DATASET.parent/"plan.json", SCORES/"plan.json",
                    SCORES/"manifest.jsonl", SCORES/"validation_final.json",
                    PRIOR/"output_hashes.json", Path(__file__).resolve(),
                    ROOT/"utils/gcg_shuffled_common.py", ROOT/"utils/gcg_shuffled_metrics.py",
                    ROOT/"utils/artifact_io.py"]
    hashes = {str(p): sha256_file(p) for p in dependencies}
    ledger = read_json(PRIOR/"output_hashes.json")
    for name in ("pairs.csv", "summary.json", "manifest.json"):
        path = PRIOR/name
        hashes[str(path)] = sha256_file(path)
        if hashes[str(path)] != ledger[name]:
            raise ValueError(f"prior artifact hash mismatch: {path}")

    dataset, primary, refs = inspect_inputs(conditions=("gcg", "shuffled"))
    indices = sorted(primary)
    if indices != read_json(PRIOR/"manifest.json")["primary_indices"]:
        raise ValueError("primary18 differs from prior analysis")
    ids = {r["dataset_index"]: r["prompt_id"] for r in dataset}
    if len({ids[i] for i in indices}) != 18:
        raise ValueError("duplicate primary prompt IDs")
    values, lengths, diagnostics = from_scores(refs, indices, ("gcg", "shuffled"), hashes)
    np.testing.assert_array_equal(lengths[:, 0], lengths[:, 1])

    grid = pd.MultiIndex.from_product([indices, range(32), range(32)], names=KEYS)
    prior = pd.read_csv(PRIOR/"pairs.csv", float_precision="round_trip").set_index(
        KEYS, verify_integrity=True).sort_index()
    if not prior.index.equals(grid):
        raise ValueError("Head/pair keys differ from prior Full analysis")
    np.testing.assert_array_equal(prior.token_count, np.repeat(lengths[:,0], 1024))
    if (values[...,0] <= 0).any():
        raise ValueError("undefined reference Norm ratio")
    ratio_difference = 0.0
    samples = grid.to_frame(index=False)
    samples.insert(1, "prompt_id", samples.dataset_index.map(ids))

    def add_condition(c, rms, sequence_lengths):
        samples[f"seq_len_{c}"] = np.repeat(sequence_lengths, 1024)
        for j, part in enumerate(("total", *COMPONENTS)):
            n = rms[..., j].ravel()
            samples[f"N_{part}_{c}"] = n
            samples[f"fro_norm_{part}_{c}"] = n*samples[f"seq_len_{c}"]

    for k, c in enumerate(("gcg","shuffled")):
        add_condition(c, values[:,k], lengths[:,k])
        for j, part in enumerate(COMPONENTS, 1):
            ratio = (values[:,k,:,j]/values[:,k,:,0]).ravel()
            np.testing.assert_allclose(ratio, prior[f"R_{part}_{c}"], atol=TOL, rtol=0)
            ratio_difference = max(ratio_difference, float(np.abs(ratio-prior[f"R_{part}_{c}"]).max()))
    for part in COMPONENTS:
        samples[f"delta_N_{part}"] = samples[f"N_{part}_gcg"]-samples[f"N_{part}_shuffled"]
        samples[f"abs_delta_N_{part}"] = samples[f"delta_N_{part}"].abs()

    original = dict(available=False, n_samples=0)
    try:
        _, original_primary, original_refs = inspect_inputs(conditions=("original",))
        if sorted(original_primary) != indices:
            raise ValueError("Original indices differ from primary18")
        original_values, original_lengths, original_diagnostics = from_scores(
            original_refs, indices, ("original",), hashes)
    except (OSError, ValueError) as exc:
        original["reason"] = str(exc)
    else:
        original = dict(available=True, n_samples=18, n_head_rows=18*1024)
        add_condition("original", original_values[:,0], original_lengths[:,0])
        diagnostics.extend(original_diagnostics)

    heads, summary = summarize(values)
    paper = heads.set_index(["layer","head"]).loc[PAPER].reset_index()
    prior_summary = read_json(PRIOR/"summary.json")["full"]
    if prior_summary["heads_abs_mean_asym_gt_sym"] != 976:
        raise ValueError("unexpected prior Full ratio dominance")
    validation = dict(
        status="passed", n_pairs=18, head_rows=1024, sample_head_rows=len(samples),
        token_length_mismatches=0, length_min=int(lengths.min()), length_max=int(lengths.max()),
        prior_ratio_max_abs_error=ratio_difference,
        reconstruction_max_abs=max(r["reconstruction_max_abs"] for r in diagnostics),
        energy_max_relative=max(r["energy_max_relative"] for r in diagnostics),
        score_checks=diagnostics, original=original,
    )
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError("input changed during analysis")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    manifest = dict(created_at=utc_now(), git_commit=git("rev-parse", "HEAD"),
                    git_branch=git("branch", "--show-current"), git_status=git("status", "--short"),
                    input_and_code_sha256=hashes, primary_indices=indices, prompt_ids={i:ids[i] for i in indices},
                    model=plan["model"], model_revision=plan["model_revision"], tokenizer=plan["tokenizer_info"],
                    score_extraction_commit=plan["git_commit"], score_computation=plan["computation"],
                    device="CPU", score_dtype="float32", analysis_dtype="float64",
                    model_forward=False, region="entire unmasked L×L", heads=[32,32],
                    normalization="component Frobenius norm / actual sequence length",
                    delta="GCG - Shuffled", head_tolerance=TOL, pair_atol=TIE_ATOL, pair_rtol=TIE_RTOL,
                    numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__)
    out.mkdir(parents=True, exist_ok=False)
    summary_rows = pd.DataFrame([dict(section=k, metric=m, value=v)
                                 for k, section in summary.items() for m,v in section.items()])
    for name, frame in (("sample_heads", samples), ("heads", heads), ("paper_heads", paper),
                        ("summary", summary_rows)):
        write_csv(out/f"{name}.csv", frame)
    for name, data in (("summary", summary), ("validation", validation), ("manifest", manifest)):
        write_json(out/f"{name}.json", data)
    figures(out, heads, paper)
    with (out/"report.md").open("x", encoding="utf-8") as handle:
        handle.write(report(summary, validation, paper))
    write_json(out/"output_hashes.json", {p.name: sha256_file(p) for p in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
