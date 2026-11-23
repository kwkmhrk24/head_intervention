#!/usr/bin/env python3
"""Analyze paired component percent changes from the saved primary18 RMS CSV."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gcg-component-percent")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from utils.artifact_io import read_json, read_jsonl, sha256_file, utc_now
from utils.gcg_shuffled_common import DATASET, write_csv, write_json

PHASE = ROOT/"data/outputs/gcg_vs_shuffled/legacy_selection20/10_existing_attention"
RMS = PHASE/"full_component_rms_primary18/20261122"
RATIO = PHASE/"full_norm_change_primary18/20261121"
KEYS = ["dataset_index", "layer", "head"]
PARTS = ("sym", "asym")
PAPER = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
TINY = 1e-12
QUANTILES = (0, .01, .05, .25, .5, .75, .95, .99, 1)
P_SIGNED = ["mean_P_sym", "mean_P_asym"]
P_ABS = ["mean_abs_P_sym", "mean_abs_P_asym"]


def changes(gcg, baseline):
    """Return percent and natural log changes; undefined entries stay NaN."""
    g, b = np.broadcast_arrays(np.asarray(gcg, dtype=float), np.asarray(baseline, dtype=float))
    valid = np.isfinite(g) & np.isfinite(b) & (g >= 0) & (b > 0)
    log_valid = valid & (g > 0)
    percent = np.full(g.shape, np.nan)
    logarithm = np.full(g.shape, np.nan)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        percent[valid] = (g[valid]-b[valid])/b[valid]*100
        logarithm[log_valid] = np.log(g[log_valid])-np.log(b[log_valid])
        close = log_valid & (np.abs(g-b) <= b/2)
        logarithm[close] = np.log1p((g[close]-b[close])/b[close])
    percent[~np.isfinite(percent)] = np.nan
    logarithm[~np.isfinite(logarithm)] = np.nan
    return percent, logarithm


def read_inputs():
    hashes = {}
    for directory, names in ((RMS, ("sample_heads.csv","heads.csv","summary.json","manifest.json")),
                             (RATIO, ("pairs.csv","summary.json","manifest.json"))):
        ledger = directory/"output_hashes.json"
        hashes[str(ledger)] = sha256_file(ledger)
        expected = read_json(ledger)
        for name in names:
            path = directory/name
            hashes[str(path)] = sha256_file(path)
            if hashes[str(path)] != expected[name]:
                raise ValueError(f"input hash mismatch: {path}")
    manifest = read_json(RMS/"manifest.json")
    indices = manifest["primary_indices"]
    if (len(indices) != 18 or indices != read_json(RATIO/"manifest.json")["primary_indices"]
            or manifest["region"] != "entire unmasked L×L"):
        raise ValueError("primary18 or Full Score provenance mismatch")
    hashes[str(DATASET)] = sha256_file(DATASET)
    if hashes[str(DATASET)] != manifest["input_and_code_sha256"][str(DATASET)]:
        raise ValueError("dataset hash mismatch")
    dataset = {r["dataset_index"]: r for r in read_jsonl(DATASET)}
    for index in indices:
        conditions = dataset[index]["conditions"]
        if [conditions[c]["evaluation"]["success"] for c in ("original","gcg","shuffled")] != [False, True, False]:
            raise ValueError(f"primary18 labels differ: {index}")

    grid = pd.MultiIndex.from_product([indices, range(32), range(32)], names=KEYS).sort_values()
    samples = pd.read_csv(RMS/"sample_heads.csv", float_precision="round_trip").set_index(
        KEYS, verify_integrity=True).sort_index()
    reference = pd.read_csv(RATIO/"pairs.csv", float_precision="round_trip").set_index(
        KEYS, verify_integrity=True).sort_index()
    if not samples.index.equals(grid) or not reference.index.equals(grid):
        raise ValueError("saved Head/pair grids differ")
    index = samples.index.get_level_values("dataset_index")
    np.testing.assert_array_equal(samples.prompt_id, [dataset[i]["prompt_id"] for i in index])
    lengths = reference.token_count.to_numpy()
    max_n_error = max_ratio_error = 0.0
    for condition in ("gcg","shuffled"):
        np.testing.assert_array_equal(samples[f"seq_len_{condition}"], lengths)
        np.testing.assert_array_equal(lengths, [len(dataset[i]["conditions"][condition]["input_ids"]) for i in index])
        total = samples[f"N_total_{condition}"]
        for part in PARTS:
            n = samples[f"N_{part}_{condition}"]
            if (n.dropna() < 0).any():
                raise ValueError("negative component Norm")
            restored = samples[f"fro_norm_{part}_{condition}"]/lengths
            ratio = n/total
            finite_n = np.isfinite(n)
            finite_ratio = finite_n & np.isfinite(total) & (total > 0)
            for actual, expected, valid in ((n, restored, finite_n),
                                            (ratio, reference[f"R_{part}_{condition}"], finite_ratio)):
                np.testing.assert_allclose(actual[valid], expected[valid], atol=TINY, rtol=TINY)
            max_n_error = max(max_n_error, float(abs(n-restored)[finite_n].max()) if finite_n.any() else 0)
            max_ratio_error = max(max_ratio_error, float(abs(ratio-reference[f"R_{part}_{condition}"])[finite_ratio].max())
                                  if finite_ratio.any() else 0)
    return samples.reset_index(), hashes, manifest, dict(
        sample_head_rows=len(grid), primary_indices=indices, zero_based_grid=True,
        prompt_ids_match=True, token_length_mismatches=0,
        length_range=[int(lengths.min()), int(lengths.max())],
        norm_reconstruction_max_error=max_n_error, full_R_max_error=max_ratio_error,
    )


def quantiles(values):
    valid = values[np.isfinite(values)]
    return {str(q): float(valid.quantile(q)) if len(valid) else None for q in QUANTILES}


def dominance(sym, asym):
    gap = asym-sym
    return dict(sym=int((gap < 0).sum()), asym=int((gap > 0).sum()),
                tie=int((gap == 0).sum()), missing=int(gap.isna().sum()))


def describe(samples):
    keys = ["layer","head"]
    heads = samples[keys].drop_duplicates().sort_values(keys).set_index(keys)
    summary, diagnoses, anomaly_rows = {}, {}, []
    for part in PARTS:
        n, baseline = (samples[f"N_{part}_{c}"] for c in ("gcg","shuffled"))
        percent, logarithm = changes(n, baseline)
        samples[f"P_{part}"], samples[f"log_change_{part}"] = percent, logarithm
        samples[f"abs_P_{part}"], samples[f"abs_log_change_{part}"] = abs(percent), abs(logarithm)
        paired = np.isfinite(percent) & np.isfinite(logarithm)
        sign_mismatch = int(np.count_nonzero(np.sign(percent[paired]) != np.sign(logarithm[paired])))
        if sign_mismatch:
            raise ValueError(f"percent/log sign mismatch: {part}")
        finite = np.isfinite(n) & np.isfinite(baseline)
        zero = baseline.eq(0)
        tiny = baseline.gt(0) & baseline.le(TINY)
        low_cut = baseline[baseline.gt(0) & np.isfinite(baseline)].quantile(.01)
        low = baseline.gt(0) & baseline.le(low_cut)
        extreme_cut = samples[f"abs_P_{part}"].quantile(.99)
        extreme = samples[f"abs_P_{part}"].gt(extreme_cut)
        flagged = zero | tiny | low | extreme | ~finite | ~np.isfinite(percent) | ~np.isfinite(logarithm)
        flags = samples.loc[flagged, ["dataset_index","prompt_id",*keys,f"P_{part}",f"log_change_{part}"]].copy()
        flags.columns = ["dataset_index","prompt_id",*keys,"percent","log_change"]
        anomaly_rows.append(flags.assign(
            component=part, baseline=baseline[flagged], gcg=n[flagged],
            zero_baseline=zero[flagged], tiny_baseline=tiny[flagged], baseline_bottom_1pct=low[flagged],
            extreme_percent=extreme[flagged], nonfinite_input=(~finite)[flagged]))
        diagnoses[part] = dict(
            zero_baseline=int(zero.sum()), tiny_positive_baseline=int(tiny.sum()),
            zero_gcg=int(n.eq(0).sum()), nonfinite_inputs=int((~finite).sum()),
            missing_percent=int((~np.isfinite(percent)).sum()), missing_log=int((~np.isfinite(logarithm)).sum()),
            percent_log_sign_mismatch=sign_mismatch, comparable_sign_pairs=int(paired.sum()),
            baseline_bottom_1pct_cut=float(low_cut), baseline_bottom_1pct_count=int(low.sum()),
            extreme_abs_percent_cut=float(extreme_cut), extreme_abs_percent_count=int(extreme.sum()),
        )
        group = samples.groupby(keys, sort=True)
        for variable in (f"P_{part}",f"log_change_{part}"):
            heads[f"mean_{variable}"] = group[variable].mean()
            heads[f"median_{variable}"] = group[variable].median()
            heads[f"mean_abs_{variable}"] = group[f"abs_{variable}"].mean()
            heads[f"n_valid_{variable}"] = group[variable].count()
        heads[f"mean_baseline_{part}"] = group[f"N_{part}_shuffled"].mean()
        heads[f"mean_abs_delta_N_{part}"] = group[f"abs_delta_N_{part}"].mean()
        heads[f"mean_minus_median_P_{part}"] = heads[f"mean_P_{part}"]-heads[f"median_P_{part}"]
        p, log = heads[f"mean_P_{part}"], heads[f"mean_log_change_{part}"]
        signed_gap = heads[f"mean_minus_median_P_{part}"]
        gap_top = signed_gap.abs().nlargest(5)
        summary[part] = dict(
            mean_percent=float(p.mean()), mean_abs_percent=float(heads[f"mean_abs_P_{part}"].mean()),
            median_percent=float(samples[f"P_{part}"].median()),
            median_of_head_medians=float(heads[f"median_P_{part}"].median()),
            increase=int((p>0).sum()), decrease=int((p<0).sum()), tie=int((p==0).sum()),
            mean_log=float(log.mean()), mean_abs_log=float(heads[f"mean_abs_log_change_{part}"].mean()),
            median_log=float(samples[f"log_change_{part}"].median()),
            log_increase=int((log>0).sum()), log_decrease=int((log<0).sum()),
            log_tie=int((log==0).sum()), head_percent_log_sign_agreement=int((np.sign(p)==np.sign(log)).sum()),
            baseline_quantiles=quantiles(baseline), percent_quantiles=quantiles(samples[f"P_{part}"]),
            top_mean_median_gaps=[dict(layer=l, head=h, gap=float(signed_gap.loc[(l,h)]),
                                      mean=float(p.loc[(l,h)]), median=float(heads.loc[(l,h),f"median_P_{part}"]))
                                 for l,h in gap_top.index],
        )
        for level, source, base, delta_n, abs_p in (
            ("head", heads, f"mean_baseline_{part}", f"mean_abs_delta_N_{part}", f"mean_abs_P_{part}"),
            ("pair", samples, f"N_{part}_shuffled", f"abs_delta_N_{part}", f"abs_P_{part}"),
        ):
            summary[part][f"{level}_baseline_correlations"] = {
                kind: {v: float((source[base] if kind == "pearson" else source[base].rank()).corr(
                    source[col] if kind == "pearson" else source[col].rank()))
                       for v,col in (("absolute_delta_N",delta_n),("absolute_P",abs_p))}
                for kind in ("pearson","spearman")}
    return heads, summary, diagnoses, pd.concat(anomaly_rows, ignore_index=True)


def compare_methods(samples, heads):
    p_signed = dominance(heads.mean_P_sym.abs(), heads.mean_P_asym.abs())
    p_absolute = dominance(heads.mean_abs_P_sym, heads.mean_abs_P_asym)
    log_signed = dominance(heads.mean_log_change_sym.abs(), heads.mean_log_change_asym.abs())
    log_absolute = dominance(heads.mean_abs_log_change_sym, heads.mean_abs_log_change_asym)
    gap = heads.mean_P_asym.abs()-heads.mean_P_sym.abs()
    gap_abs = heads.mean_abs_P_asym-heads.mean_abs_P_sym
    heads["signed_P_asym_minus_sym"] = gap
    heads["absolute_P_asym_minus_sym"] = gap_abs
    sign_product = np.sign(heads.mean_P_sym)*np.sign(heads.mean_P_asym)
    comparison = {
        "P_signed": p_signed, "P_absolute": p_absolute,
        "log_signed": log_signed, "log_absolute": log_absolute,
        "P_mean_sign_relation": dict(same=int((sign_product>0).sum()), opposite=int((sign_product<0).sum()),
                                     tie=int((sign_product==0).sum()), missing=int(sign_product.isna().sum())),
        "P_log_signed_dominance_agreement": int((np.sign(gap)==np.sign(
            heads.mean_log_change_asym.abs()-heads.mean_log_change_sym.abs())).sum()),
        "P_log_absolute_dominance_agreement": int((np.sign(gap_abs)==np.sign(
            heads.mean_abs_log_change_asym-heads.mean_abs_log_change_sym)).sum()),
    }
    # Remove the same pair for both components, once per Head. Keep all rows in the main analysis.
    values = samples[["P_sym","P_asym"]].to_numpy().reshape(18, 1024, 2)
    finite = np.isfinite(values)
    largest = np.where(finite, abs(values), -np.inf).max(axis=2).argmax(axis=0)
    trimmed = values.copy()
    trimmed[largest, np.arange(1024), :] = np.nan
    signed = pd.DataFrame(trimmed.reshape(-1,2)).groupby(np.tile(np.arange(1024),18)).mean()
    absolute = pd.DataFrame(abs(trimmed).reshape(-1,2)).groupby(np.tile(np.arange(1024),18)).mean()
    comparison["omit_largest_pair_per_head"] = dict(
        signed_means=signed.mean().tolist(), absolute_means=absolute.mean().tolist(),
        signed_dominance=dominance(signed[0].abs(), signed[1].abs()),
        absolute_dominance=dominance(absolute[0],absolute[1]),
    )
    for j, part in enumerate(PARTS):
        heads[f"mean_P_without_max_pair_{part}"] = signed[j].to_numpy()
        heads[f"mean_abs_P_without_max_pair_{part}"] = absolute[j].to_numpy()
        heads[f"max_pair_removed_index"] = samples.dataset_index.unique()[largest]
    old_n = read_json(RMS/"summary.json")
    old_r = read_json(RATIO/"summary.json")["full"]
    prior_heads = pd.read_csv(RMS/"heads.csv", float_precision="round_trip").set_index(
        ["layer","head"], verify_integrity=True).sort_index()
    if not heads.index.equals(prior_heads.index):
        raise ValueError("prior RMS Head grid mismatch")
    for part in PARTS:
        np.testing.assert_allclose(heads[f"mean_abs_delta_N_{part}"],
                                   prior_heads[f"mean_abs_delta_N_{part}"], atol=TINY, rtol=0)
    comparison.update(R_signed=dict(asym=old_r["heads_abs_mean_asym_gt_sym"]),
                      N_signed=old_n["signed_dominance"], N_absolute=old_n["absolute_dominance"])
    if [comparison[k]["asym"] for k in ("R_signed","N_signed","N_absolute")] != [976,246,86]:
        raise ValueError("prior dominance counts changed")
    return comparison


def plot(out, heads, paper):
    def save(fig, filename):
        fig.savefig(out/f"{filename}.png", dpi=190)
        plt.close(fig)

    limit = heads[P_SIGNED].abs().to_numpy().max()
    fig, axes = plt.subplots(1,2, figsize=(10,4.6), layout="constrained")
    for ax, column, label in zip(axes, P_SIGNED, PARTS):
        im = ax.imshow(heads[column].to_numpy().reshape(32,32), cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(title=f"Mean P_{label}", xlabel="Head", ylabel="Layer")
    fig.colorbar(im, ax=axes, label="GCG − Shuffled (%)")
    save(fig, "signed_percent_heatmaps")

    fig, ax = plt.subplots(figsize=(5.7,5), layout="constrained")
    limit_abs = heads[P_ABS].max().max()
    ax.scatter(*[heads[col] for col in P_ABS], s=10, alpha=.45)
    ax.plot([0,limit_abs],[0,limit_abs], color=".4", linewidth=.8)
    ax.set(xlabel="Mean |P_sym| (%)", ylabel="Mean |P_asym| (%)", aspect="equal")
    save(fig, "mean_absolute_percent")

    fig, axes = plt.subplots(1,2, figsize=(10,4.6), layout="constrained")
    cmap = ListedColormap(["#4477AA","#e6e6e6","#CC6677"])
    norm = BoundaryNorm([-1.5,-.5,.5,1.5], 3)
    for ax, column, label in zip(axes, ("signed_P_asym_minus_sym","absolute_P_asym_minus_sym"),
                                ("|Mean P|", "Mean |P|")):
        im = ax.imshow(np.sign(heads[column].to_numpy()).reshape(32,32), cmap=cmap, norm=norm)
        ax.set(title=label, xlabel="Head", ylabel="Layer")
    bar = fig.colorbar(im, ax=axes, ticks=[-1,0,1])
    bar.ax.set_yticklabels(["Sym dominant","Tie","Asym dominant"])
    save(fig, "percent_dominance_maps")

    for name in ("baseline_vs_absolute_delta", "percent_vs_log"):
        fig, axes = plt.subplots(1,2, figsize=(10,4.3), layout="constrained")
        for ax, part in zip(axes, PARTS):
            if name == "baseline_vs_absolute_delta":
                ax.scatter(heads[f"mean_baseline_{part}"],heads[f"mean_abs_delta_N_{part}"],s=10,alpha=.45)
                ax.set(xlabel="Mean Shuffled component RMS",ylabel="Mean |ΔN|")
            else:
                ax.scatter(heads[f"mean_P_{part}"],100*heads[f"mean_log_change_{part}"],s=10,alpha=.45)
                ax.axhline(0,color=".5",linewidth=.7)
                ax.axvline(0,color=".5",linewidth=.7)
                ax.set(xlabel="Mean P (%)",ylabel="100 × mean log change")
            ax.set_title(part.capitalize())
        save(fig, name)

    fig, axes = plt.subplots(1,2, figsize=(10,4), layout="constrained")
    x = np.arange(5)
    for ax, cols, label in zip(axes, (P_SIGNED,P_ABS), ("Mean P (%)","Mean |P| (%)")):
        for j, (part,color) in enumerate(zip(PARTS,("#4477AA","#CC6677"))):
            ax.bar(x+(j-.5)*.36,paper[cols[j]], .36, label=part,color=color)
        ax.axhline(0,color=".4",linewidth=.8)
        ax.set(xticks=x,xticklabels=[f"L{l} H{h}" for l,h in PAPER],ylabel=label)
    axes[0].legend()
    save(fig,"paper_heads_percent")


def report(summary, comparison, checks, samples, heads):
    s, a = (summary[c] for c in PARTS)
    lines = [
        "# GCG vs Shuffled primary18のFull Score成分Norm変化率", "",
        "| 指標 | sym | asym |", "|---|---:|---:|",
        f"| 平均変化率（%） | {s['mean_percent']:+.5f} | {a['mean_percent']:+.5f} |",
        f"| 平均絶対変化率（%） | {s['mean_abs_percent']:.5f} | {a['mean_abs_percent']:.5f} |",
        f"| 変化率中央値（%） | {s['median_percent']:+.5f} | {a['median_percent']:+.5f} |",
        f"| 増加Head数 | {s['increase']} | {a['increase']} |",
        f"| 減少Head数 | {s['decrease']} | {a['decrease']} |", "",
        "平均は18ペアのHead別平均を1,024 Headで等重み平均。中央値は全サンプル×Headをプールした値。",
        f"Head別中央値の中央値はsym {s['median_of_head_medians']:+.5f}%、asym {a['median_of_head_medians']:+.5f}%。",
        "", "| 指標 | asym優勢Head数 | 割合 |", "|---|---:|---:|",
    ]
    labels = [
        ("R_signed","R：符号付き平均差の絶対値"), ("N_signed","N：符号付き平均差の絶対値"),
        ("N_absolute","N：平均絶対差"), ("P_signed","P：符号付き平均変化率の絶対値"),
        ("P_absolute","P：平均絶対変化率"),
    ]
    for key, label in labels:
        count = comparison[key]["asym"]
        lines.append(f"| {label} | {count}／1,024 | {count/1024:.2%} |")
    signs = comparison["P_mean_sign_relation"]
    lines.extend(["",
        f"Pの平均が同方向のHeadは{signs['same']}、逆方向は{signs['opposite']}、"
        f"少なくとも片方がゼロは{signs['tie']}、欠損は{signs['missing']}。", "",
        "## 基準Normと変化率", "",
        "| 成分 | Shuffled基準Normの最小／中央値／最大 | Head別の基準Normと mean \u007cΔN\u007c のPearson | Head別の基準Normと mean \u007cP\u007c のPearson |",
        "|---|---:|---:|---:|"])
    # Escape Markdown table separators inside labels.
    lines[-2] = lines[-2].replace("|ΔN|", "\\|ΔN\\|").replace("|P|", "\\|P\\|")
    for part, info in summary.items():
        q, c = info["baseline_quantiles"], info["head_baseline_correlations"]["pearson"]
        lines.append(f"| {part} | {q['0']:.6g}／{q['0.5']:.6g}／{q['1']:.6g} | "
                     f"{c['absolute_delta_N']:+.3f} | {c['absolute_P']:+.3f} |")
    baseline_wins = int((samples.N_sym_shuffled > samples.N_asym_shuffled).sum())
    lines.extend(["",
        f"Shuffledではsym基準Normがasymより大きいのは{baseline_wins:,}／18,432ペア×Head。",
        "同じ絶対差なら (P=100\u00d7\u0394N/N_{Shuffled}) により、基準Normが小さい成分ほど大きな絶対変化率になる。",
        "上表の相関は各成分のHead間の記述値。基準Normの差はNからPへの優勢数の変化を読む手掛かりだが、"
        "相関は成分間の基準値差だけでΔNの大小が決まることを示さない。",
        "分位点とHead・ペア水準のPearson／Spearman相関は [summary.json](summary.json)。",
        "", "## 分母・極端な値への感度", ""])
    for part, info in summary.items():
        d = checks["components"][part]
        extreme = samples.loc[samples[f"P_{part}"].abs().idxmax()]
        top = info["top_mean_median_gaps"][0]
        lines.append(
            f"- {part}。ゼロ分母{d['zero_baseline']}、正で1e-12以下の分母{d['tiny_positive_baseline']}、"
            f"非有限入力{d['nonfinite_inputs']}、P欠損{d['missing_percent']}、log欠損{d['missing_log']}。"
            f"基準Norm下位1%は{d['baseline_bottom_1pct_count']}件（境界 {d['baseline_bottom_1pct_cut']:.6g}）。"
            f"|P|の上位1%は{d['extreme_abs_percent_count']}件（境界 {d['extreme_abs_percent_cut']:.4f}%）。"
            f"最大|P|の観測は dataset index {extreme.dataset_index}、L{extreme.layer} H{extreme['head']} の"
            f" {extreme[f'P_{part}']:+.4f}%（基準Norm {extreme[f'N_{part}_shuffled']:.6g}）。"
            f"平均と中央値の差が最大のHeadは L{top['layer']} H{top['head']}（平均 {top['mean']:+.4f}%、"
            f"中央値 {top['median']:+.4f}%）。")
    trim = comparison["omit_largest_pair_per_head"]
    lines.extend(["",
        "基準Norm下位1%・|P|上位1%等の元プロンプトIDとHeadは [flagged_pairs.csv](flagged_pairs.csv)。"
        "主集計では分母への定数加算や外れ値の除外を行っていない。",
        f"感度確認として、Headごとに両成分のうち最大|P|を含む1ペアを両成分から外すと、"
        f"全Head平均Pは sym {trim['signed_means'][0]:+.5f}%、asym {trim['signed_means'][1]:+.5f}%。"
        f"asym優勢は符号付き平均の絶対値で {trim['signed_dominance']['asym']} Head、"
        f"平均絶対変化率で {trim['absolute_dominance']['asym']} Head。",
        "", "## log変化率による頑健性", "",
        f"各ペアのPと自然log変化の符号は、両指標が定義された全ペア×Headで一致した。"
        f"Head平均の符号一致はsym {s['head_percent_log_sign_agreement']}、asym {a['head_percent_log_sign_agreement']}／1,024。",
        f"logでのasym優勢は符号付き平均の絶対値で{comparison['log_signed']['asym']}、"
        f"平均絶対値で{comparison['log_absolute']['asym']} Head。Pと同じ優勢判定はそれぞれ"
        f"{comparison['P_log_signed_dominance_agreement']}、{comparison['P_log_absolute_dominance_agreement']} Head。",
        f"全Head平均logはsym {s['mean_log']:+.6f}、asym {a['mean_log']:+.6f}。"
        "Headごとの平均・中央値・平均絶対値は [heads.csv](heads.csv)。",
        "", "## 条件・検証・解釈", "",
        "Llama-2-7b-chat-hf、既存primary18、32 Layer×32 Head。既存の自動判定でGCG成功・Shuffled失敗の18ペア。",
        "RoPE後・Causal Mask前・Softmax前のFull L×L Score由来の保存NormをCPUで再利用した。",
        "元プロンプトID・Headの対応と18,432キーを照合。全ペアで系列長は条件間一致し40〜49 Tokenだった。",
        f"生Norm/Lからの最大差は {checks['norm_reconstruction_max_error']:.3g}、既存Full Rとの最大差は {checks['full_R_max_error']:.3g}。",
        "Pは各成分自身の基準Normに対する変化率。Rの二乗和1を直接は満たさないが、"
        "両成分は同じScoreの直交分解から得られ、成分エネルギーの和は全体エネルギーになる。",
        "測定対象が異なるためR・N・Pの優劣を優勢Head数から判断できない。"
        "今回の観測は選択済み18ペアの記述であり、Jailbreak成功への因果的寄与は未検証。",
        "", "[ペア別CSV](sample_heads.csv) · [全体集計CSV](summary.csv) · [固定5 Head](paper_heads.csv) · "
        "[来歴](manifest.json) · [検証](validation.json)",
        "", "[符号付きヒートマップ](signed_percent_heatmaps.png) · [平均絶対変化率](mean_absolute_percent.png) · "
        "[分類マップ](percent_dominance_maps.png) · [基準Normと絶対変化量](baseline_vs_absolute_delta.png) · "
        "[Pとlogの比較](percent_vs_log.png) · [固定5 Headの図](paper_heads_percent.png)", ""])
    return "\n".join(lines)


def flatten(data, prefix=()):
    for k, v in data.items():
        key = (*prefix, k)
        if isinstance(v, dict):
            yield from flatten(v, key)
        elif not isinstance(v, list):
            yield dict(metric=".".join(key), value=v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    out = parser.parse_args().output_dir
    if out.exists():
        raise FileExistsError(out)
    samples, hashes, source, checks = read_inputs()
    for path in (Path(__file__).resolve(), ROOT/"utils/artifact_io.py",
                 ROOT/"utils/gcg_shuffled_common.py"):
        hashes[str(path)] = sha256_file(path)
    for part in PARTS:
        delta = samples[f"N_{part}_gcg"]-samples[f"N_{part}_shuffled"]
        np.testing.assert_allclose(delta, samples[f"delta_N_{part}"], atol=TINY, rtol=0)
        np.testing.assert_allclose(abs(delta), samples[f"abs_delta_N_{part}"], atol=TINY, rtol=0)
    heads, summary, checks["components"], flags = describe(samples)
    comparison = compare_methods(samples, heads)
    checks.update(status="passed", heads=len(heads),
                  aggregation="sample-level change then summarize per Head",
                  sample_head_rows_per_head=18, prior_dominance_reproduced=True)
    paper = heads.loc[PAPER].reset_index()
    if any(sha256_file(path) != sha for path,sha in hashes.items()):
        raise ValueError("input changed during analysis")

    def git(*args):
        return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()

    manifest = dict(
        created_at=utc_now(), git_commit=git("rev-parse","HEAD"),
        git_branch=git("branch","--show-current"), git_status=git("status","--short"),
        input_and_code_sha256=hashes, source_rms_commit=source["git_commit"],
        primary_indices=source["primary_indices"], prompt_ids=source["prompt_ids"],
        model=source["model"], model_revision=source["model_revision"], tokenizer=source["tokenizer"],
        device="CPU", analysis_dtype="float64", model_forward=False, region=source["region"],
        score_computation=source["score_computation"], heads=[32,32],
        percentage="100 * (GCG - Shuffled) / Shuffled component RMS",
        log_change="natural log of GCG/Shuffled, both finite and positive",
        missing="empty CSV fields for undefined values", denominator_tiny_threshold=TINY,
        sensitivity="omit each Head's max-|P| pair in both components; ties use first dataset index",
        numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__)
    result = dict(components=summary, comparison=comparison)
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in (("sample_heads",samples), ("heads",heads.reset_index()),
                        ("flagged_pairs",flags), ("paper_heads",paper),
                        ("summary",pd.DataFrame(flatten(result)))):
        write_csv(out/f"{name}.csv",frame)
    for name, payload in (("summary",result), ("validation",checks), ("manifest",manifest)):
        write_json(out/f"{name}.json",payload)
    plot(out,heads,paper)
    with (out/"report.md").open("x",encoding="utf-8") as handle:
        handle.write(report(summary,comparison,checks,samples,heads))
    write_json(out/"output_hashes.json",{p.name:sha256_file(p) for p in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
