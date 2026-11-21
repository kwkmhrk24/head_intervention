#!/usr/bin/env python3
"""Compare Head directions using the saved signed-mean Tail20 and Full CSVs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-llama2-head-directions")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

from utils.artifact_io import read_json, sha256_file, utc_now
from utils.gcg_shuffled_common import write_csv, write_json

BASE = ROOT / "data/outputs"
TAIL = BASE / "llama2/tail20_norm_ratio_comparison/20261121"
FULL = BASE / "gcg_vs_shuffled/legacy_selection20/10_existing_attention/full_norm_change_primary18/20261121"
LABELS = {"A": ("AutoDAN − Original", 130, "Tail20", 994),
          "B": ("GCG − Original", 160, "Tail20", 987),
          "C1": ("GCG − Shuffled", 18, "Tail20", 975),
          "C2": ("GCG − Shuffled", 18, "Full L×L", 976)}
PAPER = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
SYM, ASYM = "mean_delta_R_sym", "mean_delta_R_asym"
CLASSES = ["asym dominant, increase", "asym dominant, decrease",
           "not asym dominant, increase", "not asym dominant, decrease", "near zero"]
TOL = 1e-7


def read_sources():
    hashes = {}
    for directory in (TAIL, FULL):
        ledger = directory / "output_hashes.json"
        hashes[str(ledger)] = sha256_file(ledger)
        expected = read_json(ledger)
        for name in ("heads.csv", "summary.json", "manifest.json"):
            path = directory / name
            hashes[str(path)] = sha256_file(path)
            if hashes[str(path)] != expected[name]:
                raise ValueError(f"input hash mismatch: {path}")
    tail = pd.read_csv(TAIL/"heads.csv", float_precision="round_trip")
    full = pd.read_csv(FULL/"heads.csv", float_precision="round_trip")
    grid = pd.MultiIndex.from_product([range(32), range(32)], names=["layer", "head"])
    if set(tail.condition) != {"A", "B", "C"}:
        raise ValueError("unexpected Tail20 conditions")
    sources = read_json(TAIL/"summary.json")
    sources["C2"] = read_json(FULL/"summary.json")["full"]
    frames, summaries = [], []
    for code, (comparison, pairs, region, expected_count) in LABELS.items():
        source_code = "C" if code == "C1" else code
        raw = full if code == "C2" else tail.loc[tail.condition == source_code]
        raw = raw.set_index(["layer", "head"], verify_integrity=True).sort_index()
        n = raw["n_samples" if code == "C2" else "n_pairs"]
        if not raw.index.equals(grid) or not n.eq(pairs).all():
            raise ValueError(f"Head grid or pair count mismatch: {code}")
        sym, asym = raw[SYM].to_numpy(), raw[ASYM].to_numpy()
        if not np.isfinite(raw[[SYM, ASYM]].to_numpy()).all():
            raise ValueError(f"nonfinite signed mean: {code}")
        gap = np.abs(asym)-np.abs(sym)
        dominant, increase, decrease = gap > 0, asym > 0, asym < 0
        near = np.abs(asym) <= TOL
        classification = np.select(
            [near, dominant & increase, dominant & decrease, increase, decrease],
            [4, 0, 1, 2, 3], default=-1)
        np.testing.assert_allclose(gap, raw.abs_mean_asym_minus_sym, atol=1e-15, rtol=0)
        prior = sources[source_code]
        prior_count = prior["heads_abs_mean_asym_gt_sym" if code == "C2" else "asym_dominant"]
        if int(dominant.sum()) != prior_count or prior_count != expected_count:
            raise ValueError(f"historical dominance count mismatch: {code}")
        if code != "C2":
            np.testing.assert_array_equal(dominant, raw.asym_dominant)
        prior_mean_keys = ("mean_delta_sym", "mean_delta_asym") if code == "C2" else (SYM, ASYM)
        np.testing.assert_allclose([sym.mean(), asym.mean()],
                                   [prior[k] for k in prior_mean_keys],
                                   atol=1e-15, rtol=0)
        strict_sign_counts = [int(increase.sum()), int(decrease.sum()), int((asym==0).sum())]
        map_counts = np.bincount(classification, minlength=5)
        if sum(strict_sign_counts) != 1024 or map_counts.sum() != 1024:
            raise ValueError(f"category denominator mismatch: {code}")
        near_gap = np.abs(gap) <= TOL
        summaries.append(dict(
            condition=code, comparison=comparison, region=region, n_pairs=pairs, n_heads=1024,
            asym_dominant=int(dominant.sum()), asym_dominant_fraction=float(dominant.mean()),
            asym_increase=strict_sign_counts[0], asym_increase_fraction=float(increase.mean()),
            asym_decrease=strict_sign_counts[1], asym_decrease_fraction=float(decrease.mean()),
            asym_exact_zero=strict_sign_counts[2],
            asym_dominant_and_increase=int((dominant & increase).sum()),
            asym_dominant_and_decrease=int((dominant & decrease).sum()),
            asym_dominant_and_exact_zero=int((dominant & (asym == 0)).sum()),
            mean_delta_R_sym=float(sym.mean()), mean_delta_R_asym=float(asym.mean()),
            near_zero=int(near.sum()), near_zero_positive=int((near & increase).sum()),
            near_zero_negative=int((near & decrease).sum()),
            near_zero_min=float(asym[near].min()) if near.any() else None,
            near_zero_max=float(asym[near].max()) if near.any() else None,
            dominance_gap_near_zero=int(near_gap.sum()),
            dominance_gap_near_zero_min=float(gap[near_gap].min()) if near_gap.any() else None,
            dominance_gap_near_zero_max=float(gap[near_gap].max()) if near_gap.any() else None,
            **{f"map_class_{i}": int(c) for i, c in enumerate(map_counts)},
        ))
        frames.append(raw[[SYM, ASYM]].reset_index().assign(
            condition=code, n_pairs=pairs, asym_dominance_gap=gap, asym_dominant=dominant,
            asym_sign=np.sign(asym).astype(int), asym_near_zero=near, map_class=classification,
            map_label=[CLASSES[i] for i in classification]))
    return pd.concat(frames, ignore_index=True), summaries, hashes


def plot(out, heads, summary):
    labels = ["A\nTail20", "B\nTail20", "C1\nTail20", "C2\nFull"]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    for offset, key, title, color in (
        (-.19, "asym_dominant_fraction", "Asym dominant", "#4477AA"),
        (.19, "asym_increase_fraction", "Asym increases", "#CC6677"),
    ):
        bars = ax.bar(x+offset, [100*r[key] for r in summary], .35, label=title, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Share of all 1,024 heads", ylim=(0, 110))
    ax.legend(loc="lower left")
    fig.savefig(out/"asym_head_fractions.png", dpi=200)
    plt.close(fig)

    limit = heads[ASYM].abs().max()
    colors = ["#a6222c", "#285ab0", "#eda8a2", "#a3bee9", "#e6e6e6"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(6)-.5, cmap.N)
    for column, filename in ((ASYM, "asym_signed_heatmaps.png"), ("map_class", "head_classification_maps.png")):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
        for ax, (code, (comparison, _, region, _)) in zip(axes.flat, LABELS.items()):
            values = heads.loc[heads.condition == code, column].to_numpy().reshape(32, 32)
            style = dict(cmap="RdBu_r", vmin=-limit, vmax=limit) if column == ASYM else dict(cmap=cmap, norm=norm)
            image = ax.imshow(values, interpolation="nearest", **style)
            ax.set(title=f"{code} · {comparison} · {region}", xlabel="Head", ylabel="Layer")
        if column == ASYM:
            fig.colorbar(image, ax=axes, label="Mean ΔR_asym", shrink=.8)
        else:
            bar = fig.colorbar(image, ax=axes, ticks=range(5), shrink=.8)
            bar.ax.set_yticklabels(CLASSES)
        fig.savefig(out/filename, dpi=200)
        plt.close(fig)


def report(summary, paper):
    lines = ["# Attention Score Norm比の4条件比較", "",
             "| 条件 | asym優勢Head | asym増加Head | asym減少Head | asym優勢かつ増加 |",
             "|---|---:|---:|---:|---:|"]
    for r in summary:
        label = r["condition"]
        if label.startswith("C"):
            label += f"（{r['region']}）"
        metrics = [f"{r['asym_'+k]}/1,024（{r['asym_'+k+'_fraction']:.2%}）"
                   for k in ("dominant", "increase", "decrease")]
        lines.append(f"| {label} | " + " | ".join(metrics) + f" | {r['asym_dominant_and_increase']} |")
    lines.extend(["", "## 条件と追加集計", "",
        "すべてLlama-2-7b-chat-hfのRoPE後・Causal Mask前・Softmax前Score。Tail20の窓幅は min(20, L)。",
        "C1・C2は同じprimary18で窓が異なる。A・BとCではサンプル数、入力集合、比較対象が異なる。",
        "全条件でペア差を符号付きで平均し、その後で絶対値を比較した。", "",
        "| 条件 | Attack − Control | ペア数 | 領域 | 優勢かつ減少 | 全Head平均 ΔR_sym | 全Head平均 ΔR_asym |",
        "|---|---|---:|---|---:|---:|---:|"])
    for r in summary:
        lines.append(f"| {r['condition']} | {r['comparison']} | {r['n_pairs']} | {r['region']} | "
                     f"{r['asym_dominant_and_decrease']} | {r[SYM]:+.8f} | {r[ASYM]:+.8f} |")
    lines.extend(["", "## ゼロ近傍と分母", "",
        "増減の主集計は厳密な正負。分類図では |mean ΔR_asym| ≤ 1e-7 を優先してゼロ近傍に分類した。",
        "優勢判定の境界についても、絶対値の差が ±1e-7 以内のHeadを確認した。",
        "1e-7は前回のTail20比較で用いた感度確認の閾値。", "",
        "| 条件 | 完全なゼロ | ゼロ近傍（正・負） | ゼロ近傍の値域 | 優勢境界の近傍 |",
        "|---|---:|---:|---:|---:|"])
    for r in summary:
        interval = (f"{r['near_zero_min']:+.3g} ～ {r['near_zero_max']:+.3g}" if r["near_zero"] else "該当なし")
        lines.append(f"| {r['condition']} | {r['asym_exact_zero']} | "
                     f"{r['near_zero']}（{r['near_zero_positive']}・{r['near_zero_negative']}） | "
                     f"{interval} | {r['dominance_gap_near_zero']} |")
    lines.extend(["", "各条件で増加・減少・完全なゼロの合計、および分類図の5区分の合計は1,024。",
                  "全条件の0始まりLayer・Headキー、保存済み平均値と優勢件数、入力ハッシュを照合した。",
                  "", "## 固定paper_heads", "",
                  "各セルは対称／反対称の符号付き平均差。", "",
                  "| Head | A | B | C1 | C2 |", "|---|---:|---:|---:|---:|"])
    indexed = paper.set_index(["condition", "layer", "head"])
    for l, h in PAPER:
        values = ["／".join(f"{v:+.6f}" for v in indexed.loc[(code,l,h), [SYM, ASYM]]) for code in LABELS]
        lines.append(f"| L{l} H{h} | " + " | ".join(values) + " |")
    lines.extend(["", "## 解釈", "",
        "各ペアでは R_sym² + R_asym² = 1 なので、変化があれば比率は逆方向に動く。",
        "ペア内の変化幅の大小も両入力の比率に制約される。サンプル間で符号付き平均を取った後の",
        "優勢・増減の件数は、この恒等式だけでは決まらない。優勢は変化幅、増加は変化方向を表す。",
        "入力や比較対象が異なるため、asym増加をJailbreakへの因果的寄与とは解釈できない。", "",
        "[割合の棒グラフ](asym_head_fractions.png) · [共通スケールのヒートマップ](asym_signed_heatmaps.png) · "
        "[5区分の分類マップ](head_classification_maps.png)", "",
        "[Head別CSV](heads.csv) · [比較サマリーCSV](summary.csv) · [比較サマリーJSON](summary.json) · "
        "[固定5 Head CSV](paper_heads.csv) · [検証結果](validation.json) · [入力のProvenance](manifest.json)", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    out = parser.parse_args().output_dir
    if out.exists():
        raise FileExistsError(out)
    heads, summary, hashes = read_sources()
    paper = heads.set_index(["condition","layer","head"]).loc[
        [(code,l,h) for code in LABELS for l,h in PAPER], [SYM, ASYM]].reset_index()
    if any(sha256_file(path) != expected for path, expected in hashes.items()):
        raise ValueError("source changed during analysis")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/"summary.csv", pd.DataFrame(summary))
    write_csv(out/"heads.csv", heads)
    write_csv(out/"paper_heads.csv", paper)
    write_json(out/"summary.json", dict(conditions=summary, near_zero_threshold=TOL,
                                       map_classes=dict(enumerate(CLASSES))))
    write_json(out/"validation.json", dict(status="passed", input_hashes_match=True, zero_based_grid=True,
                 expected_dominance_counts={k: v[-1] for k,v in LABELS.items()},
                 observed_dominance_counts={r["condition"]: r["asym_dominant"] for r in summary},
                 head_denominators={r["condition"]: r["n_heads"] for r in summary},
                 sign_and_map_partition_counts_checked=True, prior_means_match=True))
    write_json(out/"manifest.json", dict(created_at=utc_now(), git_commit=git("rev-parse", "HEAD"),
                 git_branch=git("branch", "--show-current"), git_status=git("status", "--short"),
                 input_sha256=hashes, script_sha256=sha256_file(__file__),
                 model="Llama-2-7b-chat-hf", device="CPU", model_forward=False,
                 aggregation="reuse saved signed mean(Attack − Control) per Head",
                 near_zero_threshold=TOL, numpy=np.__version__, pandas=pd.__version__,
                 matplotlib=matplotlib.__version__))
    plot(out, heads, summary)
    with (out/"report.md").open("x", encoding="utf-8") as handle:
        handle.write(report(summary, paper))
    write_json(out/"output_hashes.json", {p.name: sha256_file(p) for p in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
