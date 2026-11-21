#!/usr/bin/env python3
"""Compare paired signed Tail20 Norm-ratio changes from saved Llama2 artifacts."""

import argparse
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-llama2-tail20")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.prepare_gcg_repro_data import build_pairs, EXPECTED_LLAMA2_FALLBACK_INDICES
from utils.artifact_io import read_json, read_jsonl, sha256_file, utc_now
from utils.gcg_shuffled_common import DATASET, SCORES, inspect_inputs, write_csv, write_json
from utils.prompt_utils import apply_template

OUT = ROOT/"data/outputs"
RAW = ROOT/"data/raw"
DERIVED = ROOT/"data/derived/llama2"
A_DIR = OUT/"llama2/analysis"
B_DIR = OUT/"llama2/attention_scores/gcg_tail20_asym_lower"
TOKENIZER = Path("/mnt/mp_nas_mks/labmember/h.kawakami/models/Llama-2-7b-chat-hf")
PAPER = [(3, 0), (7, 16), (9, 9), (13, 4), (3, 5)]
LABELS = dict(A="Original vs AutoDAN", B="Original vs GCG", C="GCG vs Shuffled")
KEYS = ["prompt_id", "layer", "head"]
NAMES = ["mean_delta_R_sym", "mean_delta_R_asym", "abs_mean_asym_minus_sym"]
RATIO_TOL = 1e-6
HEAD_TOL = 1e-7


def tracked(paths, hashes):
    for path in paths:
        hashes[str(path)] = sha256_file(path)


def prompt_sources(code, hashes):
    attack = {"A": "autodan", "B": "gcg"}[code]
    raw = RAW/"llama2"/{"A": "AutoDAN_llama2.jsonl", "B": "GCG_llama2.jsonl"}[code]
    derived = [DERIVED/attack/f"{c}_prompts.jsonl" for c in ("original", "jailbreak")]
    adv_path = RAW/"advbench.jsonl"
    tracked([raw, adv_path, *derived], hashes)
    adv, source = read_jsonl(adv_path), read_jsonl(raw)
    by_index = {r["index"]: r for r in adv}
    original, jailbreak = [read_jsonl(p) for p in derived]
    if code == "B":
        expected_o, expected_j, audit = build_pairs("llama2", source, adv, by_index, fallback_offset=60)
        if (original != expected_o or jailbreak != expected_j or audit["prefix_matched"] != 157
                or audit["fallback_advbench_indices"] != EXPECTED_LLAMA2_FALLBACK_INDICES):
            raise ValueError("GCG derived mapping differs from documented pairing")
    else:
        if ([r["prompt"] for r in original] != [by_index[r["index"]]["prompt"] for r in source]
                or [r["prompt"] for r in jailbreak] != [r["prompt"] for r in source]):
            raise ValueError("AutoDAN raw/derived pairing differs")
        audit = dict(source_index_in_csv="blank; verified present-day JSONL order and token lengths")
    return (original, jailbreak), [r["index"] for r in (jailbreak if code == "B" else source)], audit


def historical_csv(code, tokenizer, hashes):
    prompts, source_indices, provenance = prompt_sources(code, hashes)
    n = {"A": 130, "B": 160}[code]
    if len(source_indices) != n or len(set(source_indices)) != n:
        raise ValueError("unexpected pair count")
    if code == "A":
        files = [A_DIR/f"attention_score_metrics_{c}130_all_heads_tail20.csv"
                 for c in ("original", "jailbreak")]
        logs = [ROOT/"logs"/f"analyze_attention_scores_{c}130_all_heads_tail20.log"
                for c in ("original", "jailbreak")]
        summary = A_DIR/"summary_all_heads_vs_paper_heads_tail20.csv"
        asym = "skew"
    else:
        files = [B_DIR/f"{c}_tail20.csv" for c in ("original", "jailbreak")]
        logs = [ROOT/"logs"/f"llama2_gcg_{c}160_tail20_asym_lower.log"
                for c in ("original", "jailbreak")]
        summary = B_DIR/"summary_paper_heads_tail20.csv"
        asym = "asym"
    tracked([*files, *logs, summary], hashes)
    grid = pd.MultiIndex.from_product([range(n), range(32), range(32)], names=KEYS)
    ratios, legacy, lengths = [], [], []
    for path, log, records in zip(files, logs, prompts):
        lines = log.read_text().splitlines()
        if lines[-2:] != [f"wrote rows: {n*1024}", f"output_csv: {path.relative_to(ROOT)}"]:
            raise ValueError(f"unexpected log destination or count: {log}")
        frame = pd.read_csv(path, float_precision="round_trip").set_index(KEYS, verify_integrity=True).sort_index()
        if not frame.index.equals(grid) or not frame.window_type.eq("tail_20").all():
            raise ValueError(f"invalid Head grid/window: {path}")
        formatted = apply_template([r["prompt"] for r in records], None, tokenizer, {"model_name": "Llama"})
        tokens = tokenizer(formatted, add_special_tokens=False)["input_ids"]
        length = np.asarray([len(ids) for ids in tokens])
        np.testing.assert_array_equal(frame.seq_len, np.repeat(length, 1024))
        np.testing.assert_array_equal(frame.window_size, np.repeat(np.minimum(length, 20), 1024))
        if code == "B":
            np.testing.assert_array_equal(frame.source_index, np.repeat(source_indices, 1024))
        elif not frame.source_index.isna().all():
            raise ValueError("unexpected historical AutoDAN source_index format")
        norms = frame[["fro_norm", "sym_norm", f"{asym}_norm"]].to_numpy()
        if not np.isfinite(norms).all() or (norms[:, 0] <= 0).any() or (norms < 0).any():
            raise ValueError(f"invalid saved norms: {path}")
        ratios.append((norms[:, 1:]/norms[:, :1]).reshape(n, 1024, 2))
        legacy.append(frame[["sym_ratio", f"{asym}_ratio"]].to_numpy().reshape(n, 1024, 2))
        lengths.append(length)

    reconstructed = np.asarray(ratios)
    stored = np.asarray(legacy)
    np.testing.assert_allclose(reconstructed, stored, rtol=0, atol=RATIO_TOL)
    historical = pd.read_csv(summary).set_index(["condition", "group"])
    groups = dict(all_heads=list(range(1024)), selected_heads_mean=[32*l+h for l,h in PAPER])
    groups.update({f"layer{l}_head{h}": [32*l+h] for l,h in PAPER})
    stored_delta = stored[1]-stored[0]
    legacy_error = 0.0
    for group, heads in groups.items():
        old = historical.loc[("delta_jailbreak_minus_original", group)]
        observed = stored_delta[:, heads].mean(axis=(0, 1))
        expected = old[["delta_sym_ratio_mean", f"delta_{asym}_ratio_mean"]].to_numpy(dtype=float)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
        legacy_error = max(legacy_error, float(abs(observed-expected).max()))
    delta_error = (reconstructed[1]-reconstructed[0]).mean(axis=0)-stored_delta.mean(axis=0)
    return reconstructed, lengths, source_indices, dict(
        **provenance, csv=[str(p) for p in files], logs=[str(p) for p in logs],
        reference_summary=str(summary), token_length_matches_current_local_tokenizer=True,
        historical_exact_command_and_model_tokenizer_revision="unknown",
        historical_summary_reproduction_max_error=legacy_error,
        historical_ratio_per_row_max_error=float(abs(reconstructed-stored).max()),
        historical_vs_reconstructed_head_delta_max_error=float(abs(delta_error).max()),
    )


def shuffled_scores(hashes):
    _, primary, refs = inspect_inputs()
    primary = sorted(primary)
    tracked([DATASET, DATASET.parent/"plan.json", SCORES/"plan.json", SCORES/"manifest.jsonl",
             SCORES/"validation_final.json"], hashes)
    plan = read_json(SCORES/"plan.json")
    if (plan["model_name"] != "meta-llama/Llama-2-7b-chat-hf"
            or not plan["computation"].startswith("post-RoPE")):
        raise ValueError("unexpected primary18 Score provenance")
    by_key = {(r["dataset_index"], r["condition"]): r for r in refs}
    ratios, lengths = [], []
    for condition in ("shuffled", "gcg"):
        values, sizes = [], []
        for index in primary:
            ref = by_key[index, condition]
            hashes[ref["path"]] = ref["sha256"]
            hashes[ref["metadata_path"]] = ref["metadata_sha256"]
            score = np.load(ref["path"], mmap_mode="r", allow_pickle=False)
            length = score.shape[-1]
            sizes.append(length)
            k = min(20, length)
            b = np.asarray(score[..., -k:, -k:], dtype=np.float64)
            if not np.isfinite(b).all():
                raise ValueError("nonfinite Tail20")
            norm = np.linalg.norm(b, axis=(-2, -1))
            if (norm <= 0).any():
                raise ValueError("zero Tail20 norm")
            t = b.swapaxes(-1, -2)
            values.append(np.stack([np.linalg.norm(x, axis=(-2, -1))/norm
                                    for x in ((b+t)/2, (b-t)/2)], axis=-1).reshape(1024, 2))
        ratios.append(values)
        lengths.append(np.asarray(sizes))
    np.testing.assert_array_equal(*lengths)
    return np.asarray(ratios), lengths, primary, dict(
        score_extraction_commit=plan["git_commit"], model_revision=plan["model_revision"],
        tokenizer=plan["tokenizer_info"], computation=plan["computation"],
        conditions=["shuffled", "gcg"], suffix_UU_used=False,
    )


def analyze(code, ratios, lengths, indices):
    if not np.isfinite(ratios).all():
        raise ValueError(f"nonfinite ratio: {code}")
    identity = np.abs(np.sum(ratios**2, axis=-1)-1)
    delta = ratios[1]-ratios[0]
    residual = np.abs((delta*ratios.sum(axis=0)).sum(axis=-1))
    tolerance = 1e-12 if code == "C" else RATIO_TOL
    np.testing.assert_allclose(identity, 0, atol=tolerance, rtol=0)
    np.testing.assert_allclose(residual, 0, atol=tolerance, rtol=0)
    averages = delta.mean(axis=0)
    gap = abs(averages[:, 1])-abs(averages[:, 0])
    opposite = averages[:, 0]*averages[:, 1] < 0
    frame = pd.DataFrame(dict(condition=code, layer=np.repeat(range(32), 32), head=np.tile(range(32), 32),
                              n_pairs=len(indices), mean_delta_R_sym=averages[:, 0],
                              mean_delta_R_asym=averages[:, 1], abs_mean_asym_minus_sym=gap,
                              asym_dominant=gap > 0, opposite_sign=opposite))
    sample_rows = pd.DataFrame(dict(
        condition=code, prompt_id=range(len(indices)), source_index=indices,
        control_length=lengths[0], attack_length=lengths[1],
        control_window=np.minimum(20, lengths[0]), attack_window=np.minimum(20, lengths[1])))
    summary = dict(
        pair_count=len(indices), mean_delta_R_sym=float(averages[:, 0].mean()),
        mean_delta_R_asym=float(averages[:, 1].mean()), asym_dominant=int((gap > 0).sum()),
        sym_dominant=int((gap < 0).sum()), tied=int((gap == 0).sum()),
        asym_dominant_fraction=float((gap > 0).mean()), opposite_sign=int(opposite.sum()),
        opposite_sign_fraction=float(opposite.mean()),
        asym_dominant_at_1e_7=int((gap > HEAD_TOL).sum()),
        head_gaps_within_1e_7=int((abs(gap) <= HEAD_TOL).sum()),
        opposite_sign_at_1e_7=int((opposite & (abs(averages) > HEAD_TOL).all(axis=1)).sum()),
        paired_lengths_equal=int((lengths[0] == lengths[1]).sum()),
        lengths={role: dict(min=int(a.min()), max=int(a.max()), mean=float(a.mean()),
                            shorter_than_20=int((a < 20).sum()))
                 for role, a in zip(("control", "attack"), lengths)},
    )
    validation = dict(ratio_identity_max_error=float(identity.max()),
                      weighted_delta_identity_max_error=float(residual.max()),
                      identity_tolerance=tolerance, head_sensitivity_tolerance=HEAD_TOL)
    return frame, sample_rows, summary, validation


def make_plots(out, heads, paper, summary):
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), layout="constrained")
    common = heads[NAMES[:2]].abs().to_numpy().max()
    gap_limit = heads[NAMES[2]].abs().max()
    for j, title in enumerate(("Mean ΔR_sym", "Mean ΔR_asym", "|Mean ΔR_asym| − |Mean ΔR_sym|")):
        limit = common if j < 2 else gap_limit
        for i, code in enumerate(LABELS):
            data = heads.loc[heads.condition == code, NAMES[j]].to_numpy().reshape(32, 32)
            plot = axes[i, j].imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            axes[i, j].set(title=f"{code} · {title}", xlabel="Head", ylabel="Layer")
        fig.colorbar(plot, ax=axes[:, j], shrink=.82)
    fig.savefig(out/"head_heatmaps.png", dpi=200)
    plt.close(fig)

    colors = ["#4477AA", "#228866", "#AA4477"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.bar(list(LABELS), [summary[k]["asym_dominant_fraction"]*100 for k in LABELS], color=colors)
    ax.bar_label(bars, labels=[f"{summary[k]['asym_dominant_fraction']:.2%}" for k in LABELS])
    ax.set(ylabel="Heads (%)", ylim=(0, 108), title="Heads with |Mean ΔR_asym| > |Mean ΔR_sym|")
    fig.savefig(out/"asym_dominant_fraction.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    x = np.arange(5)
    for ax, code in zip(axes, LABELS):
        rows = paper[paper.condition == code]
        ax.bar(x-.18, rows[NAMES[0]], .36, label="Sym", color="#4477AA")
        ax.bar(x+.18, rows[NAMES[1]], .36, label="Asym", color="#CC6677")
        ax.axhline(0, color="0.3", linewidth=.8)
        ax.set(xticks=x, xticklabels=[f"L{l} H{h}" for l,h in PAPER], title=f"{code} · {LABELS[code]}")
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("Mean ΔR (Attack − Control)")
    axes[0].legend()
    fig.savefig(out/"paper_heads.png", dpi=200)
    plt.close(fig)


def report(summary, overlaps, pairwise, paper, validation):
    results, lengths, fixed, checks = [], [], [], []
    for code, r in summary.items():
        results.append(f"| {code} | {r['pair_count']} | {r['mean_delta_R_sym']:+.8f} | "
                       f"{r['mean_delta_R_asym']:+.8f} | {r['asym_dominant']}/1024 "
                       f"({r['asym_dominant_fraction']:.2%}) | {r['opposite_sign']}/1024 "
                       f"({r['opposite_sign_fraction']:.2%}) |")
        for role, d in r["lengths"].items():
            lengths.append(f"| {code} | {role} | {d['min']}–{d['max']} | {d['mean']:.3f} | {d['shorter_than_20']} |")
        v = validation[code]
        legacy = v.get("historical_summary_reproduction_max_error")
        restored = v.get("historical_vs_reconstructed_head_delta_max_error")
        checks.append(f"| {code} | {v['ratio_identity_max_error']:.3g} | "
                      f"{v['weighted_delta_identity_max_error']:.3g} | "
                      f"{'—' if legacy is None else format(legacy, '.3g')} | "
                      f"{'—' if restored is None else format(restored, '.3g')} | "
                      f"{r['asym_dominant_at_1e_7']} | {r['opposite_sign_at_1e_7']} |")
    for row in paper.itertuples(index=False):
        fixed.append(f"| {row.condition} | L{row.layer} H{row.head} | "
                     f"{row.mean_delta_R_sym:+.8f} | {row.mean_delta_R_asym:+.8f} |")
    overlap_text = "\n".join(
        f"| {r.A} | {r.B} | {r.C} | {r.head_count} |" for r in overlaps.itertuples(index=False))
    pair_text = "\n".join(
        f"| {r.conditions} | {r.both} | {r.first_only} | {r.second_only} | {r.neither} |"
        for r in pairwise.itertuples(index=False))
    return f"""# Llama-2-7b-chat-hfのTail20 Norm比比較

AはAutoDAN−Original、BはGCG−Original、CはGCG−Shuffled。すべて32 Layer×32 Head。
各ペアでNorm比の差を計算し、Head別に符号付きで平均した。
判定の絶対値は平均の後に取った。分母は全体Frobenius Normで、追加の系列長除算は行っていない。

| 条件 | ペア数 | 全Head平均ΔR_sym | 全Head平均ΔR_asym | 反対称優勢Head | 逆符号Head |
|---|---:|---:|---:|---:|---:|
{chr(10).join(results)}

## 反対称優勢Headの一致

1はその条件で |mean ΔR_asym| > |mean ΔR_sym|。HeadのIDは [head_membership.csv](head_membership.csv) に収録。

| A | B | C | Head数 |
|---:|---:|---:|---:|
{overlap_text}

| 条件 | 両方優勢 | 前者のみ | 後者のみ | どちらも優勢でない |
|---|---:|---:|---:|---:|
{pair_text}

## 固定paper_heads

| 条件 | Head | mean ΔR_sym | mean ΔR_asym |
|---|---|---:|---:|
{chr(10).join(fixed)}

## 系列長と入力対応

| 条件 | 役割 | Token数範囲 | 平均 | 20未満の入力数 |
|---|---|---:|---:|---:|
{chr(10).join(lengths)}

RoPE後・Mask前・Softmax前のScoreを対象とし、対角と未来方向も含む。
窓幅は min(20, L)。A・Bはこの窓で保存済みの成分Normを使い、比率をFP64で再計算した。
Cは既存primary18の保存済みFull Scoreから末尾の行と列を抽出した。Suffix位置で定義するUUとは別領域。
CのGCGとShuffledは全18ペアで実Token数が一致。A・BもローカルTokenizerで再度Token化し、
保存CSVの系列長・窓幅と全入力で一致した。新規モデル推論は行っていない。

Bは両CSV各163,840行。prompt_idとsource_indexの対応、全Headグリッド、元JSONLと派生JSONLを検証した。
AdvBench index 60〜219の160ペアで、157件は最長prefix一致、63・93・105は既存のraw index + 60規則による
文言差の補完。ログは両CSVの出力先と行数を記録し、既存集計とも一致した。
当時の実行コマンド全文、重み・Tokenizerの正確なrevision、生成時の入力ハッシュは未確認。
Aの歴史CSVではsource_indexが空欄のため、現在の元データ・派生JSONLの順序、再Token化した系列長、
保存ログと集計の一致を確認した。6月22日のMTG資料そのものとの照合は未確認。

## 検証と解釈

| 条件 | 二乗和の最大誤差 | ペア恒等式の最大残差 | 保存比率から旧集計を復元した最大差 | 生Normから再計算したHead平均差と保存比率との差 | 1e-7閾値の反対称優勢数 | 1e-7閾値の逆符号数 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(checks)}

各ペアで R_sym² + R_asym² = 1。これを両条件で引くと、
ΔR_sym × (R_sym,attack + R_sym,control) + ΔR_asym × (R_asym,attack + R_asym,control) = 0。
個々のペアでは変化方向が逆で、両条件の比率の和が小さい成分ほど絶対変化が大きい。
重みはサンプルによって異なるので、符号付き平均後の大小・符号・Headの一致数は観測結果。
A・B・Cでは入力集合と比較対象が異なり、Head割合の差から攻撃手法の因果効果は特定できない。

[Head別CSV](heads.csv) · [サンプル別の長さ](samples.csv) · [全体集計](summary.json) ·
[検証と来歴](validation.json) · [ハッシュと設定](manifest.json)

[3条件のヒートマップ](head_heatmaps.png)（±の色を対称化、sym/asymは全条件で共通スケール） ·
[反対称優勢割合](asym_dominant_fraction.png) · [固定5 Headの棒グラフ](paper_heads.png)
"""


def main():
    from transformers import AutoTokenizer, __version__ as transformers_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        raise FileExistsError(out)
    hashes = {}
    tokenizer_files = sorted(p for p in TOKENIZER.iterdir() if p.is_file() and (
        p.name.startswith(("tokenizer", "special_tokens", "added_tokens", "chat_template"))
        or p.name == "config.json"))
    tracked(tokenizer_files, hashes)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    tools_used = [Path(__file__), ROOT/"utils/gcg_shuffled_common.py", ROOT/"utils/artifact_io.py",
                  ROOT/"utils/prompt_utils.py", ROOT/"scripts/prepare_gcg_repro_data.py"]
    tracked(tools_used, hashes)
    frames, samples, summary, validation = [], [], {}, {}
    for code in LABELS:
        inputs = shuffled_scores(hashes) if code == "C" else historical_csv(code, tokenizer, hashes)
        ratios, lengths, indices, provenance = inputs
        frame, sample, summary[code], validation[code] = analyze(code, ratios, lengths, indices)
        validation[code].update(provenance)
        frames.append(frame)
        samples.append(sample)
    heads = pd.concat(frames, ignore_index=True)
    paper = heads.set_index(["condition", "layer", "head"]).loc[
        [(code, l, h) for code in LABELS for l, h in PAPER]].reset_index()
    membership = heads.pivot(index=["layer", "head"], columns="condition", values="asym_dominant").astype(int)
    patterns = pd.MultiIndex.from_product([[0, 1]]*3, names=list(LABELS))
    overlaps = membership.value_counts().reindex(patterns, fill_value=0).rename("head_count").reset_index()
    pairwise = pd.DataFrame([
        dict(conditions=a+b, both=int(((membership[a]==1)&(membership[b]==1)).sum()),
             first_only=int(((membership[a]==1)&(membership[b]==0)).sum()),
             second_only=int(((membership[a]==0)&(membership[b]==1)).sum()),
             neither=int(((membership[a]==0)&(membership[b]==0)).sum()))
        for a,b in combinations(LABELS, 2)])
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError("input artifact changed")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    manifest = dict(
        created_at=utc_now(), git_commit=git("rev-parse", "HEAD"), git_branch=git("branch", "--show-current"),
        git_status=git("status", "--short"), june_aggregation_commit=git("rev-parse", "d5be824"),
        input_and_code_sha256=hashes, labels=LABELS, model="Llama-2-7b-chat-hf",
        local_tokenizer=str(TOKENIZER), transformers=transformers_version, numpy=np.__version__,
        pandas=pd.__version__, matplotlib=matplotlib.__version__, device="CPU", model_forward=False,
        window="last min(20,L) queries and keys, pre-mask pre-softmax post-RoPE",
        ratio="unsquared component Frobenius norm / block Frobenius norm",
        aggregation="signed paired differences, sample mean per Head, then magnitude comparison",
        paper_heads=PAPER,
    )
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in dict(heads=heads, samples=pd.concat(samples, ignore_index=True),
                            paper_heads=paper, head_membership=membership.reset_index(),
                            overlap_patterns=overlaps, overlap_pairwise=pairwise).items():
        write_csv(out/f"{name}.csv", frame)
    write_json(out/"summary.json", summary)
    write_json(out/"validation.json", validation)
    write_json(out/"manifest.json", manifest)
    make_plots(out, heads, paper, summary)
    with (out/"report.md").open("x", encoding="utf-8") as handle:
        handle.write(report(summary, overlaps, pairwise, paper, validation))
    write_json(out/"output_hashes.json", {p.name: sha256_file(p) for p in sorted(out.iterdir())})


if __name__ == "__main__":
    main()
