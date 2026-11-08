#!/usr/bin/env python3
"""Create an additive, local results catalog; never move/modify legacy artifacts."""
from pathlib import Path
import argparse
import ast
import html
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from utils.artifact_io import read_json,sha256_file,utc_now
from utils.gcg_shuffled_common import write_json,write_csv
import pandas as pd

LEGACY=[
 ('00_inputs/comparison_dataset','shuffled_gcg_comparison/legacy_selection20_20261106T171044Z','20 matched Original/GCG/Shuffled inputs','scripts/build_shuffled_gcg_comparison.py','plan.json'),
 ('00_inputs/raw_attention_scores','full_attention_scores/legacy_selection20_20261107T053240Z','60 saved FP32 post-RoPE raw scores','scripts/extract_full_attention_scores.py','plan.json'),
 ('10_existing_attention/four_metrics','gcg_shuffled_four_metrics/legacy_selection20_20261107T060324Z','final Query and UU four metrics','scripts/analyze_gcg_shuffled_four_metrics.py','manifest.json'),
 ('10_existing_attention/token_position','gcg_shuffled_token_analysis/legacy_selection20_20261107T075659Z','fixed Head selection and token diagnostics','scripts/analyze_gcg_shuffled_tokens.py','manifest.json'),
 ('20_region_scores/rms_and_frobenius','gcg_shuffled_region_analysis/legacy_selection20_20261107T094154Z','OO/UU/UO norms and RMS','scripts/analyze_gcg_shuffled_regions.py','manifest.json'),
 ('20_region_scores/signed_mean','gcg_shuffled_signed_means/legacy_selection20_20261108T_signed_mean_v1','OO/UU/UO signed means','scripts/analyze_gcg_shuffled_signed_means.py','manifest.json'),
 ('00_inputs/pilot5_reference','shuffled_gcg_pilot/legacy_selection20_seed42_20261106T165429Z','historical pilot5, provenance only; not new analysis','scripts/evaluate_shuffled_gcg_pilot.py','plan.json')]


def safe_link(link,target):
    if not target.is_dir():raise FileNotFoundError(target)
    if link.is_symlink():
        if link.resolve()!=target.resolve():raise ValueError('existing alias has different target')
    elif link.exists():raise FileExistsError(link)
    else:
        link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=True)
    if not link.is_dir() or link.resolve()!=target.resolve():raise ValueError('invalid legacy reference')


def describe_run(path,alias,purpose,script,metadata):
    meta=read_json(path/metadata);fields=['created_at','created_utc','git_commit','model_name','dataset','dataset_path','primary_indices','all_indices','selected_indices','source_hashes','dataset_sha256']
    return dict(alias=alias,run_id=path.name,purpose=purpose,source_path=str(path.resolve()),script=script,
        metadata=str((path/metadata).resolve()),metadata_sha256=sha256_file(path/metadata),
        recorded_provenance={key:meta.get(key,'unknown') for key in fields},
        csvs=[str(p.relative_to(path)) for p in sorted(path.rglob('*.csv'))],
        figures=[str(p.relative_to(path)) for p in sorted(path.rglob('*.png'))],
        validations=[str(p.relative_to(path)) for p in sorted(path.glob('*validation*.json'))],
        manifests=[str(p.relative_to(path)) for p in sorted(path.glob('*manifest*'))],
        hash_files=[str(p.relative_to(path)) for p in sorted(path.glob('*hash*.json'))])


def build(root,run_id):
    if (root/'index.json').exists():raise FileExistsError('catalog exists; use explicit separate version instead of overwriting')
    records=[]
    for alias,relative,purpose,script,metadata in LEGACY:
        target=ROOT/'data/outputs'/relative;safe_link(root/alias,target)
        records.append(describe_run(target,alias,purpose,script,metadata))
    for stage,purpose in [('30_sign_decomposition','signed score energy, sign fractions and per-Head quantiles'),('40_query_softmax','same causal Suffix20/closing4 queries, per-Head softmax'),('50_row_centered','visible row offset and relative regional score change'),('60_integrated_analysis','old RMS/signed mean and new metrics; associations')]:
        path=root/stage/run_id
        records.append(describe_run(path,str(path.relative_to(root)),purpose,'scripts/analyze_gcg_shuffled_additional.py','manifest.json'))
    code=[];dependencies=set()
    for folder in ('scripts','utils','tests','docs'):
        for p in sorted((ROOT/folder).glob('*')):
            if not p.is_file() or not any(k in p.name for k in ('gcg_shuffled','shuffled_gcg','full_attention','gcg_vs_shuffled')):continue
            imports=[]
            if p.suffix=='.py':imports=[n.module for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.ImportFrom) and n.module]
            dependencies.update(i for i in imports if i.startswith(('scripts.','utils.')))
            code.append(dict(path=str(p.relative_to(ROOT)),sha256=sha256_file(p),imports=imports,
                             action='inventory only for historical generation/extraction; otherwise phase-scoped analysis'))
    owned={r['path'].removesuffix('.py').replace('/','.') for r in code}
    excluded=sorted(dependencies-owned)
    write_json(root/'index.json',dict(created_at=utc_now(),experiment='legacy_selection20 GCG vs Shuffled',new_run_id=run_id,
        old_outputs_policy='relative symlink only; original roots remain immutable',runs=records,code_inventory=code,
        imported_support_not_refactored=excluded,excluded_scope='other model/attack/intervention/defense phases; independent100 remains untouched',
        validation=str(root/'90_validation'/run_id)))
    write_csv(root/'runs.csv',pd.DataFrame([{k:r[k] for k in ('alias','run_id','purpose','source_path','script','metadata')} for r in records]))
    write_csv(root/'code_inventory.csv',pd.DataFrame(code))
    lines=['# GCG vs Shuffled: legacy_selection20','',
      'Llama-2-7b-chat-hfの同じ20入力ペア。主分析primary18、全20件は補助。Head群は既存の固定群。自動拒否表現判定であり、人手確認済み攻撃成功・因果・有意性・一般化は主張しない。',
      '既存成果物は元の場所に残し、00/10/20は相対シンボリックリンク。30〜60は今回の実出力。50は同一Query表を40から参照し重複しない。', '',
      '| 知りたいこと | 分析・場所 |','|---|---|',
      '| 最終QueryのSuffix Attention割合 | 10_existing_attention/four_metrics、token_position |',
      '| RegionのScore規模・符号付き水準 | 20_region_scores/rms_and_frobenius、signed_mean |',
      '| RMS変化の正負別寄与 | 30_sign_decomposition |',
      '| 同じSuffix QueryからのAttention配分 | 40_query_softmax |',
      '| 行全体の水準とKey間の相対差 | 50_row_centered |',
      '| 統合結果とMTG要約 | 60_integrated_analysis |',
      '| 互換性・入力不変・テスト監査 | 90_validation |','',
      '## Run一覧','', '| 参照先 | 目的 | 入口 |','|---|---|---|']
    for r in records:lines.append(f"| [{r['alias']}]({r['alias']}/) | {r['purpose']} | `{r['script']}` |")
    lines += ['', '各runのmetrics.csvはサンプル/条件/Layer/Head別、head_summary_18/20.csvはHead集計、view_summaryは固定群/個別Head、sample_summaryはサンプル別。overview.csvは全1024 Head等重み集計。q4とsuffix_mean20は別Query集合である。',
        '元Scoreの形状は[32,32,L,L] FP32、FP64でCPU計算。Fullは未来を含み、CausalはKey<=Query（対角含む）。Norm/Softmax/分位点などをHeadごとに計算してから平均する。',
        '詳細はindex.json（入力ハッシュ・対応・時刻・commit・コード依存）、runs.csv、code_inventory.csv。画像一覧はfigures.md / figures.html。',
        '', '## 再現', '```bash',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/analyze_gcg_shuffled_additional.py --run-id NEW_RUN',
        '/home/h.kawakami/miniconda3/envs/harmbench-blackwell/bin/python -B scripts/audit_gcg_shuffled_compatibility.py --before BEFORE --after AFTER --output NEW_AUDIT.json','```',
        'NEW_RUN/NEW_AUDITは新規パス。旧CLIと旧出力パスは維持。索引はscripts/index_gcg_shuffled_phase.py（初回作成専用）で再構成できる。既存カタログを上書きせず、改訂は別パスで版管理する。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    figures=['# 画像一覧','', '図のタイトルとファイル名から対象Query・領域・Maskを確認する。新規analysisの図は主要18件が中心。',''];htmls=['<!doctype html><meta charset="utf-8"><title>GCG vs Shuffled figures</title><h1>画像一覧</h1>']
    for r in records:
        figures += ['## '+r['alias'],'']
        for name in r['figures']:
            link=r['alias']+'/'+name
            figures.append(f'- [{name}]({link})')
            htmls.append(f'<p>{html.escape(r["alias"])} / <a href="{html.escape(link)}">{html.escape(name)}</a></p><a href="{html.escape(link)}"><img src="{html.escape(link)}" loading="lazy" style="max-width:600px"></a>')
        figures.append('')
    (root/'figures.md').write_text('\n'.join(figures));(root/'figures.html').write_text('\n'.join(htmls))
    for r in records:
        for name in r['csvs']+r['figures']:
            if not (root/r['alias']/name).is_file():raise ValueError('catalog broken reference')
    write_json(root/'catalog_validation.json',dict(status='passed',runs=len(records),code_files=len(code),legacy_links=len(LEGACY),all_figure_csv_links_resolve=True,
        existing_directories_moved=False,old_paths_preserved=True,large_artifacts_copied=False))
    print('Catalog:',root,'runs:',len(records),'files:',len(code))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/outputs/gcg_vs_shuffled/legacy_selection20');p.add_argument('--run-id',required=True)
    a=p.parse_args();build(a.root,a.run_id)
