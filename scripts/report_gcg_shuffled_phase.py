#!/usr/bin/env python3
"""Build Japanese interpretation/MTG notes from saved tables, without inference."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from utils.artifact_io import read_json,sha256_file
from scripts.analyze_gcg_shuffled_four_metrics import write_json,write_csv
from utils.gcg_shuffled_plot_common import plt,finish


def build(root,run_id):
    p={k:root/v/run_id for k,v in [('sign','30_sign_decomposition'),('query','40_query_softmax'),('center','50_row_centered'),('integrated','60_integrated_analysis')]}
    m=read_json(p['integrated']/'manifest.json');o=pd.read_csv(p['integrated']/'overview.csv');a=o[o.subset==18].set_index('metric')
    q=pd.read_csv(p['query']/'overview.csv');q=q[q.subset==18]
    h=pd.read_csv(p['integrated']/'head_summary_18.csv');groups=pd.read_csv(p['query']/'view_summary_18.csv')
    corr=pd.read_csv(p['integrated']/'correlations.csv')
    # Keep exact numeric citations and supplementary direction quadrants.
    relations=[]
    for region in ('UU','UO'):
        for policy in ('full','causal'):
            x=h[h.metric==f'RMS_{region}_{policy}'][['layer','head','delta_mean']]
            y=h[h.metric==f'E_neg_{region}_{policy}'][['layer','head','delta_mean']]
            z=x.merge(y,on=['layer','head'],suffixes=('_rms','_neg'),validate='one_to_one')
            relations.append(dict(region=region,policy=policy,rms_up_negative_energy_up=int(((z.delta_mean_rms>0)&(z.delta_mean_neg>0)).sum()),head_count=len(z)))
    write_csv(p['integrated']/'energy_head_correspondence.csv',pd.DataFrame(relations))
    # Export suffix-position deltas in a compact, easy-to-cite table.
    positions=q[q['query'].str.startswith('suffix_') & (q['query']!='suffix_mean20')]
    write_csv(p['integrated']/'suffix_position_effects.csv',positions)
    # Additional centered-Key and paired Attention-delta diagnostic for the
    # same predetermined existing representative Heads as the original sheet.
    chosen=[]
    for v in m['views']:
        if v['kind']=='selected_group' and v['members'][0] not in chosen:chosen.append(v['members'][0])
        if len(chosen)==4:break
    index=m['primary_indices'][0];ref=next(r for r in m['score_references'] if r['dataset_index']==index and r['condition']=='gcg')
    ranges=ref['input']['token_ranges'];visible_end=ranges['suffix'][1];data=[]
    with np.load(p['query']/'row_profiles'/f'index_{index:03d}.npz') as z:
        fig,axes=plt.subplots(2,4,figsize=(18,8))
        for c,head in enumerate(chosen):
            vi=next(i for i,v in enumerate(m['views']) if v['members']==[head])
            for r,field in enumerate(('centered','weight')):
                g=z['gcg_'+field][vi,19,:visible_end];s=z['shuffled_'+field][vi,19,:visible_end]
                for k,(gg,ss) in enumerate(zip(g,s)):data.append(dict(dataset_index=index,layer=head//32,head=head%32,query='suffix_19',key=k,field=field,gcg=gg,shuffled=ss,delta=gg-ss))
                ax=axes[r,c]
                if field=='centered':
                    ax.plot(g,label='GCG',color='tab:red');ax.plot(s,label='Shuffled',color='tab:blue');ax.legend(fontsize=8)
                else:ax.plot(g-s,label='GCG-Shuffled',color='purple')
                for region in ('original','suffix'):ax.axvline(ranges[region][0]-.5,color='grey',lw=.6)
                ax.axhline(0,color='grey',lw=.6);ax.set(title=f'L{head//32} H{head%32}',xlabel='Visible Key position; prefix/O/U boundaries',ylabel='Centered score' if field=='centered' else 'Delta Attention fraction')
        finish(fig,p['integrated']/'figures/centered_key_diagnostics.png',f'Index {index}, n=1; suffix Query19 | predetermined old group-first Heads; illustration, no selection from new metrics')
    write_csv(p['integrated']/'centered_key_diagnostics.csv',pd.DataFrame(data))
    lines=['# GCG vs Shuffled：追加分析・統合考察','',
        '## 条件と計算',
        'Llama-2-7b-chat-hfの保存済みFP32 Score（RoPE後、Mask/Softmax前）。既存selection20、主要primary18、全1024 Headと従来の固定群を維持。差はGCG−Shuffled。CPU/FP64計算のみ。新しいモデル推論・抽出・Suffix・ASR・介入・独立100件は使用しない。',
        'サンプル/Head/Queryごとに特徴を計算し、等重みHead平均、その後サンプル平均。SDはサンプル差のddof=1。Fullは未来要素を含み、CausalはKey<=Query、対角を含む。',
        'Region Causal UUの平均は210要素等重みである一方、suffix_mean20は各Queryの指標を求めて20 Queryを等重み平均する。可視Suffix数がQueryごとに異なるため、両者は同じ平均ではない。',
        '', '## ② 正負別エネルギー：観測事実',
        f"Full UUのΔE_pos={a.loc['E_pos_UU_full','delta_mean']:+.6f}、ΔE_neg={a.loc['E_neg_UU_full','delta_mean']:+.6f}。Full UOではそれぞれ{a.loc['E_pos_UO_full','delta_mean']:+.6f}、{a.loc['E_neg_UO_full','delta_mean']:+.6f}。",
        '全Head/サンプル平均の二乗エネルギー増加は、ほぼ負Score側の増加に対応する。これは平均RMS²の差への加法分解であり、平均RMS差への割合分解ではない。各Headで正負どちらが寄与するかは一様ではない。',
        f"Signed MeanのFull UU差はmu_pos寄与{a.loc['mu_pos_UU_full','delta_mean']:+.6f}、mu_neg寄与{a.loc['mu_neg_UU_full','delta_mean']:+.6f}。UOでは{a.loc['mu_pos_UO_full','delta_mean']:+.6f}、{a.loc['mu_neg_UO_full','delta_mean']:+.6f}。負側の平均寄与低下が主で、UOの正側寄与は小さく増加している。",
        f"Causal UUもΔE_pos={a.loc['E_pos_UU_causal','delta_mean']:+.6f}、ΔE_neg={a.loc['E_neg_UU_causal','delta_mean']:+.6f}。Fullの未来側だけに限定された傾向ではない。OOは正負分解も完全一致。",
        '出典：[正負overview](../../30_sign_decomposition/'+run_id+'/overview.csv)、[正負寄与図](../../30_sign_decomposition/'+run_id+'/figures/sign_contributions.png)。正負割合・分位点・中央値は同stageのmetrics.csv/head_summaryに保存。分位点平均は混合分布分位点ではない。',
        '', '## ③ 同一QueryのSoftmax：観測事実',
        f"Suffix Query20位置平均のM_UはGCG {a.loc['suffix_mean20_M_U','gcg_mean']:.6f}、Shuffled {a.loc['suffix_mean20_M_U','shuffled_mean']:.6f}、差{100*a.loc['suffix_mean20_M_U','delta_mean']:+.4f}ポイント。17/18サンプルで低下した。",
        f"M_O差は{100*a.loc['suffix_mean20_M_O','delta_mean']:+.4f}ポイント、D差は{a.loc['suffix_mean20_D','delta_mean']:+.6f}（9増/9減）、条件付きM_U差は{100*a.loc['suffix_mean20_conditional_M_U','delta_mean']:+.4f}ポイント。DとM_Uの平均変化は非線形・分母・他領域の配分により同じ符号になるとは限らない。",
        'これは最終QueryのM_U +5.5594ポイント、M_O −1.7958ポイント、D +1.059337とは異なる。したがって「GCGでは全QueryがSuffixへより強くAttentionする」という一般化は支持されない。q1〜q4は保存済み値と照合した。位置比較はToken内容の対応ではない。',
        '出典：[Query overview](../../40_query_softmax/'+run_id+'/overview.csv)、[Suffix位置図](../../40_query_softmax/'+run_id+'/figures/suffix_query_profiles.png)、suffix_position_effects.csv。全可視Keyを分母に含め、HeadごとにSoftmaxを取っている。',
        '', '## ④ 行中心化：観測事実と解釈',
        f"Suffix Query平均のrow_mean差は{a.loc['suffix_mean20_row_mean','delta_mean']:+.6f}。中心化後のOriginal平均差は{a.loc['suffix_mean20_centered_O','delta_mean']:+.6f}、Suffix平均差は{a.loc['suffix_mean20_centered_U','delta_mean']:+.6f}、Suffix−Originalの相対平均差は{a.loc['suffix_mean20_gap_U_minus_O','delta_mean']:+.6f}。",
        f"q4ではrow_mean差{a.loc['q4_row_mean','delta_mean']:+.6f}に対し、中心化Original差{a.loc['q4_centered_O','delta_mean']:+.6f}、中心化Suffix差{a.loc['q4_centered_U','delta_mean']:+.6f}、相対差{a.loc['q4_gap_U_minus_O','delta_mean']:+.6f}。",
        'q4の変化はQuery行全体への共通定数シフトだけでは説明できず、OriginalとSuffixの相対的な変化が残る。Suffix内部Queryの中心化結果はq4とは異なる方向。中心化前後でSoftmaxとDは不変なので、中心化それ自体が配分を変えたわけではない。',
        '特定Keyが機構の原因かは確定していない。事前の固定群先頭Head・先頭primary Indexの例はcentered_key_diagnostics.csv/figures/centered_key_diagnostics.png。全20件の45個別Head・固定群・全Headのraw/centered/weight分布は40 stageのrow_profiles/。Original位置をサンプル間で無条件平均していない。',
        '', '## 固定Head群・サンプルの違い',
        '旧12群を維持し、新規指標でHeadを選び直していない。M_U増加群はq4 ΔM_U +37.6418ポイントだが、Suffix Query平均では+0.8832ポイント。D増加群はq4 +21.6162ポイントに対しSuffix平均−2.0297ポイント、F増加群はq4 +12.0859ポイントに対しSuffix平均−5.7498ポイント。全Head平均とも群間とも一様ではない。',
        '固定群の各5 Head、18/20集計、サンプル差SD・増減数は各stageのfixed_group_member_effects.csv、view_summary_18/20.csv、sample_summary.csv。正負エネルギーのHead局在はenergy_head_heatmaps、群内の不一致はenergy_fixed_groups、サンプル一貫性はenergy_samplesを参照。',
        '', '## 主要18と全20・統合表','', '| N | 指標 | GCG−Shuffled | サンプル差SD | 増/減 |','|---|---|---:|---:|---|']
    names=['RMS_UU_full','mu_UU_full','E_pos_UU_full','E_neg_UU_full','RMS_UO_full','mu_UO_full','E_pos_UO_full','E_neg_UO_full','suffix_mean20_M_U','suffix_mean20_M_O','suffix_mean20_D','suffix_mean20_gap_U_minus_O','q4_M_U','q4_D','q4_gap_U_minus_O']
    for row in o[o.metric.isin(names)].itertuples():lines.append(f'| {row.subset} | {row.metric} | {row.delta_mean:+.7f} | {row.delta_std:.7f} | {row.gcg_larger_count}/{row.shuffled_larger_count} |')
    lines += ['', '主要な方向は全20件でも維持される。ただし20件も同じ選択済みデータで独立検証ではない。',
        '## 相関と方向一致（記述のみ）','', '| N18単位 | X | Y | Pearson | Spearman | 同方向率 |','|---|---|---|---:|---:|---:|']
    for row in corr[corr.subset==18].itertuples():lines.append(f'| {row.unit} | {row.x} | {row.y} | {row.pearson:.4f} | {row.spearman:.4f} | {row.same_direction_fraction:.4f} |')
    lines+=['', 'correlations.csvとfigures/integrated_relationships.pngが出典。Head相関は依存した1024 Headの記述であり、標本数1024の独立検定ではない。サンプル単位は各サンプル内Head平均を先に取った18件の記述。Region全体とq4のQuery集合の違いを保持している。',
        '', '## 互換性・コード・出力整理',
        'utils/gcg_shuffled_common.pyに入力/ハッシュ/対応検証、CSV/JSON、OO/UU/UO/Mask、Head対応、view平均、ペア集計を抽出。旧関数名とCLIを再exportで保持。図共通設定はgcg_shuffled_plot_common.py。計算式・平均順序・Mask・Head群は変更していない。',
        '変更前コード、旧4入口の前後再実行、新規分析の前後再実行を90_validation/'+run_id+'/に保存。全20件の各サンプル/Head、18/20集計、固定群/個別、列/行順、dtype/shape、NaN、要素数、manifest科学条件を比較し、数値最大差0。旧CLI --helpも一致、57枚の既存/新規PNGは共通化前後で同一。日時・Git状態・コードhashのprovenance差は計算値の変更とは分けている。',
        '旧成果物は00/10/20から相対リンクするだけで移動・削除・上書きしていない。新規出力は30/40/50/60の実体、50のQuery表だけは40の同一表へ相対リンク。README、index.json、runs.csv、code_inventory.csv、figures.md/htmlで横断参照できる。',
        '', '## 限界と次の検証候補',
        '確認できたのは、同じ選択済みペアで負側エネルギー増加と平均低下が伴い、最終QueryとSuffix内部Queryでは配分変化が異なるという記述的事実。因果機構、実際の有害応諾、別攻撃/モデルへの普遍性は未確認。',
        'Scoreの正負はAttentionの強弱そのものではない。条件付き割合、全Key割合、D、平均Scoreは異なる量。平均・ノルム・分位点だけでは分布全体や特定Keyの原因を確定できない。',
        '次の候補（未実施）：固定したQuery/HeadでToken内容と位置を区別する確認、同一Query行のKey別相対変化を独立な開発サンプルで再現する確認、人手確認成否との対応。独立100件は最終検証のため温存する。',
        '', '## 再現と検証',
        'report.mdの分析コマンド、docs/gcg_vs_shuffled_phase.mdのテスト/比較/索引コマンドを参照。validation.jsonは計算時恒等式、90_validationのfinal_audit.jsonは最終ハッシュ/リンク/表監査。成果物はGitに含めない。']
    (p['integrated']/'interpretation.md').write_text('\n'.join(lines)+'\n')
    mtg=['# MTG用要約：GCG vs Shuffled','',
        '**最重要点：RMS増加は主に負Score側のエネルギー増加に対応する。一方、SuffixへAttentionが増えるのはQuery集合によって異なる。**','',
        '## これまでと今回',
        '既存結果：q4のSuffix割合+5.56ポイント、Original割合−1.80ポイント。UU/UO RMSは増加、Signed Meanは低下、OOは不変。今回は正負分解、同一Query Softmax、可視行中心化をCPUで追加した。主要18ペア、全20ペア補助、固定Head群を維持。','',
        '## 重要な結果',
        '1. Full UUのΔE_neg=+1.150183に対しΔE_pos=+0.000145。UOも+1.213336対+0.000317。平均RMS²増加はほぼ負側に対応。Signed Mean低下もmu_neg側が主。Causalでも方向は維持。（30_sign_decomposition/overview.csv、figures/sign_contributions.png）',
        '2. Suffix Query20位置平均のM_Uは16.88%対17.62%、差−0.7414ポイント、17/18件で低下。q4の+5.56ポイントとは逆方向。（40_query_softmax/overview.csv、figures/suffix_query_profiles.png）',
        '3. q4の中心化ScoreはSuffix +0.398210、Original −0.505694、相対差+0.903905。共通行シフトだけでは説明できない。一方、Suffix Query平均では中心化Suffix差−0.069292、相対差−0.064698。（60_integrated_analysis/overview.csv、50_row_centered/figures/row_centered_profiles.png）',
        '4. 固定Head群でも一様ではない。D増加群のΔM_Uはq4 +21.62ポイント、Suffix平均−2.03ポイント。F増加群はq4 +12.09ポイント、Suffix平均−5.75ポイント。（40_query_softmax/view_summary_18.csv、figures/fixed_group_query_profiles.png）',
        '5. 全20件でも上記の主な平均方向は維持。OO不変・エネルギー/平均の恒等式・Softmax/中心化不変性・q1-q4旧値一致を確認。共通化前後のCSV/NPZ最大誤差0、旧CLI/旧パス維持。','',
        '## 明らかになったこと・残ること',
        '「Suffix内部Score規模増加＝Suffix内部QueryからSuffixへのAttention増加」という単純な説明は支持されない。Regionと最終Queryを区別する必要がある。',
        '特定Token/Headが成功を引き起こすか、実際の有害応諾とどう対応するか、別サンプルでも再現するかは未確認。次は固定条件でQuery/Key別の相対変化を独立な開発データで確認する候補がある。',
        '既存43拒否表現による自動ラベルの探索結果で、人手確認済み成功一般・因果・有意性は主張しない。Headは独立標本ではなく同一18件で群を選択している。','',
        '## 閲覧順',
        '統一README → 60_integrated_analysis/'+run_id+'/figures/integrated_overview.png → 30_sign_decompositionのsign_contributions.png → 40_query_softmaxのsuffix_query_profiles.png → 50_row_centeredのrow_centered_profiles.png。',
        '詳細な全数表・相関・定義・再現手順は同runのinterpretation.mdとreport.md。']
    (p['integrated']/'MTG_summary.md').write_text('\n'.join(mtg)+'\n')
    created=['interpretation.md','MTG_summary.md','energy_head_correspondence.csv','suffix_position_effects.csv','centered_key_diagnostics.csv','figures/centered_key_diagnostics.png']
    write_json(p['integrated']/'supplemental_hashes.json',{name:sha256_file(p['integrated']/name) for name in created})
    print(p['integrated']/'interpretation.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'data/outputs/gcg_vs_shuffled/legacy_selection20');p.add_argument('--run-id',required=True)
    a=p.parse_args();build(a.root,a.run_id)
