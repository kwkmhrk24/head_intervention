"""Validated, data-driven Japanese reporting; no model/tensor dependencies."""
import numpy as np
import pandas as pd

TABLE_METRICS = tuple(f'{f}_{r}_{p}' for p in ('full','causal') for r in ('OO','UU','UO')
                      for f in ('RMS','mu','E_pos','E_neg','mu_pos','mu_neg')) + tuple(
    f'{q}_{f}' for q in ('suffix_mean20','q4') for f in
    ('M_U','M_O','D','conditional_M_U','row_mean','centered_O','centered_U','gap_U_minus_O'))
SUMMARY_COLUMNS = ('gcg_mean','shuffled_mean','delta_mean','delta_std','sample_count',
                   'gcg_larger_count','shuffled_larger_count','equal_count')


def require_columns(frame, columns, label):
    missing=set(columns)-set(frame)
    if missing:raise ValueError(f'{label}: missing columns {sorted(missing)}')


def validate_summary(frame, label):
    require_columns(frame, SUMMARY_COLUMNS, label)
    if not np.isfinite(frame[list(SUMMARY_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError(f'{label}: nonfinite required reporting values')
    for c in ('sample_count','gcg_larger_count','shuffled_larger_count','equal_count'):
        x=frame[c].to_numpy(float)
        if np.any(x<0) or np.any(x!=np.floor(x)):raise ValueError(f'{label}: invalid count {c}')
    if not (frame.gcg_larger_count+frame.shuffled_larger_count+frame.equal_count==frame.sample_count).all():
        raise ValueError(f'{label}: direction counts do not sum to sample count')


class ReportData:
    """Manifest owns cohort/group definitions; CSV owns every reported result."""
    def __init__(self, manifest, overview, group_frames):
        self.manifest=manifest;self.overview=overview.copy();self.groups={}
        if manifest.get('condition_order')!=['gcg','shuffled']:raise ValueError('condition order must be GCG-Shuffled')
        self.primary=len(manifest['primary_indices']);self.all=len(manifest['all_indices'])
        if self.primary!=18 or self.all!=20 or len(set(manifest['primary_indices']))!=self.primary or len(set(manifest['all_indices']))!=self.all or not set(manifest['primary_indices'])<=set(manifest['all_indices']):
            raise ValueError('primary18/selection20 cohort mismatch')
        require_columns(overview, ('subset','metric','view_id'), 'overview')
        if overview.duplicated(['subset','metric']).any() or set(overview.subset)!={self.primary,self.all}:
            raise ValueError('missing/duplicate overview subset keys')
        if not overview.view_id.eq('all_heads').all():raise ValueError('overview is not all_heads')
        validate_summary(overview,'overview')
        if not overview.sample_count.eq(overview.subset).all():raise ValueError('overview subset/count mismatch')
        for n in (self.primary,self.all):
            if not set(TABLE_METRICS)<=set(overview[overview.subset==n].metric):raise ValueError('missing required overview metric')
        views=manifest['views'];self.view_map={v['view_id']:v for v in views}
        if len(self.view_map)!=len(views):raise ValueError('duplicate manifest view')
        self.fixed=[v['view_id'] for v in views if v['kind']=='selected_group']
        for name in ('M_U_increase','D_increase','F_increase'):
            if name not in self.fixed:raise ValueError('required fixed group missing: '+name)
        for name in self.fixed:
            members=self.view_map[name]['members']
            if len(members)!=5 or len(set(members))!=5 or any(not isinstance(h,int) or not 0<=h<1024 for h in members):
                raise ValueError('invalid fixed group members: '+name)
        for n in (self.primary,self.all):
            f=group_frames[n].copy();require_columns(f,('view_id','view_kind','head_count','query','metric'),'group summary')
            if f.duplicated(['view_id','query','metric']).any():raise ValueError('duplicate group/query/metric')
            for name in self.fixed:
                rows=f[f.view_id==name]
                if rows.empty or not rows.view_kind.eq('selected_group').all() or not rows.head_count.eq(len(self.view_map[name]['members'])).all():
                    raise ValueError('group CSV/manifest mismatch: '+name)
                required=rows[rows['query'].isin(['suffix_mean20','q4']) & rows.metric.isin(['M_U','M_O','D','centered_O','centered_U'])]
                if len(required)!=10:raise ValueError('missing required fixed group query metrics: '+name)
                validate_summary(required,'group '+name)
                if not required.sample_count.eq(n).all():raise ValueError('group cohort size mismatch')
            self.groups[n]=f

    def row(self, metric, subset=None):
        n=self.primary if subset is None else subset
        rows=self.overview[(self.overview.subset==n)&(self.overview.metric==metric)]
        if len(rows)!=1:raise ValueError(f'expected one {n}/{metric} row')
        return rows.iloc[0]

    def delta(self,metric,subset=None):return float(self.row(metric,subset).delta_mean)

    def group(self,name,query,metric,subset=None):
        n=self.primary if subset is None else subset
        if name not in self.fixed:raise ValueError('unknown fixed group '+name)
        f=self.groups[n];rows=f[(f.view_id==name)&(f['query']==query)&(f.metric==metric)]
        if len(rows)!=1:raise ValueError(f'expected one {n}/{name}/{query}/{metric}')
        return rows.iloc[0]

    def counts(self,metric,subset=None):
        r=self.row(metric,subset)
        return f'{int(r.gcg_larger_count)}増/{int(r.shuffled_larger_count)}減/{int(r.equal_count)}同等（{int(r.sample_count)}件）'


def render_reports(data,run_id,correlations):
    d=data.delta;n=data.primary;nall=data.all
    require_columns(correlations,('subset','unit','x','y','pearson','spearman','same_direction_fraction'),'correlations')
    if correlations.duplicated(['subset','unit','x','y']).any() or set(correlations.subset)!={n,nall}:
        raise ValueError('missing/duplicate correlation cohorts')
    if not np.isfinite(correlations[['pearson','spearman','same_direction_fraction']].to_numpy(float)).all():raise ValueError('nonfinite correlations')
    keys=lambda n:set(map(tuple,correlations[correlations.subset==n][['unit','x','y']].to_numpy()))
    if keys(n)!=keys(nall):raise ValueError('correlation cohort correspondence mismatch')
    negative_dominant=all(d(f'E_neg_{r}_full')>0 and abs(d(f'E_neg_{r}_full'))>abs(d(f'E_pos_{r}_full')) for r in ('UU','UO'))
    energy_conclusion=('全Head/サンプル平均の二乗エネルギー増加は、主に負Score側の増加に対応する。'
                       if negative_dominant else '正負寄与は下表のとおりで、UU/UO両方を負側の増加が主と一括して要約できない。')
    opposite=d('suffix_mean20_M_U')*d('q4_M_U')<0
    query_conclusion=('Suffix内Query平均と最終QueryのM_U変化は逆方向で、「全QueryでSuffixへAttentionが増える」という一般化は支持されない。'
                      if opposite else 'Suffix内Query平均と最終QueryのM_U変化は下表の方向。平均の一致だけから全Queryで同じ変化とは断定できない。')
    relative=any(abs(d(m))>1e-12 for m in ('q4_centered_O','q4_centered_U','q4_gap_U_minus_O'))
    centered_conclusion=('q4では中心化後にも相対変化が残り、行全体への共通定数シフトだけでは説明できない。'
                         if relative else 'q4の中心化領域平均・相対平均差の変化は数値的に0である。Key別変化の有無は平均だけでは判断できない。')
    main=['RMS_UU_full','mu_UU_full','E_pos_UU_full','E_neg_UU_full','RMS_UO_full','mu_UO_full','E_pos_UO_full','E_neg_UO_full',
          'suffix_mean20_M_U','suffix_mean20_M_O','suffix_mean20_D','suffix_mean20_gap_U_minus_O','q4_M_U','q4_D','q4_gap_U_minus_O']
    flips=[m for m in main if np.sign(d(m))!=np.sign(d(m,nall))]
    sensitivity=('主要な方向は全20件でも維持される。' if not flips else '主要18件と全20件で方向が異なる指標：'+', '.join(flips)+'。')
    def energy_text(region,policy='full'):
        return f"{policy} {region}：ΔE_pos={d(f'E_pos_{region}_{policy}'):+.6f}、ΔE_neg={d(f'E_neg_{region}_{policy}'):+.6f}（単位Score²）。"
    def signed_text(region):
        return f"Full {region}：ΔRMS={d('RMS_'+region+'_full'):+.6f}、ΔSigned Mean={d('mu_'+region+'_full'):+.6f}、mu_pos寄与{d('mu_pos_'+region+'_full'):+.6f}、mu_neg寄与{d('mu_neg_'+region+'_full'):+.6f}（単位Score）。"
    def query_text(label):
        r=data.row(label+'_M_U')
        return (f"{label}：M_UはGCG {100*r.gcg_mean:.4f}%、Shuffled {100*r.shuffled_mean:.4f}%、差{100*d(label+'_M_U'):+.4f}パーセントポイント。"
                f"M_O差{100*d(label+'_M_O'):+.4f}パーセントポイント、D差{d(label+'_D'):+.6f}（自然対数比）、"
                f"条件付きM_U差{100*d(label+'_conditional_M_U'):+.4f}パーセントポイント。M_U方向は{data.counts(label+'_M_U')}、D方向は{data.counts(label+'_D')}。")
    def center_text(label):
        return (f"{label}：row_mean差{d(label+'_row_mean'):+.6f}、中心化Original平均差{d(label+'_centered_O'):+.6f}、"
                f"中心化Suffix平均差{d(label+'_centered_U'):+.6f}、Suffix−Original相対平均差{d(label+'_gap_U_minus_O'):+.6f}（単位Score）。")
    group_lines=[]
    for name in data.fixed:
        for cohort in (n,nall):
            g=data.group(name,'q4','M_U',cohort);s=data.group(name,'suffix_mean20','M_U',cohort)
            group_lines.append(f"| {cohort} | {name} | {100*g.delta_mean:+.4f} | {100*s.delta_mean:+.4f} | {int(s.gcg_larger_count)}/{int(s.shuffled_larger_count)}/{int(s.equal_count)} |")
    limits='自動43拒否表現ルールの探索的比較で、人手確認済み成功・因果・有意性・一般化を主張しない。Headは独立標本でなく、固定群の選択と確認には同じ主要サンプルを使用している。'
    lines=['# GCG vs Shuffled：追加分析・統合考察','', '## 条件と計算',
        f'主分析{n}ペア、補助{nall}ペア。保存済みFP32 Score（RoPE後・Mask/Softmax前）をCPU/FP64で計算した既存CSVを参照する。差はGCG−Shuffled。',
        'サンプル/Head/Queryごとの特徴を計算後、等重みHead平均、サンプル平均。SDはペア差のddof=1。Fullは未来要素を含む。CausalはKey<=Queryで対角を含む。',
        'Causal UUの210要素平均と、各Queryの特徴を求めた後のsuffix_mean20平均は、重みが異なるため同じ平均ではない。',limits,'',
        '## ② 正負別エネルギー：観測事実',energy_text('UU'),energy_text('UO'),energy_text('UU','causal'),energy_conclusion,
        'ΔE_pos+ΔE_negは平均RMS²の差への加法分解であり、平均RMSの差や平均RMSの二乗差への分解ではない。',signed_text('UU'),signed_text('UO'),
        'Scoreの正負はAttentionの強弱ではない。正負割合・中央値・分位点は30_sign_decompositionのmetrics/head_summary参照。平均分位点は混合分布分位点ではない。','',
        '## ③ 同一QueryのSoftmax',query_text('suffix_mean20'),query_text('q4'),query_conclusion,
        '全可視KeyをSoftmaxの分母に含める。conditional_M_UはOriginal/Suffix間だけの配分。Dは自然対数logsumexp差。位置対応はToken内容対応を意味しない。','',
        '## ④ 行中心化',center_text('suffix_mean20'),center_text('q4'),centered_conclusion,
        '中心化は可視Keyの平均を引く操作であり、SoftmaxとDは不変。特定Keyが機構の原因かは未確認。例示のcentered_key_diagnosticsと元row_profilesは事前の固定Head・Indexから取得する。','',
        '## 固定Head群',f'既存の{len(data.fixed)}群をmanifestで照合し、再選択していない。単位はパーセントポイント。','',
        '| N | 固定群 | q4 ΔM_U | Suffix平均 ΔM_U | Suffix平均 増/減/同等 |','|---|---|---:|---:|---|',*group_lines,'',
        '## 主要18と全20・統合表','', '| N | 指標 | GCG | Shuffled | GCG−Shuffled | 差SD | 増/減/同等 |','|---|---|---:|---:|---:|---:|---|']
    for cohort in (n,nall):
        for metric in TABLE_METRICS:
            r=data.row(metric,cohort)
            lines.append(f'| {cohort} | {metric} | {r.gcg_mean:.7f} | {r.shuffled_mean:.7f} | {r.delta_mean:+.7f} | {r.delta_std:.7f} | {int(r.gcg_larger_count)}/{int(r.shuffled_larger_count)}/{int(r.equal_count)} |')
    lines += ['',sensitivity+'同じ選択済み20件の補助分析で、独立検証ではない。','', '## 相関と方向一致（記述のみ）','',
              '| N | 単位 | X | Y | Pearson | Spearman | 同方向率 |','|---|---|---|---|---:|---:|---:|']
    for row in correlations.itertuples():lines.append(f'| {row.subset} | {row.unit} | {row.x} | {row.y} | {row.pearson:.4f} | {row.spearman:.4f} | {row.same_direction_fraction:.4f} |')
    lines+=['','## 出典・再現性',
        '数値出典：60_integrated_analysis/<run>/overview.csv、head_summary_18.csv、correlations.csv、40_query_softmax/<run>/overview.csvとview_summary_18/20.csv。レポート生成では分析値を再計算していない。',
        '図：30_sign_decompositionのsign_contributions、40_query_softmaxのsuffix_query_profiles/fixed_group_query_profiles、50_row_centeredのrow_centered_profiles、60_integrated_analysisのintegrated_relationships/integrated_overview。',
        '入力CSV/manifest/profileと実行ソースのSHA256、生成ファイルの対応はreport_manifest.jsonとsupplemental_hashes.json。計算時validationと互換性監査は元runの90_validationを参照する。',
        '再現：python -B scripts/report_gcg_shuffled_phase.py --root INPUT_ROOT --run-id '+run_id+' --output-dir NEW_REPORT_DIR。既存成果物は上書きしない。','',
        '## 限界と次の検証候補',limits,
        'Region集計と最終QueryはQuery集合が異なる。平均・ノルム・エネルギーだけでは分布全体、Softmax変化の理由、特定Keyの因果を確定できない。',
        '次の候補は固定条件でのQuery/Key相対変化の独立な開発データ確認と人手成否との対応。追加実験・モデル推論・GPU・新規Suffix・独立100件は今回実施しない。']
    highlights=[energy_text('UU')+energy_text('UO')+energy_conclusion,
                query_text('suffix_mean20')+query_text('q4')+query_conclusion,
                center_text('q4')+center_text('suffix_mean20')+centered_conclusion,
                '固定群：'+' '.join(f"{name}のΔM_Uはq4 {100*data.group(name,'q4','M_U').delta_mean:+.2f}、Suffix平均{100*data.group(name,'suffix_mean20','M_U').delta_mean:+.2f}パーセントポイント。" for name in ('M_U_increase','D_increase','F_increase')),
                sensitivity+'独立検証ではない。']
    mtg=['# MTG用要約：GCG vs Shuffled','',energy_conclusion+query_conclusion,'', '## これまでと今回',
         f'主要{n}ペア、補助{nall}ペア。同じ固定Head群・入力・Mask・集計条件。既存結果：'+signed_text('UU')+signed_text('UO')+query_text('q4'),
         '', '## 最も重要な結果',*[f'{j+1}. {text}' for j,text in enumerate(highlights)],'',
         '出典：60_integrated_analysis/<run>/overview.csv、40_query_softmax/<run>/view_summary_18/20.csv。図は30のsign_contributions、40のsuffix_query_profiles/fixed_group_query_profiles、50のrow_centered_profiles、60のintegrated_overview。','',
         '## まだ説明できないこと・次の候補',limits,
         '特定Token/Headの因果、実際の有害応諾、別サンプルへの再現は未確認。固定したQuery/Key相対変化を独立な開発データと人手成否で確認する候補がある。詳細・単位・再現手順はinterpretation.mdとreport_manifest.json。']
    return '\n'.join(lines)+'\n','\n'.join(mtg)+'\n'
