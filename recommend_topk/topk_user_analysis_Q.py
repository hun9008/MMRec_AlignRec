# topk_user_overview_interaction.py
import os
import numpy as np
import pandas as pd

# =========================
# 설정
# =========================
DATA_DIR = '../data/baby'
INTER_FILE = f'{DATA_DIR}/baby.inter'
U_MAP_FILE = f'{DATA_DIR}/u_id_mapping.csv'     # 전체 유저 포함(상호작용 0명도 포함하려면 사용)
POS_THRESH = 1                                   # rating >= POS_THRESH 를 positive로 간주

TOPK = 20
ALIGNREC_FILE = 'user_metrics_k20_alignrec.csv'               # columns: user_id, recall@20, ndcg@20
ANCHOR_FILE   = 'user_metrics_k20_anchor.csv'
AA_FILE       = 'user_metrics_k20_alignrec_anchor.csv'

OUT_DIR = './analysis_out'
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 데이터 로드
# =========================
df_inter = pd.read_csv(INTER_FILE, sep='\t')
# 컬럼명 통일
col_map = {}
if 'userID' in df_inter.columns: col_map['userID'] = 'user_id'
if 'itemID' in df_inter.columns: col_map['itemID'] = 'item_id'
df_inter = df_inter.rename(columns=col_map)

if not {'user_id','item_id','rating'}.issubset(df_inter.columns):
    raise ValueError(f"baby.inter에 필요한 컬럼(user_id,item_id,rating)이 없습니다. 현재 컬럼: {df_inter.columns.tolist()}")

# 전체 유저 목록 (상호작용 0명도 포함하려면 u_id_mapping 사용)
try:
    df_user = pd.read_csv(U_MAP_FILE)
    user_id_col = 'user_id' if 'user_id' in df_user.columns else df_user.columns[0]
    all_users = df_user[user_id_col].astype(int).values
except Exception:
    # 매핑이 없으면 inter에 등장한 유저만 대상으로 함
    all_users = np.sort(df_inter['user_id'].unique())

# =========================
# positive / total 상호작용 수 집계
# =========================
df_pos = df_inter[df_inter['rating'] >= POS_THRESH]

# 유저별 카운트 (Series, index = user_id)
pos_counts = df_pos.groupby('user_id').size()
tot_counts = df_inter.groupby('user_id').size()

# 전체 유저로 reindex (없으면 0)
pos_counts = pos_counts.reindex(all_users, fill_value=0).astype(int)
tot_counts = tot_counts.reindex(all_users, fill_value=0).astype(int)

degrees = pos_counts.values  # 우리 기준 degree = positive interaction 수

# =========================
# 전체 통계
# =========================
total_users = len(all_users)
deg_max = int(degrees.max()) if total_users > 0 else 0
deg_min = int(degrees.min()) if total_users > 0 else 0
n_deg_max = int((degrees == deg_max).sum())
n_deg_min = int((degrees == deg_min).sum())

mean_deg = degrees.mean() if total_users > 0 else 0
median_deg = np.median(degrees) if total_users > 0 else 0
p90 = np.percentile(degrees, 90) if total_users > 0 else 0
p95 = np.percentile(degrees, 95) if total_users > 0 else 0
p99 = np.percentile(degrees, 99) if total_users > 0 else 0
p5  = np.percentile(degrees,  5) if total_users > 0 else 0
zero_ratio = (degrees == 0).mean() if total_users > 0 else 0

print("===== 기본 정보 (user-item positive >= {} 기준) =====".format(POS_THRESH))
print(f"전체 유저수            : {total_users}")
print(f"degree max            : {deg_max} (user 수: {n_deg_max})")
print(f"degree min            : {deg_min} (user 수: {n_deg_min})")

print("\n===== 기타 유저 통계 =====")
print(f"평균 degree           : {mean_deg:.2f}")
print(f"중앙값 degree         : {median_deg:.2f}")
print(f"90퍼센타일            : {p90:.2f}")
print(f"95퍼센타일            : {p95:.2f}")
print(f"99퍼센타일            : {p99:.2f}")
print(f"5퍼센타일             : {p5:.2f}")
print(f"positive degree==0 비율: {zero_ratio*100:.2f}%")

# =========================
# Top-K 테이블에 positive degree 붙이기
# =========================
# lookup Series (index=user_id, value=degree)
degree_lookup = pd.Series(pos_counts.values, index=pos_counts.index)

def degree_of(uid):
    try:
        return int(degree_lookup.get(int(uid), np.nan))
    except Exception:
        return np.nan

def topk_degree_table(df, k=TOPK, metric='ndcg@20', id_col='user_id'):
    if metric not in df.columns:
        raise ValueError(f"'{metric}' 컬럼이 없습니다. 사용 가능 컬럼: {df.columns.tolist()}")
    top = df.sort_values(metric, ascending=False).head(k)[[id_col, metric]].copy()
    top['pos_degree'] = top[id_col].apply(degree_of).astype('Int64')
    return top[[id_col, 'pos_degree']]

# 모델별 Top-K 로드
df_alignrec = pd.read_csv(ALIGNREC_FILE)
df_anchor   = pd.read_csv(ANCHOR_FILE)
df_aa       = pd.read_csv(AA_FILE)

# 안전 체크: 각 파일에 recall@20, ndcg@20 있어야 mean_score 계산 가능
RECALL_COL = 'recall@20'
NDCG_COL   = 'ndcg@20'
for name, dfm in [('AlignRec', df_alignrec), ('Anchor', df_anchor), ('AlignRec+Anchor', df_aa)]:
    missing = [c for c in [RECALL_COL, NDCG_COL, 'user_id'] if c not in dfm.columns]
    if missing:
        raise ValueError(f"{name} 파일에 필요한 컬럼 {missing} 가 없습니다. 현재 컬럼: {dfm.columns.tolist()}")

tbl_align = topk_degree_table(df_alignrec, k=TOPK, metric=NDCG_COL)
tbl_anchor = topk_degree_table(df_anchor, k=TOPK, metric=NDCG_COL)
tbl_aa = topk_degree_table(df_aa, k=TOPK, metric=NDCG_COL)

# 보기 좋게 합치기(행 기준 정렬 맞춰 가로 결합)
tbl_align = tbl_align.reset_index(drop=True).rename(columns={'user_id':'id_alignrec','pos_degree':'posdeg_alignrec'})
tbl_anchor = tbl_anchor.reset_index(drop=True).rename(columns={'user_id':'id_anchor','pos_degree':'posdeg_anchor'})
tbl_aa = tbl_aa.reset_index(drop=True).rename(columns={'user_id':'id_alignrec_anchor','pos_degree':'posdeg_alignrec_anchor'})

combo = pd.concat([tbl_align, tbl_anchor, tbl_aa], axis=1)

print("\n===== Top-{} 유저 positive degree (AlignRec | Anchor | AlignRec+Anchor) =====".format(TOPK))
print(combo.to_string(index=False))

# =========================
# p5(Cold), p95(Hub) 라벨링 (positive degree 분포 기준)
# =========================
cold_cut = float(p5)
hub_cut  = float(p95)

def label_by_percentile(deg):
    if pd.isna(deg):
        return 'unknown'
    if deg <= cold_cut:
        return 'cold'
    if deg >= hub_cut:
        return 'hub'
    return 'normal'

def ratio_by_label(series):
    n = len(series)
    return {
        'hub%':    (series == 'hub').sum()    / n * 100 if n>0 else 0.0,
        'cold%':   (series == 'cold').sum()   / n * 100 if n>0 else 0.0,
        'normal%': (series == 'normal').sum() / n * 100 if n>0 else 0.0,
        'unknown%':(series == 'unknown').sum()/ n * 100 if n>0 else 0.0,
    }

combo_labeled = combo.copy()
combo_labeled['alignrec_label']        = combo_labeled['posdeg_alignrec']       .apply(label_by_percentile)
combo_labeled['anchor_label']          = combo_labeled['posdeg_anchor']         .apply(label_by_percentile)
combo_labeled['alignrec_anchor_label'] = combo_labeled['posdeg_alignrec_anchor'].apply(label_by_percentile)

align_ratio = ratio_by_label(combo_labeled['alignrec_label'])
anchor_ratio = ratio_by_label(combo_labeled['anchor_label'])
aa_ratio     = ratio_by_label(combo_labeled['alignrec_anchor_label'])

print("\n===== Top-{} 라벨 비율 (p5={:.2f}, p95={:.2f}) [positive degree 기준] =====".format(TOPK, cold_cut, hub_cut))
print("AlignRec        :", f"hub {align_ratio['hub%']:.1f}%, cold {align_ratio['cold%']:.1f}%, degree avg {combo_labeled['posdeg_alignrec'].mean():.2f}")
print("Anchor          :", f"hub {anchor_ratio['hub%']:.1f}%, cold {anchor_ratio['cold%']:.1f}%, degree avg {combo_labeled['posdeg_anchor'].mean():.2f}")
print("AlignRec+Anchor :", f"hub {aa_ratio['hub%']:.1f}%, cold {aa_ratio['cold%']:.1f}%, degree avg {combo_labeled['posdeg_alignrec_anchor'].mean():.2f}")

# ============================================================
# 'zero' / (0,1) 4분위(Q1..Q4) / 'one' 방식의 점수 bin 생성
# mean_score = (recall@20 + ndcg@20)/2
# ============================================================
SCORE_NAME = 'mean_score'
CATS = ['zero', 'Q1', 'Q2', 'Q3', 'Q4', 'one']

def attach_degree_and_score(df_metrics):
    """지표 DF에 pos_degree/total_interactions/mean_score 붙이기"""
    dfm = df_metrics.copy()
    dfm['user_id'] = dfm['user_id'].astype(int)
    dfm[SCORE_NAME] = (dfm[RECALL_COL].astype(float) + dfm[NDCG_COL].astype(float)) / 2.0
    dfm['pos_degree'] = dfm['user_id'].map(degree_lookup).astype('Int64')
    dfm['total_interactions'] = dfm['user_id'].map(tot_counts).astype('Int64')
    return dfm

def score_bins_zero_mid_quartile_one(dfm, score_col=SCORE_NAME):
    """
    mean_score == 0.0 -> 'zero'
    mean_score == 1.0 -> 'one'
    0.0 < score < 1.0 -> qcut으로 4분위(Q1..Q4). 동점으로 bin 수가 줄면 Q1..Qm.
    """
    score = dfm[score_col]
    labels = pd.Series(index=dfm.index, dtype='object')

    # zero / one
    labels.loc[score == 0.0] = 'zero'
    labels.loc[score == 1.0] = 'one'

    # 중간 구간
    mask_between = (score > 0.0) & (score < 1.0)
    mid_scores = score[mask_between]

    if len(mid_scores) > 0:
        # 동점이 많을 수 있어 duplicates='drop'
        codes, bins = pd.qcut(mid_scores, q=[0, .25, .5, .75, 1.0],
                              labels=False, retbins=True, duplicates='drop')
        m = len(bins) - 1  # 실제 bin 수(<=4)
        qlabels = [f"Q{i+1}" for i in range(m)] if m > 0 else []
        if m > 0:
            labels.loc[mask_between] = pd.Series(codes, index=mid_scores.index).map(
                lambda x: qlabels[int(x)]
            )
        else:
            # 안전장치: 중간 값이 한 점수로만 구성된 극단 케이스
            labels.loc[mask_between] = 'Q1'

    # Categorical로 정렬 순서 유지
    return pd.Categorical(labels, categories=CATS, ordered=True)

def summarize_by_bin(dfm, bin_col='score_bin', score_col=SCORE_NAME):
    g = dfm.groupby(bin_col, observed=True)

    def _summ(grp):
        # pos_degree를 float로 안전 변환 (Int64 → float)
        s = pd.to_numeric(grp['pos_degree'], errors='coerce').astype(float)
        has_vals = s.notna().any()

        return pd.Series({
            'n_users': len(grp),
            'score_min': grp[score_col].min(),
            'score_max': grp[score_col].max(),
            'score_mean': grp[score_col].mean(),
            'posdeg_min': s.min() if has_vals else np.nan,
            'posdeg_max': s.max() if has_vals else np.nan,
            'posdeg_mean': s.mean() if has_vals else np.nan
        })

    out = g.apply(_summ).reset_index()
    return out

def analyze_by_zero_mid_quartile_one(df_metrics, model_name):
    dfm = attach_degree_and_score(df_metrics)
    dfm = dfm[dfm[SCORE_NAME].notna()].copy()
    if dfm.empty:
        print(f"[WARN] {model_name}: {SCORE_NAME} 데이터가 비었습니다.")
        return None, None
    dfm['score_bin'] = score_bins_zero_mid_quartile_one(dfm, score_col=SCORE_NAME)
    # 분포 출력
    vc = dfm['score_bin'].value_counts(dropna=False)
    print(f"\n[{model_name}] score_bin 분포:")
    print(vc.to_string())
    # 요약
    summary = summarize_by_bin(dfm, bin_col='score_bin', score_col=SCORE_NAME)
    summary.insert(0, 'model', model_name)
    return dfm, summary

# 세 모델 실행
df_align_scored, sum_align = analyze_by_zero_mid_quartile_one(df_alignrec, 'AlignRec')
df_anchor_scored, sum_anchor = analyze_by_zero_mid_quartile_one(df_anchor, 'Anchor')
df_aa_scored,     sum_aa     = analyze_by_zero_mid_quartile_one(df_aa,     'AlignRec+Anchor')

# -------------------------
# 분위별 교집합 비율 계산
#   overlap_same_bin_% = |A∩B∩C| / |A∪B∪C| * 100
# -------------------------
def compute_overlap_percent_by_bin(df_a, df_b, df_c, bin_col='score_bin'):
    overlaps = {}
    for b in CATS:
        a_set = set(df_a.loc[df_a[bin_col] == b, 'user_id']) if df_a is not None else set()
        b_set = set(df_b.loc[df_b[bin_col] == b, 'user_id']) if df_b is not None else set()
        c_set = set(df_c.loc[df_c[bin_col] == b, 'user_id']) if df_c is not None else set()
        union_size = len(a_set | b_set | c_set)
        inter_size = len(a_set & b_set & c_set)
        overlaps[b] = (inter_size / union_size * 100.0) if union_size > 0 else np.nan
    return overlaps

overlap_map = compute_overlap_percent_by_bin(df_align_scored, df_anchor_scored, df_aa_scored)

# 병합 및 저장
summ_all = pd.concat([x for x in [sum_align, sum_anchor, sum_aa] if x is not None], ignore_index=True)
# score_bin 기준으로 overlap_same_bin_% 붙이기
summ_all['overlap_same_bin_%'] = summ_all['score_bin'].map(overlap_map)

summ_path = os.path.join(OUT_DIR, "metrics_bins_zero_midQ_one_summary.csv")
summ_all.to_csv(summ_path, index=False)

# 모델별 출력 (구분선)
print("\n===== zero | Q1..Q4 | one 별 pos_degree 통계 요약 =====")
for model_name in ['AlignRec', 'Anchor', 'AlignRec+Anchor']:
    print("\n" + "-"*72)
    print(f"[{model_name}]")
    print("-"*72)
    sub = summ_all[summ_all['model'] == model_name]
    if sub.empty:
        print("(no data)")
    else:
        # 보기 좋게 컬럼 순서 정리
        cols = ['score_bin','n_users','score_min','score_max','score_mean',
                'posdeg_min','posdeg_max','posdeg_mean','overlap_same_bin_%']
        print(sub[cols].to_string(index=False))

# 라벨링된 테이블도 저장(재현/검증용)
if df_align_scored is not None:
    df_align_scored.to_csv(os.path.join(OUT_DIR, "alignrec_with_score_bins.csv"), index=False)
if df_anchor_scored is not None:
    df_anchor_scored.to_csv(os.path.join(OUT_DIR, "anchor_with_score_bins.csv"), index=False)
if df_aa_scored is not None:
    df_aa_scored.to_csv(os.path.join(OUT_DIR, "alignrec_anchor_with_score_bins.csv"), index=False)

# =========================
# 저장 (기존 산출물)
# =========================
combo.to_csv(os.path.join(OUT_DIR, f"top{TOPK}_users_posdegree_combo.csv"), index=False)
combo_labeled.to_csv(os.path.join(OUT_DIR, f"top{TOPK}_users_posdegree_combo_labeled.csv"), index=False)

# 전체 유저의 degree(positive/total) 분포도 함께 저장(분석 재현용)
pd.DataFrame({
    'user_id': all_users,
    'pos_degree': pos_counts.values,
    'total_interactions': tot_counts.values
}).to_csv(os.path.join(OUT_DIR, "user_interaction_counts.csv"), index=False)

print(f"\n[INFO] Saved → {OUT_DIR}")