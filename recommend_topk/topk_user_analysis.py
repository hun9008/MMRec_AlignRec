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
    top = df.sort_values(metric, ascending=False).head(k)[[id_col, metric]].copy()
    top['pos_degree'] = top[id_col].apply(degree_of).astype('Int64')
    # (선택) total 상호작용도 참고하고 싶으면 아래 주석 해제
    # top['total_interactions'] = top[id_col].map(tot_counts).astype('Int64')
    return top[[id_col, 'pos_degree']]

# 모델별 Top-K 로드
df_alignrec = pd.read_csv(ALIGNREC_FILE)
df_anchor   = pd.read_csv(ANCHOR_FILE)
df_aa       = pd.read_csv(AA_FILE)

tbl_align = topk_degree_table(df_alignrec, k=TOPK)
tbl_anchor = topk_degree_table(df_anchor, k=TOPK)
tbl_aa = topk_degree_table(df_aa, k=TOPK)

# 보기 좋게 합치기(행 기준으로 정렬을 맞춰 가로 결합)
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
        'hub%':    (series == 'hub').sum()    / n * 100,
        'cold%':   (series == 'cold').sum()   / n * 100,
        'normal%': (series == 'normal').sum() / n * 100,
        'unknown%':(series == 'unknown').sum() / n * 100,
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

# =========================
# 저장
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