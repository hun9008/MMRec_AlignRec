# common_items_by_top_users.py
import os
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
from matplotlib import pyplot as plt

# --------------------
# 설정
# --------------------
DATA_DIR = '../data/baby'
INTER_FILE = f'{DATA_DIR}/baby.inter'
POS_THRESH = 1          # rating >= POS_THRESH 를 positive로 간주 (AlignRec 논문 재현이면 1 권장)
TOPK = 20
ALIGNREC_FILE = 'user_metrics_k20_alignrec.csv'               # columns: user_id, recall@20, ndcg@20
ANCHOR_FILE   = 'user_metrics_k20_anchor.csv'
AA_FILE       = 'user_metrics_k20_alignrec_anchor.csv'
OUT_DIR = './analysis_out_common'
os.makedirs(OUT_DIR, exist_ok=True)

# “공통”의 느슨한 기준(빈도 기반): TopK 유저 중 최소 몇 %가 해당 아이템과 상호작용했는지
MIN_USERS_FRAC = 0.5         # 예: 0.5 → 절반 이상 유저가 공통으로 본 아이템들
# 또는 정수 기준을 쓰고 싶다면 아래로 (둘 중 하나만 사용):
MIN_USERS_ABS = None         # 예: 10  (None이면 FRAC 사용)

# --------------------
# 데이터 로드
# --------------------
df_inter = pd.read_csv(INTER_FILE, sep='\t')
col_map = {}
if 'userID' in df_inter.columns: col_map['userID'] = 'user_id'
if 'itemID' in df_inter.columns: col_map['itemID'] = 'item_id'
df_inter = df_inter.rename(columns=col_map)

need = {'user_id','item_id','rating'}
if not need.issubset(df_inter.columns):
    raise ValueError(f"baby.inter에 필요한 컬럼 {need} 가 없습니다. 현재: {df_inter.columns.tolist()}")

# positive interactions만 사용
df_pos = df_inter[df_inter['rating'] >= POS_THRESH][['user_id','item_id']].copy()

# 모델별 TopK 유저 로드
def load_top_users(path, topk=TOPK):
    df = pd.read_csv(path)
    # ndcg@20 기준 상위 K
    df_top = df.sort_values('ndcg@20', ascending=False).head(topk)
    return df_top['user_id'].astype(int).tolist()

top_align = load_top_users(ALIGNREC_FILE, TOPK)
top_anchor = load_top_users(ANCHOR_FILE, TOPK)
top_aa = load_top_users(AA_FILE, TOPK)

# --------------------
# 유저별 positive 아이템 set 만들기
# --------------------
# 빠른 lookup을 위해 dict: user_id -> set(item_id)
user2items = (
    df_pos.groupby('user_id')['item_id']
    .apply(lambda s: set(s.astype(int).tolist()))
    .to_dict()
)

def items_for_users(user_list):
    """해당 유저 리스트의 아이템 집합 리스트 반환 (유저가 없으면 빈 set)"""
    return [user2items.get(u, set()) for u in user_list]

def strict_intersection(item_sets):
    """모든 유저가 공통으로 가진 아이템(교집합)"""
    if not item_sets: return set()
    inter = set(item_sets[0]).copy()
    for s in item_sets[1:]:
        inter &= s
        if not inter:
            break
    return inter

def frequent_items(item_sets, min_users_frac=MIN_USERS_FRAC, min_users_abs=MIN_USERS_ABS):
    """아이템 빈도 기반 공통 집합: 최소 사용자 수 이상이 가진 아이템들"""
    n_users = len(item_sets)
    if n_users == 0:
        return set()
    thresh = min_users_abs if (min_users_abs is not None) else int(np.ceil(n_users * min_users_frac))
    cnt = Counter()
    for s in item_sets:
        cnt.update(s)   # set이라 사용자당 중복 없음
    return {it for it, c in cnt.items() if c >= thresh}, cnt, thresh

item_global_pop = df_pos.groupby('item_id').size()

def summarize(model_name, users):
    item_sets = items_for_users(users)

    # 각 유저의 아이템 개수(=degree) 평균·합
    degrees = [len(s) for s in item_sets]
    avg_deg = np.mean(degrees) if degrees else 0
    total_deg = np.sum(degrees) if degrees else 0

    # 유저 전체가 가진 아이템 합집합(고유 아이템)
    union_items = set().union(*item_sets)

    # 교집합(엄격)
    inter = strict_intersection(item_sets)

    # 빈도 기반(느슨)
    common_set, freq_counter, thresh = frequent_items(item_sets)

    # ✅ union 아이템들의 "그룹 내" 빈도 합 = total_deg (검증용)
    group_union_freq_sum = sum(freq_counter[i] for i in union_items)  # == total_deg

    # ✅ union 아이템들의 "글로벌" 인기 합/평균
    union_items_list = list(union_items)
    global_pop_vals = item_global_pop.reindex(union_items_list).fillna(0).astype(int)
    global_pop_sum = int(global_pop_vals.sum())
    global_pop_avg = float(global_pop_vals.mean()) if len(global_pop_vals) else 0.0

    print(f"\n===== {model_name} (Top-{len(users)} users) =====")
    print(f"- Users avg degree             : {avg_deg:.1f}")
    print(f"- Users total degree(합)       : {total_deg}")
    print(f"- Union items size             : {len(union_items)}")
    print(f"- Strict intersection size     : {len(inter)}")
    print(f"- Frequent items size (≥{thresh} users): {len(common_set)}")

    # ▶ 추가 출력: union 아이템 degree 합(내부/글로벌)
    print(f"- [Union] 그룹내 빈도 합       : {group_union_freq_sum}  (== total_deg)")
    print(f"- [Union] 글로벌 인기 합       : {global_pop_sum}")
    print(f"- [Union] 글로벌 인기 평균     : {global_pop_avg:.2f}")

    # 상위 빈도 아이템 20개 미리보기(그룹 내)
    top_freq = pd.DataFrame(freq_counter.most_common(20), columns=['item_id','user_count'])
    print(top_freq.to_string(index=False))

    # 저장
    pd.Series(sorted(inter)).to_csv(os.path.join(OUT_DIR, f'{model_name}_strict_intersection_items.csv'),
                                    index=False, header=['item_id'])
    pd.DataFrame({'item_id': list(common_set),
                  'user_count': [freq_counter[i] for i in common_set]}) \
      .sort_values('user_count', ascending=False) \
      .to_csv(os.path.join(OUT_DIR, f'{model_name}_frequent_items.csv'), index=False)

    # (옵션) union 아이템의 글로벌 인기도 테이블 저장
    pd.DataFrame({'item_id': union_items_list,
                  'global_pop': global_pop_vals.values}) \
      .sort_values('global_pop', ascending=False) \
      .to_csv(os.path.join(OUT_DIR, f'{model_name}_union_items_global_pop.csv'), index=False)# --------------------
# 실행
# --------------------
summarize('AlignRec', top_align)
summarize('Anchor', top_anchor)
summarize('AlignRec_Anchor', top_aa)

print(f"\n[INFO] Saved CSVs to {OUT_DIR}")

def subgraph_edges_for_users(user_ids, df_pos, max_edges_per_user=None):
    sub = df_pos[df_pos['user_id'].isin(user_ids)].copy()
    if max_edges_per_user is not None:
        sub = (sub.groupby('user_id', group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), max_edges_per_user), random_state=42)))
    users = sub['user_id'].unique().tolist()
    items = sub['item_id'].unique().tolist()
    edges = list(sub[['user_id','item_id']].itertuples(index=False, name=None))  # (u,i) 튜플
    return users, items, edges

def make_bipartite_layout(users, items):
    pos = {}
    uy = np.linspace(0, 1, max(len(users),1))
    iy = np.linspace(0, 1, max(len(items),1))
    for i, u in enumerate(users):
        pos[u] = (0.0, uy[i % len(uy)])
    for j, it in enumerate(items):
        pos[it] = (1.0, iy[j % len(iy)])
    return pos

# ===== (A) 3패널 비교(좌우로 나란히) =====
def plot_three_side_by_side(top_align, top_anchor, top_aa, df_pos, out_path,
                            max_edges_per_user=20, layout='bipartite'):
    # 각 모델 서브그래프 데이터
    uA, iA, eA = subgraph_edges_for_users(top_align, df_pos, max_edges_per_user)
    uB, iB, eB = subgraph_edges_for_users(top_anchor, df_pos, max_edges_per_user)
    uC, iC, eC = subgraph_edges_for_users(top_aa,    df_pos, max_edges_per_user)

    figs = plt.figure(figsize=(18, 6))
    axes = [figs.add_subplot(1,3,k+1) for k in range(3)]
    titles = ['AlignRec', 'Anchor', 'AlignRec+Anchor']
    node_sets = [(uA,iA,eA),(uB,iB,eB),(uC,iC,eC)]

    for ax, title, (u,i,e) in zip(axes, titles, node_sets):
        G = nx.Graph()
        G.add_nodes_from(u, bipartite=0)
        G.add_nodes_from(i, bipartite=1)
        G.add_edges_from(e)
        if layout == 'bipartite':
            pos = make_bipartite_layout(u, i)
        else:
            pos = nx.spring_layout(G, k=0.2, iterations=50, seed=42)

        nx.draw_networkx_nodes(G, pos, nodelist=u, node_color='lightblue', node_size=120, ax=ax, label='users')
        nx.draw_networkx_nodes(G, pos, nodelist=i, node_color='salmon',    node_size=60,  ax=ax, label='items')
        nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.4, ax=ax)
        ax.set_title(f'{title} (Top-{len(u)})')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[VIS] saved → {out_path}')

# ===== (B) 한 그래프에 3모델 오버레이(엣지 색상으로 구분) =====
def plot_overlay_three_models(top_align, top_anchor, top_aa, df_pos, out_path,
                              max_edges_per_user=20, layout='bipartite'):
    # 각 모델 서브그래프 데이터
    uA, iA, eA = subgraph_edges_for_users(top_align, df_pos, max_edges_per_user)
    uB, iB, eB = subgraph_edges_for_users(top_anchor, df_pos, max_edges_per_user)
    uC, iC, eC = subgraph_edges_for_users(top_aa,    df_pos, max_edges_per_user)

    # 전체 노드(유니온)로 단일 그래프 구성
    users_all = sorted(set(uA) | set(uB) | set(uC))
    items_all = sorted(set(iA) | set(iB) | set(iC))

    GA = nx.Graph(); GA.add_nodes_from(users_all, bipartite=0); GA.add_nodes_from(items_all, bipartite=1); GA.add_edges_from(eA)
    GB = nx.Graph(); GB.add_nodes_from(users_all, bipartite=0); GB.add_nodes_from(items_all, bipartite=1); GB.add_edges_from(eB)
    GC = nx.Graph(); GC.add_nodes_from(users_all, bipartite=0); GC.add_nodes_from(items_all, bipartite=1); GC.add_edges_from(eC)

    # 동일 레이아웃(좌우 분리 or spring)로 그려 공정 비교
    if layout == 'bipartite':
        pos = make_bipartite_layout(users_all, items_all)
    else:
        # 스프링 레이아웃은 결합 그래프 기준으로 한 번만 계산
        G_union = nx.Graph()
        G_union.add_nodes_from(users_all, bipartite=0)
        G_union.add_nodes_from(items_all, bipartite=1)
        G_union.add_edges_from(eA + eB + eC)
        pos = nx.spring_layout(G_union, k=0.25, iterations=80, seed=42)

    plt.figure(figsize=(12, 8))
    # 노드는 한 번만 그림
    nx.draw_networkx_nodes(GA, pos, nodelist=users_all, node_color='lightblue', node_size=120, alpha=0.9, label='users')
    nx.draw_networkx_nodes(GA, pos, nodelist=items_all, node_color='salmon',    node_size=60,  alpha=0.85, label='items')

    # 엣지는 모델별 색상
    nx.draw_networkx_edges(GA, pos, edge_color='#1f77b4', width=0.8, alpha=0.6, label='AlignRec')
    nx.draw_networkx_edges(GB, pos, edge_color='#2ca02c', width=0.8, alpha=0.6, label='Anchor')
    nx.draw_networkx_edges(GC, pos, edge_color='#ff7f0e', width=0.8, alpha=0.6, label='AlignRec+Anchor')

    plt.title(f'Overlay: Top-{len(top_align)} users per model')
    plt.axis('off')
    plt.legend(loc='upper center', ncol=4, frameon=False)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[VIS] saved → {out_path}')


# ===== 실행 =====
plot_three_side_by_side(
    top_align, top_anchor, top_aa, df_pos,
    out_path=os.path.join(OUT_DIR, 'topK_subgraphs_3panels.png'),
    max_edges_per_user=20, layout='bipartite'
)

plot_overlay_three_models(
    top_align, top_anchor, top_aa, df_pos,
    out_path=os.path.join(OUT_DIR, 'topK_subgraphs_overlay.png'),
    max_edges_per_user=20, layout='bipartite'   # 'spring'으로 바꿔도 됨
)