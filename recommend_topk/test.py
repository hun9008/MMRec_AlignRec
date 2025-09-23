import numpy as np
import pandas as pd

path = '../data/baby/user_graph_dict.npy'
g = np.load(path, allow_pickle=True).item()  # dict로 로드되는지 확인

# 임의 유저 하나 확인
some_uid = next(iter(g.keys()))
print(type(some_uid), type(g[some_uid]), g[some_uid][:5] if hasattr(g[some_uid], '__len__') else g[some_uid])

item_path = '../data/baby/i_id_mapping.csv'
user_path = '../data/baby/u_id_mapping.csv'

df_item = pd.read_csv(item_path)
df_user = pd.read_csv(user_path)

print(f"아이템 개수: {len(df_item)}, 유저 개수: {len(df_user)}")

inter_path = '../data/baby/baby.inter'

# 어떤 구분자로 되어 있는지 모르니 sep='\t'나 ','로 먼저 시도
df_inter = pd.read_csv(inter_path, sep='\t')  # 보통 RecBole .inter는 탭 구분

print("전체 행 수(상호작용 수):", len(df_inter))
print("컬럼:", df_inter.columns.tolist())
print(df_inter.head())

# 실제 컬럼명에 맞게 사용
print("유저 수:", df_inter['userID'].nunique())
print("아이템 수:", df_inter['itemID'].nunique())

# rating의 통계치
print(df_inter['rating'].describe())
