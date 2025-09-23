import os
import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

DATA_DIR = '../data/baby'
INTER_FILE = f'{DATA_DIR}/baby.inter'
POS_THRESH = 1          # rating >= POS_THRESH 를 positive로 간주 (AlignRec 논문 재현이면 1 권장)
TOPK = 20

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


## item의 degree 분포 plot
item_counts = df_pos['item_id'].value_counts().sort_index()
plt.figure(figsize=(8,5))
plt.hist(item_counts, bins=50, log=True)
plt.xlabel('Item Positive Degree (Number of Users Interacted)')
plt.ylabel('Number of Items (log scale)')
plt.title('Item Positive Degree Distribution')
plt.grid(axis='y', linestyle='--', alpha=0.7)
os.makedirs('./analysis_out_common', exist_ok=True)
plt.savefig('./analysis_out_common/item_positive_degree_distribution.png')
plt.close()

# degree 가 400 이상 item 출력
high_degree_items = item_counts[item_counts >= 400]
print("Degree 400 이상 아이템 수:", len(high_degree_items))
print("Degree 400 이상 아이템들 (item_id: degree):")
for item_id, degree in high_degree_items.items():
    print(f"{item_id}: {degree}")

# item의 95퍼센타일, 5퍼센타일 degree
p95 = int(np.percentile(item_counts, 95))
p5  = int(np.percentile(item_counts, 5))
print(f"Item Positive Degree 95 Percentile: {p95}")
print(f"Item Positive Degree 5 Percentile : {p5}")

## user의 degree 분포 plot
user_counts = df_pos['user_id'].value_counts().sort_index()
plt.figure(figsize=(8,5))
plt.hist(user_counts, bins=50, log=True)
plt.xlabel('User Positive Degree (Number of Items Interacted)')
plt.ylabel('Number of Users (log scale)')
plt.title('User Positive Degree Distribution')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('./analysis_out_common/user_positive_degree_distribution.png')
plt.close()

# degree 가 80 이상 user 출력
high_degree_users = user_counts[user_counts >= 80]
print("Degree 80 이상 유저 수:", len(high_degree_users))
print("Degree 80 이상 유저들 (user_id: degree):")
for user_id, degree in high_degree_users.items():
    print(f"{user_id}: {degree}")

# user의 95퍼센타일, 5퍼센타일 degree
p95 = int(np.percentile(user_counts, 95))
p5  = int(np.percentile(user_counts, 5))
print(f"User Positive Degree 95 Percentile: {p95}")
print(f"User Positive Degree 5 Percentile : {p5}")

