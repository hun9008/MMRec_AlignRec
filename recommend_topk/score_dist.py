import pandas as pd
import numpy as np

df = pd.read_csv('user_metrics_k20_alignrec.csv')
df['mean_score'] = (df['recall@20'] + df['ndcg@20']) / 2.0

score = df['mean_score']

# 0점과 1점 라벨링
labels = pd.Series(index=df.index, dtype='object')
labels[score == 0.0] = 'zero'
labels[score == 1.0] = 'one'

# 0<score<1 구간만 qcut 4분위
mask_between = (score > 0.0) & (score < 1.0)
mid_scores = score[mask_between]

if len(mid_scores) > 0:
    # 중간구간을 4분위로 나눔
    codes, bins = pd.qcut(mid_scores, q=4, labels=False, retbins=True, duplicates='drop')
    nbin = len(bins)-1
    qlabels = [f"Q{i+1}" for i in range(nbin)]
    labels[mask_between] = pd.Series(codes,index=mid_scores.index).map(lambda x: qlabels[int(x)])
else:
    print("중간 점수 구간이 없습니다.")

df['score_bin'] = labels

print(df['score_bin'].value_counts())