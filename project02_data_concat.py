# -----------------------  Step 0. 준비
import pandas as pd
import glob
import datetime
import sys

# -----------------------  Step 1. 데이터 불러오기
data_path = glob.glob('./crawling_data/*.csv')
print(data_path)
# -----------------------  Step 2. 데이터 합치기
df = pd.DataFrame()
for path in data_path[0:]:
    df_temp = pd.read_csv(path)
    df = pd.concat([df, df_temp])
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)
df.sort_values(['분류'])
df.to_csv('./test_data/data_concat.csv', index=False, encoding='utf-8-sig')
