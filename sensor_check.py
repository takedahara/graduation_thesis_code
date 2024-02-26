# フレーム差分を計算する関数
import pandas as pd

# 表示する最大行数と列数を設定（None は制限なしを意味する）
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# CSV ファイルを読み込む
df = pd.read_csv('/home/mimamori/rpi-dev121/data/20220909_Dazaifu-1-1.csv')

# eTVOC の値の差分（絶対値の変化量）を計算
df['eTVOC_diff'] = df['eTVOC'].diff().abs()

# 差分の平均値と標準偏差を計算
mean_diff_etvoc = df['eTVOC_diff'].mean()
std_diff_etvoc = df['eTVOC_diff'].std()

# 閾値を計算（平均値 + 標準偏差の倍）
threshold_etvoc = mean_diff_etvoc + std_diff_etvoc*2

# 閾値を超える行を特定
exceed_threshold_etvoc = df[df['eTVOC_diff'] > threshold_etvoc][['Time', 'eTVOC', 'eTVOC_diff']]

# 計算した閾値と該当する行を出力
print("Calculated Threshold for eTVOC:", threshold_etvoc)
print(exceed_threshold_etvoc)
