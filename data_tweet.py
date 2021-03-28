from sklearn.linear_model import LinearRegression
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import glob

csv_files = glob.glob('*.csv')
csv_files

df = pd.DataFrame()
for data in csv_files:
    df_new = pd.read_csv(data)
    df = pd.concat([df, df_new])

df.shape
# (1291, 40)

df.columns

data_list = ['ツイートID', 'ツイート本文', '時間', 'インプレッション', 'エンゲージメント',
      'エンゲージメント率', 'リツイート', 'いいね', 'ユーザープロフィールクリック', 'URLクリック数',
      'ハッシュタグクリック', '詳細クリック', 'メディアの再生数', 'メディアのエンゲージメント数']

len(data_list)

data_list_columns = ['ID', 'text', 'date', 'impression', 'engagement',
      'engagement_ratio', 'RT', 'fav', 'profile_click', 'URL_click',
      'hashtag_click', 'detail_click', 'media_click', 'media_engagement']

len(data_list_columns)

df = df[data_list]
df.columns = data_list_columns
df

df.info()

sns.pairplot(df.drop('ID', axis=1))

corr_data = df.corr()
corr_data

plt.figure(figsize=(20, 10))
sns.heatmap(corr_data, annot=True)

df.describe()

fav_mean = df['fav'].mean()
# 不偏標準偏差（母集団の分散が不明であるときの母分散の推定値）　https://ai-trend.jp/basic-study/estimator/unbiasedness/
fav_std = df['fav'].std()
fav_std

sigma3_p = fav_mean + fav_std*3
sigma3_n = fav_mean - fav_std*3

print(f'外れ値+3σ:{sigma3_p}')
print(f'外れ値-3σ:{sigma3_n}')

df_3sigma = df[(df['fav'] <sigma3_p) & (df['fav'] >1)]
df_3sigma

df_3sigma['fav'].value_counts()

sns.pairplot(df_3sigma.drop('ID', axis=1), corner=True)

plt.figure(figsize=(20, 10))
sns.heatmap(corr_data, annot=True)

sns.pairplot(df_3sigma, x_vars=["impression", "fav", "RT", 'hashtag_click', 'media_engagement'], y_vars=[
             "profile_click"], height=5, aspect=.8, kind="reg")

sns.lmplot(x="impression", y="profile_click", data=df_3sigma, x_estimator=np.mean)

df_3sigma.hist(figsize=(20, 10))

sns.distplot(df_3sigma['fav'])


follower = {
    '2020-12': 208,
    '2021-01': 235,
    '2021-02': 176,
    '2021-03': 127,
}

def date_day(x):
   return x.split(' ')[0].split('-')[0]+'-'+x.split(' ')[0].split('-')[1]

df_3sigma['year_month'] = df_3sigma['date'].apply(date_day)

df_3sigma_month = df_3sigma.groupby('year_month').sum()
df_3sigma_month

df_3sigma_month['followers'] = pd.DataFrame(follower.values(), index=df_3sigma_month.index)
df_3sigma_month['followers_per_profile_click'] = df_3sigma_month['followers']/df_3sigma_month['profile_click']
df_3sigma_month

1/df_3sigma_month['followers_per_profile_click'].mean()

X = df_3sigma[['impression']]
y = df_3sigma['profile_click']


# 単回帰分析
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
y_pred[:10]

target_profile_click = 1/df_3sigma_month['followers_per_profile_click'].mean()
x_ = (target_profile_click - model.intercept_)/model.coef_[0]

x_

# 単回帰分析
X = df_3sigma[['fav']]
y = df_3sigma['profile_click']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
x_ = (target_profile_click - model.intercept_)/model.coef_[0]

x_

