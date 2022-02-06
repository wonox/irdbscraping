# IRDBからスクレイピングしてPANDASで機関リポジトリ登録コンテンツの分析をする
## 概要
IRDB（https://irdb.nii.ac.jp/）から、JPCOAR加盟機関のデータをスクレイピングして、PANDASで若干の分析を行う。
JPCOAR加盟機関は、https://jpcoar.repo.nii.ac.jp/page/40 から抽出し、「図書館」などの文字列を削除し、機関名を抽出

## スクレイピング
機関名をハードコーディングしたくなかったので、機関名を羅列したテキストファイルを読み込んで配列に代入することにした。
```python
import pickle

inst_array = []
# insts = 　　[　　'東京大学',　　'東北大学',　　'東京工業大学'　　]
# 機関名を羅列したテキストファイルを読み込んで配列に代入
with open('kikanmei.txt', 'r',encoding="utf-8_sig") as f:
    insts = f.read().split("\n")

for inst in insts:
    inst_array.append(inst)

# 配列（list）をpickleに保存
with open("inst_array.pkl","wb") as f:
    pickle.dump(inst_array, f)
```

上で保存したpickleから配列を読みだして、スクレイピングする。
600件で18分程度(sleep(1))かかった。

```python
# 保存済みの機関名の配列を読み込み
with open('inst_array.pkl', 'rb') as f:
    institutions = pickle.load(f)
    
## 必要なライブラリのインポート ————–
# スクレイピング
from bs4 import BeautifulSoup
import time
import requests
import traceback
import logging
logger = logging.getLogger()

def main():
  type_list = {}
  #type_list = scrap('東北大学')
  for kikan in institutions:
    time.sleep(1)
    # 1秒スリープ
    type_list[kikan] = scrap(kikan)
    # type_list = scrap(kikan)
  # print(type_list)
  return type_list

def scrap(kikan):
  URL = 'https://irdb.nii.ac.jp/search?kikanid=' + kikan
  # スクレイピング —————————
  try:
    # URLを開く
    response = requests.get(URL)
    # HTMLを取得
    soup = BeautifulSoup(response.content, 'html.parser')
    logger.info('webサイトと接続できました')
  except:
    tb = traceback.format_exc()
    print(tb)
    print('Webサイトとの接続が確立できませんでした')

  # HTML上の任意の箇所を抽出
  '''
  <section class="facet-inactive block-facet--links block block-facets block-facet-blockresourcetype clearfix"
   id="block-resourcetype">
  <li class="facet-item"><a href="/search?kikanid=%E6%9D%B1%E4%BA%AC%E5%A4%A7%E5%AD%A6&amp;
  facet%5B0%5D=typefc%3Abook" rel="nofollow" 
  data-drupal-facet-item-id="typefc-book" data-drupal-facet-item-value="book"
  data-drupal-facet-item-count="109"><span class="facet-item__value">book</span>
  <span class="facet-item__count">(109)</span>
  '''
  soup2 = soup.select('#block-resourcetype a')
  # テキストのみをリストで取得
  soup_list = [[kikan, x.attrs['data-drupal-facet-item-value'], x.attrs['data-drupal-facet-item-count']] for x in soup2]
  return soup_list

# institutions = ['東京大学','東北大学','東京工業大学'] とするのをpickle　で置き換えている

if __name__ == "__main__":
    type_list = main()
```

## データ加工
機関名['institutions']をindexとし、資源タイプ['typename']ごとの数を表にする。

```python
# データ加工
import pandas as pd
# list化
listOfValues = list(type_list.values())
# フラット化
flat_list = [item for l in listOfValues for item in l]
koumoku = ['institutions', 'typename', 'nums']
# 項目名を加えて辞書化
listext = []
for v in flat_list:
    listext.append(dict(zip(koumoku,v)))

# pandasのDataFrame化
df = pd.DataFrame(listext)

# 縦持ちデータフレームdfを、横持ちに変換
pivot_orders_df = df.pivot_table(values=['nums'], index=['institutions'], columns=['typename'], aggfunc='sum')
pivot_orders_df = pivot_orders_df.rename(index={'': '全体'})
pivot_orders_df.fillna(0,inplace=True)
pivot_orders_df
# マルチインデックスを解除
#pivot_orders_df_reset = pivot_orders_df.reset_index(level=0)
pivot_orders_df.columns = pivot_orders_df.columns.droplevel(0)
#pandas 0.18.0 and higher
pivot_orders_df = pivot_orders_df.rename_axis(None, axis=1)
pivot_orders_df
# 数字に見えるものを数値化
cols = pivot_orders_df.columns
pivot_orders_df[cols] = pivot_orders_df[cols].apply(pd.to_numeric, errors='coerce')
# 前回保存したDataFrameのファイルから読み込む
import pandas as pd
pivot_orders_df = pd.read_pickle('./pivot_orders_df.pkl.gz', compression='gzip') # 圧縮有り
```

# 資源タイプ
使われている資源タイプは35個あった。
> Index(['article', 'bachelor thesis', 'book', 'book part',
>  'cartographic material', 'conference object', 'conference paper',
>  'conference poster', 'conference proceedings', 'data paper', 'dataset',
>  'departmental bulletin paper', 'doctoral thesis', 'editorial', 'image',
>  'interview', 'journal article', 'learning object', 'lecture',
>  'manuscript', 'master thesis', 'musical notation', 'newspaper', 'other',
>  'periodical', 'report', 'report part', 'research report',
>  'review article', 'software', 'sound', 'still image',
>  'technical report', 'thesis', 'working paper'],
>  dtype='object')

ただし、JPCOARスキーマ 語彙としては47ある

> 語彙 conference paper data paper departmental bulletin paper editorial journal article newspaper periodical review article software paper article book book part cartographic material map conference object conference proceedings conference poster dataset interview image still image moving image video lecture patent internal report report research report technical report policy report report part working paper data management plan sound thesis bachelor thesis master thesis doctoral thesis interactive resource learning object manuscript musical notation research proposal software technical documentation workflow other

# DataFrame の作製
```python
# 'article == 113164'は総計の数字。IRDBは検索該当なしだと全件になってしまうので。
# IRDBは、JPCOAR外のリポジトリも含んだ数字なので、総計は削除してよい
# print((pivot_orders_df.query('article == 113164')))
pivot_orders_df_2index = list(pivot_orders_df.query('article == 113164').index)
# del pivot_orders_df_2index[:2]  # 先頭から２つ[全体,＃NA]をlistから削除する場合
pivot_orders_df_2index  # 削除するindexのlist
# index で指定した行をまとめて削除したDFを生成
pivot_orders_df2 = pivot_orders_df.drop(index=pivot_orders_df_2index)
```

# 試しに階層型クラスタリングして、デンドログラムを書いてみる。
あまり役に立たない？

```python
# 階層型クラスタリング(ウォード法)
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
 
# linkage_result = linkage(pivot_orders_df2, method='ward', metric='euclidean')
linkage_result = linkage(pivot_orders_df2, method='average', metric='euclidean')
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor='w', edgecolor='k')
dendrogram(linkage_result, labels=pivot_orders_df2.index)
plt.show()
```

## 階層型クラスタリング
ユークリッド距離とウォード法または群平均法を使用してクラスタリングしてみる
average:群平均法：クラスター分析で使用される、クラスター間の距離算出方法の一つ。 2つのクラスター間で可能な全ての組み合わせにおける非類似度の平均からクラスターを形成する方法。

```python
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.datasets import load_digits

def gen_data():
    digits = load_digits()
    label_uniq = np.unique(digits.target)
    result = []
    for label in label_uniq:
        result.append(digits.data[digits.target == label].mean(axis=0))
    return result, label_uniq

def clustering_fcluster():
    X, y = gen_data()
    S = pdist(X)
    # methods = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    # Z = linkage(pivot_orders_df2, metric='euclidean', method='ward')
    Z = linkage(pivot_orders_df2, metric='euclidean', method='average')
    # Z = linkage(pivot_orders_df2, method="average")
    # criterion= ‘maxclust’、最大クラスタ数
    # ’distance’ 距離で閾値を指定
    # inconsistent (default)
    # monocrit, maxclust_monocrit
    result = fcluster(Z, t=7, criterion="maxclust")
    # result = fcluster(Z, t=5, criterion="distance")
    d = defaultdict(list)
    for i, r in enumerate(result):
        d[r].append(i)
    for k, v in d.items():
        print(k, pivot_orders_df2.index[v])
        # print(k,v) # v は行番号のリスト
        
if __name__ == "__main__":
    clustering_fcluster()
```
### 結果
> 3 Index(['お茶の水女子大学', 'こども教育宝仙大学', 'つくば国際大学', 'びわこ成蹊スポーツ大学', 'ものつくり大学',
>        'アジア成長研究所', 'デジタルハリウッド大学', 'ノートルダム清心女子大学', 'フェリス女学院大学', 'ヤマザキ動物看護大学',
>        ...
>        '鶴見大学', '鹿児島国際大学', '鹿児島大学', '鹿児島女子短期大学', '鹿児島県立短期大学', '鹿児島純心女子大学',
>        '鹿児島純心女子短期大学', '鹿屋体育大学', '麗澤大学', '麻布大学'],
>       dtype='object', name='institutions', length=602)
> 6 Index(['京都大学'], dtype='object', name='institutions')
> 4 Index(['北海道大学', '大阪大学', '早稲田大学', '東京大学', '筑波大学'], dtype='object', name='institutions')
> 5 Index(['千葉大学'], dtype='object', name='institutions')
> 1 Index(['慶應義塾大学'], dtype='object', name='institutions')
> 7 Index(['東京工業大学'], dtype='object', name='institutions')
> 2 Index(['東北大学'], dtype='object', name='institutions')

```python
# 行と列に合計値を追加する 
pivot_orders_df3 = pd.concat([pivot_orders_df2,pd.DataFrame(pivot_orders_df2.sum(axis=0),columns=['Grand Total']).T])
pivot_orders_df3 = pd.concat([pivot_orders_df2,pd.DataFrame(pivot_orders_df2.sum(axis=1),columns=['Total'])],axis=1)
pivot_orders_df3
# Totalの降順にソートする
pivot_orders_df4 = pivot_orders_df3.sort_values(by='Total', ascending=False) 
# 日付をファイル名にしてエクセルに出力
import datetime
now = datetime.datetime.now()
filename = './irdbscraping_' + now.strftime('%Y%m%d_%H%M%S') + '.xlsx'
pivot_orders_df4.to_excel('./'+filename, sheet_name='filename')
```
## 合計の概要を見てみる
```python
print(pivot_orders_df4.describe(percentiles=[0.2, 0.4, 0.6, 0.8,0.95]).loc[:,"Total"])
```

```
count       612.000000
mean       4921.506536
std       18220.891731
min           1.000000
20%         235.000000
40%         615.800000
50%         875.500000
60%        1395.200000
80%        3972.400000
95%       19454.200000
max      309580.000000
Name: Total, dtype: float64
```

# ヒストグラム

```python
import matplotlib.pyplot as plt
# タイトル追加
plt.title('機関ごと件数の合計値でヒストグラム')
# x軸にscore、y軸にfreq
plt.xlabel('機関ごとの件数')
plt.ylabel('機関数')
# plt.hist(pivot_orders_df4['Total'], range=(00, 100), bins=10)
pivot_orders_df4['Total'].hist(bins=100) # histogramを出すだけならこちらでもよい
# '紀要の百分率でヒストグラム'
# plt.hist(kiyou['journal article'], range=(00, 100), bins=10)
```

![機関ごと件数の合計値でヒストグラム](data:image/png;base64,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)

# ソースなど
https://github.com/wonox/irdbscraping/blob/main/irdbscraping.ipynb