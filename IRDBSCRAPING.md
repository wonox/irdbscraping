# IRDB からスクレイピングして PANDAS で機関リポジトリ登録コンテンツの分析をする

## 概要

IRDB（https://irdb.nii.ac.jp/）から、JPCOAR加盟機関のデータをスクレイピングして、PANDASで若干の分析を行う。
JPCOAR 加盟機関は、https://jpcoar.repo.nii.ac.jp/page/40 から抽出し、「図書館」などの文字列を削除し、機関名を抽出

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

上で保存した pickle から配列を読みだして、スクレイピングする。
600 件で 18 分程度(sleep(1))かかった。

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

機関名['institutions']を index とし、資源タイプ['typename']ごとの数を表にする。

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

使われている資源タイプは 35 個あった。

> Index(['article', 'bachelor thesis', 'book', 'book part',
>
> > 'cartographic material', 'conference object', 'conference paper',
> > 'conference poster', 'conference proceedings', 'data paper', 'dataset',
> > 'departmental bulletin paper', 'doctoral thesis', 'editorial', 'image',
> > 'interview', 'journal article', 'learning object', 'lecture',
> > 'manuscript', 'master thesis', 'musical notation', 'newspaper', 'other',
> > 'periodical', 'report', 'report part', 'research report',
> > 'review article', 'software', 'sound', 'still image',
> > 'technical report', 'thesis', 'working paper'],
> > dtype='object')

ただし、JPCOAR スキーマ 語彙としては 47 ある

```
語彙 conference paper data paper departmental bulletin paper editorial journal article newspaper periodical review article software paper article book book part cartographic material map conference object conference proceedings conference poster dataset interview image still image moving image video lecture patent internal report report research report technical report policy report report part working paper data management plan sound thesis bachelor thesis master thesis doctoral thesis interactive resource learning object manuscript musical notation research proposal software technical documentation workflow other
```

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
![デンドログラム](https://raw.githubusercontent.com/wonox/irdbscraping/main/output_dendrogram.png)

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
average:群平均法：クラスター分析で使用される、クラスター間の距離算出方法の一つ。 2 つのクラスター間で可能な全ての組み合わせにおける非類似度の平均からクラスターを形成する方法。

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

```
3 Index(['お茶の水女子大学', 'こども教育宝仙大学', 'つくば国際大学', 'びわこ成蹊スポーツ大学', 'ものつくり大学',
 'アジア成長研究所', 'デジタルハリウッド大学', 'ノートルダム清心女子大学', 'フェリス女学院大学', 'ヤマザキ動物看護大学',
 ...
 '鶴見大学', '鹿児島国際大学', '鹿児島大学', '鹿児島女子短期大学', '鹿児島県立短期大学', '鹿児島純心女子大学',
 '鹿児島純心女子短期大学', '鹿屋体育大学', '麗澤大学', '麻布大学'],
dtype='object', name='institutions', length=602)
 6 Index(['京都大学'], dtype='object', name='institutions')
 4 Index(['北海道大学', '大阪大学', '早稲田大学', '東京大学', '筑波大学'], type='object', name='institutions')
 5 Index(['千葉大学'], dtype='object', name='institutions')
 1 Index(['慶應義塾大学'], dtype='object', name='institutions')
 7 Index(['東京工業大学'], dtype='object', name='institutions')
 2 Index(['東北大学'], dtype='object', name='institutions')
```

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

![機関ごと件数の合計値でヒストグラム](https://github.com/wonox/irdbscraping/blob/main/%E6%A9%9F%E9%96%A2%E3%81%94%E3%81%A8%E4%BB%B6%E6%95%B0%E3%83%92%E3%82%B9%E3%83%88%E3%82%B0%E3%83%A9%E3%83%A0.png)

```python
# 度数分布表を一発で自動生成
# https://qiita.com/TakuTaku36/items/91032625e482f2ae6e18
import numpy as np
# def Frequency_Distribution(data, class_width=None):
def Frequency_Distribution(data, class_width):
    data = np.asarray(data)
    if class_width is None:
        class_size = int(np.log2(data.size).round()) + 1
        class_width = round((data.max() - data.min()) / class_size)

    bins = np.arange(0, data.max()+class_width+1, class_width)
    hist = np.histogram(data, bins)[0]
    cumsum = hist.cumsum()

    return pd.DataFrame({'階級値': (bins[1:] + bins[:-1]) / 2,
                         '度数': hist,
                         '累積度数': cumsum,
                         '相対度数': hist / cumsum[-1],
                         '累積相対度数': cumsum / cumsum[-1]},
                        index=pd.Index([f'{bins[i]}以上{bins[i+1]}未満'
                                        for i in range(hist.size)],
                                       name='階級'))
x = list(pivot_orders_df4['Total'])
# x = [0, 3, 3, 5, 5, 5, 5, 7, 7, 10, 11, 14, 14, 14]
class_width = None #を指定すると自動
Frequency_Distribution(x,class_width)
```

## 度数分布表
|階級   |階級値   |度数   |累積度数   |相対度数   |累積相対度数   |
|---|---|---|---|---|---|
|0以上30958未満   |15479.0   |591   |591   |0.965686   |0.965686   |
|30958以上61916未満   |	46437.0   |	14   |	605   |	0.022876   |	0.988562   |
|61916以上92874未満   |	77395.0   |	4   |	609   |	0.006536   |	0.995098   |
|92874以上123832未満   |	108353.0   |	1   |	610   |	0.001634   |	0.996732   |
|92874以上123832未満   |	108353.0   |	1   |	610   |	0.001634   |	0.996732   |
|92874以上123832未満   |	108353.0   |	1   |	610   |	0.001634   |	0.996732   |
|123832以上154790未満   |	139311.0   |	0   |	610   |	0.000000   |	0.996732   |
|92874以上123832未満   |	108353.0   |	1   |	610   |	0.001634   |	0.996732   |
|92874以上123832未満   |	108353.0   |	1   |	610   |	0.001634   |	0.996732   |
|154790以上185748未満   |	170269.0   |	0   |	610   |	0.000000   |	0.996732   |
|185748以上216706未満   |	201227.0   |	0   |	610   |	0.000000   |	0.996732   |
|216706以上247664未満   |	232185.0   |	1   |	611   |	0.001634   |	0.998366   |
|247664以上278622未満   |	263143.0   |	0   |	611   |	0.000000   |	0.998366   |
|278622以上309580未満   |	294101.0   |	0   |	611   |	0.000000   |	0.998366   |
|309580以上340538未満   |	325059.0   |	1   |	612   |	0.001634   |	1.000000   |

```python
# コンテンツ数上位50機関で構成比率の帯グラフ化
# グラフの文字化け対策
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams["font.family"] = "MS Gothic"
# df.iloc[:, 1:] = df.iloc[:, 1:].div(df['total'], axis=0).mul(100).round(2).astype(str).add(' %')
# 行ごとの百分率に変換する
pivot_orders_df5 = pivot_orders_df4.div(pivot_orders_df4['Total'], axis=0).mul(100)  # .round(2) 四捨五入
# df.drop("b", axis=1)
# pivot_orders_df5.iloc[:50,].drop("Total", axis=1).plot.bar(stacked=True)  
# .iloc[:12,] 12機関目までにする
# 判例の位置調整　https://qiita.com/matsui-k20xx/items/291400ed56a39ed63462
pivot_orders_df5.iloc[:50,].drop("Total", axis=1)\
    .plot(kind='bar', stacked=True, figsize=(10,5), width=1, linewidth=0,title='上位50機関',)\
    .legend(bbox_to_anchor=(0, -0.5), loc='upper left', borderaxespad=0, fontsize=18)
```

![コンテンツ数上位50機関で構成比率の帯グラフ](https://github.com/wonox/irdbscraping/blob/main/%E4%B8%8A%E4%BD%8D50%E6%A9%9F%E9%96%A2%E7%A9%8D%E3%81%BF%E4%B8%8A%E3%81%92%E7%99%BE%E5%88%86%E7%8E%87output.png)

```python
# コンテンツ数上位50機関の積み上げグラフ
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams["font.family"] = "MS Gothic"
# .iloc[:12,] 12機関目までにする
# 判例の位置調整　https://qiita.com/matsui-k20xx/items/291400ed56a39ed63462
pivot_orders_df4.iloc[:50,].drop("Total", axis=1)\
    .plot(kind='bar', stacked=True, figsize=(10,5), width=1, linewidth=0,title='上位50機関',)\
    .legend(bbox_to_anchor=(0, -0.5), loc='upper left', borderaxespad=0, fontsize=18)
```

![コンテンツ数上位50機関の積み上げグラフ](https://github.com/wonox/irdbscraping/blob/main/%E4%B8%8A%E4%BD%8D50%E6%A9%9F%E9%96%A2%E7%A9%8D%E3%81%BF%E4%B8%8A%E3%81%92.png)



# ソースなど

https://github.com/wonox/irdbscraping/blob/main/irdbscraping.ipynb
