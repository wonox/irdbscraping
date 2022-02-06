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



# ソースなど
https://github.com/wonox/irdbscraping/blob/main/irdbscraping.ipynb