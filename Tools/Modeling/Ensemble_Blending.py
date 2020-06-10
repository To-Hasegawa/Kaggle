# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_1= pd.read_csv("***.csv")
df_2= pd.read_csv("***.csv")
df_3= pd.read_csv("***.csv")
df_4= pd.read_csv("***.csv")
df_5= pd.read_csv("***.csv")
df_6= pd.read_csv("***.csv")
df_7= pd.read_csv("***.csv")

#各モデルに重み付け
weights = {'w1': 0.1,
           'w2': 0.2, 
           'w3': 0.1, 
           'w4': 0.2,
           'w5':0.2,
           'w6':0.1,
           'w7':0.1}

#予測値に重みをかけて和を取る（Blendする）
final_pred = (df_1["open_channels"] * weights['w1']) +(df_2["open_channels"] * weights['w2'])+(df_3["open_channels"] * weights['w3'])+(df_4["open_channels"] * weights['w4'])+(df_5["open_channels"] * weights['w5'])+(df_6["open_channels"] * weights['w6'])+(df_7["open_channels"] * weights['w7'])

#Blendingの結果の最終的な出力がカテゴリーの予測なら端数を丸めてintに変換
def pred_proc(pred):
    pred = np.round(np.clip(pred, 0, 10))
    return pred.astype(int)

final_pred=pred_proc(final_pred)