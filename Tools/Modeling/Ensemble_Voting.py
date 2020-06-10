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


#データから予測値のみを抽出
model1=df_1["open_channels"]
model2=df_2["open_channels"]
model3=df_3["open_channels"]
model4=df_4["open_channels"]
model5=df_5["open_channels"]
model6=df_6["open_channels"]
model7=df_7["open_channels"]

#convert to one-hot
#カテゴリーの予測の場合、以下のようにデータを一旦One-Hot形式に変換する。
#出力は０〜１０なので11カテゴリーに変換
from keras.utils import to_categorical
model1_onehot=to_categorical(model1, 11)
model2_onehot=to_categorical(model2, 11)
model3_onehot=to_categorical(model3, 11)
model4_onehot=to_categorical(model4, 11)
model5_onehot=to_categorical(model5, 11)
model6_onehot=to_categorical(model6, 11)
model7_onehot=to_categorical(model7, 11)

#各モデルに重み付け
k1=0.1
k2=0.1
k3=0.2
k4=0.2
k5=0.1
k6=0.2
k7=0.1

#Voting
voting_result=[]
for i in range(len(model1)):
    voting = np.argmax(model1_onehot[i]*k1 + model2_onehot[i]*k2+ model3_onehot[i]*k3+ model4_onehot[i]*k4+ model5_onehot[i]*k5+ model6_onehot[i]*k6+ model7_onehot[i]*k7)
    voting_result.append(voting)

final_pred=voting_result