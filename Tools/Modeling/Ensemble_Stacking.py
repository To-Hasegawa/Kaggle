# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
"""
StackingではInputとしてトレーニングデータの予測結果と答え、及びテストデータに対する予測値を使用する。
複数のモデルのTraning dataとTest Dataの予測結果をもとに、新たに線形回帰モデルを作成し、
効果的にモデルを組み合わせることで最終的な予測精度を上げる。
ここでは各モデルの予測値からなる1層目のデータ（7個）をもとに、2層目で線形回帰モデルを作る
2層のStackingモデルを作る。
"""

#get training data
train=pd.read_csv("***.csv")
train_y=train["open_channels"]

#Prepare ensemble data--prediction of training and test data
#load training prediction data
df_1= pd.read_csv("***.csv")
df_2= pd.read_csv("***.csv")
df_3= pd.read_csv("***.csv")
df_4= pd.read_csv("***.csv")
df_5= pd.read_csv("***.csv")
df_6= pd.read_csv("***.csv")
df_7= pd.read_csv("***.csv")

train_pred_1=df_1["y_pred"]
train_pred_2=df_2["y_pred"]
train_pred_3=df_3["y_pred"]
train_pred_4=df_4["y_pred"]
train_pred_5=df_5["y_pred"]
train_pred_6=df_6["y_pred"]
train_pred_7=df_7["y_pred"]

#load test_prediction data
df_1_test= pd.read_csv("***.csv")
df_2_test= pd.read_csv("***.csv")
df_3_test= pd.read_csv("***.csv")
df_4_test= pd.read_csv("***.csv")
df_5_test= pd.read_csv("***.csv")
df_6_test= pd.read_csv("***.csv")
df_7_test= pd.read_csv("***.csv")

test_pred_1=df_1_test["open_channels"]
test_pred_2=df_2_test["open_channels"]
test_pred_3=df_3_test["open_channels"]
test_pred_4=df_4_test["open_channels"]
test_pred_5=df_5_test["open_channels"]
test_pred_6=df_6_test["open_channels"]
test_pred_7=df_7_test["open_channels"]

#from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
#今回はすでにCVもして予測済みのデータを使用するので以下のpredict_cvは不使用
#def predict_cv(model, train_x, train_y, test_x):
#    preds = []
#    preds_test = []
#    va_idxes = []

#    kf = KFold(n_splits=4, shuffle=True, random_state=71)

#    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
#    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
#        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
#        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        #model.fit(tr_x, tr_y, va_x, va_y)
#        pred = model.oof_pred[va_idx]
#        preds.append(pred)
#        pred_test = model.y_pred
#        preds_test.append(pred_test)
#        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
#    va_idxes = np.concatenate(va_idxes)
#    preds = np.concatenate(preds, axis=0)
#    order = np.argsort(va_idxes)
#    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
#    preds_test = np.mean(preds_test, axis=0)

#    return pred_train, preds_test


# 1層目
# pred_train_1a, pred_train_1b...は、学習データのクロスバリデーションでの予測値
# pred_test_1a, pred_test_1b...は、テストデータの予測値
# 1層目のモデルの評価(F1-score)
from sklearn.metrics import f1_score
print(f'rmse: {f1_score(train_y, train_pred_1):.4f}')
print(f'rmse: {f1_score(train_y, train_pred_2):.4f}')
print(f'rmse: {f1_score(train_y, train_pred_3):.4f}')
print(f'rmse: {f1_score(train_y, train_pred_4):.4f}')
print(f'rmse: {f1_score(train_y, train_pred_5):.4f}')
print(f'rmse: {f1_score(train_y, train_pred_6):.4f}')
print(f'rmse: {f1_score(train_y, train_pred_7):.4f}')



# 予測値を特徴量としてデータフレームを作成
train_x_2 = pd.DataFrame({'pred_1': train_pred_1, 'pred_2': train_pred_2,'pred_3': train_pred_3,'pred_4': train_pred_4,'pred_5': train_pred_5,'pred_6': train_pred_6,'pred_7': train_pred_7})
test_x_2 = pd.DataFrame({'pred_1': test_pred_1, 'pred_2': test_pred_2,'pred_3': test_pred_3,'pred_4': test_pred_4,'pred_5': test_pred_5,'pred_6': test_pred_6,'pred_7': test_pred_7})


# 2層目
#from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv2(model, train_x, train_y, test_x):
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=71)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test

# 2層目のモデル(Ridge回帰)
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
class Model2Linear:

    def __init__(self):
        self.model = None
        self.scaler = None

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.scaler = StandardScaler()
        self.scaler.fit(tr_x)
        tr_x = self.scaler.transform(tr_x)
        self.model = Ridge()
        self.model.fit(tr_x, tr_y)

    def predict(self, x):
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred
    
# pred_train_2は、2層目のモデルの学習データのクロスバリデーションでの予測値
# pred_test_2は、2層目のモデルのテストデータの予測値
model_2 = Model2Linear()
pred_train_2, pred_test_2 = predict_cv2(model_2, train_x_2, train_y, test_x_2)
print(f1_score(train_y, pred_train_2, average='macro'))

final_pred=pred_test_2