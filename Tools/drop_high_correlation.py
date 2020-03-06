# -*- coding: utf-8 -*-
#load data
train = read.csv("***")
train.shape
# call feature engineering function
features = train.columns
#Drop high correlation features
counter = 0
to_remove = []
for feat_a in features:
    for feat_b in features:
        if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
            c = np.corrcoef(train[feat_a], train[feat_b])[0][1]
            if c > 0.995:
                counter += 1
                to_remove.append(feat_b)
                print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))

#remove high correlation features
features = [x for x in features if x not in to_remove]
train = train[features]
train.shape