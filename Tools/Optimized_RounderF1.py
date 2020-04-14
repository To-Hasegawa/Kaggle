# -*- coding: utf-8 -*-
#ref:https://www.kaggle.com/artgor/eda-and-model-f1-optimization
import numpy as np
import scipy as sp
from functools import partial
from sklearn import metrics
from sklearn.metrics import f1_score

class OptimizedRounderF1(object):
    """
    An optimizer for rounding thresholds
    to maximize f1 score
    """
    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -metrics.f1_score(y, X_p, average='macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

optR = OptimizedRounderF1()
optR.fit(y_train_pred.reshape(-1,), y_train_true)
coefficients=optR.coefficients()
print(coefficients)

opt_pred = optR.predict(y_train_pred.reshape(-1,), coefficients)
print('f1', metrics.f1_score(y_train_true, opt_pred, average = 'macro'))

test_preds=lgb_model.y_pred
opt_test_preds = optR.predict(test_preds.reshape(-1,), coefficients)