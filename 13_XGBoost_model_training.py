# -*- coding: utf-8 -*-
"""
@author: cemalettin
"""



# XGBoost model library
from xgboost import XGBClassifier

# Training and evaluating the XGBoost model
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train, verbose=False)
