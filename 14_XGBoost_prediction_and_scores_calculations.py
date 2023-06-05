
# -*- coding: utf-8 -*-
"""

@author: cemalettin
"""

# XGBoost model kütüphanesi
from xgboost import XGBClassifier

# XGBoost modelini eğitim ve değerlendirme
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train,  verbose=False)
