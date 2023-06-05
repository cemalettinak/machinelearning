
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:07:47 2023

@author: cemalettin
"""



from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
# Pipeline oluşturma
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
