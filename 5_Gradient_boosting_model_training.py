# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:27:30 2023

@author: cemalettin
"""



from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
gb_model.fit(X_train, y_train)
