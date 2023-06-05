# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:50:30 2023

@author: cemalettin
"""

# Desicion Tree
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier
# Decision Tree modelinin eğitimi ve testi
dt_model = DecisionTreeClassifier(random_state=42, min_samples_leaf=5)
dt_model.fit(X_egitim, y_egitim)
