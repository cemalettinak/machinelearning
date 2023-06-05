# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:27:30 2023

@author: cemalettin
"""


gb_pred = gb_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
gb_acc = accuracy_score(y_test, gb_pred)
gb_prec = precision_score(y_test, gb_pred, average='weighted', zero_division=1)
gb_rec = recall_score(y_test, gb_pred, average='weighted', zero_division=1)
gb_f1 = f1_score(y_test, gb_pred, average='weighted', zero_division=1)
print("\nGradient Boosting:\nAccuracy:", gb_acc, "\nPrecision:", gb_prec, "\nRecall:", gb_rec, "\nF1 Score:", gb_f1)
