
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:27:30 2023

@author: cemalettin
"""


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred, average='weighted', zero_division=1)
dt_rec = recall_score(y_test, dt_pred, average='weighted', zero_division=1)
dt_f1 = f1_score(y_test, dt_pred, average='weighted', zero_division=1)
print("\nDecision Tree:\nAccuracy:", dt_acc, "\nPrecision:", dt_prec, "\nRecall:", dt_rec, "\nF1 Score:", dt_f1)
