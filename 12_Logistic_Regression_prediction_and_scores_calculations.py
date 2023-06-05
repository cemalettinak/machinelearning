
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:07:47 2023

@author: cemalettin
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Logistic Regression modeliyle tahmin ve sonuçlar
lr_pred = pipe.predict(X_test)
# Normalize şekilde tahmin yapmak için pipe ile tahmin yapıyoruz
lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred, average='weighted', zero_division=1)
lr_rec = recall_score(y_test, lr_pred, average='weighted', zero_division=1)
lr_f1 = f1_score(y_test, lr_pred, average='weighted', zero_division=1)
print("\nLogistic Regression:\nAccuracy:", lr_acc, "\nPrecision:", lr_prec, "\nRecall:", lr_rec, "\nF1 Score:", lr_f1)
