# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:27:30 2023

@author: cemalettin
"""
#Prediction y
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, average='weighted', zero_division=1)
rf_rec = recall_score(y_test, rf_pred, average='weighted', zero_division=1)
rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=1)
print("\nRandom Forest:\nAccuracy:", rf_acc, "\nPrecision:", rf_prec, "\nRecall:", rf_rec, "\nF1 Score:", rf_f1)
