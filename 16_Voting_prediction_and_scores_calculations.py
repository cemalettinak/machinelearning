


# -*- coding: utf-8 -*-
"""
@author: cemalettin
"""


voting_pred = voting_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

voting_acc = accuracy_score(y_test, voting_pred)
voting_prec = precision_score(y_test, voting_pred, average='weighted', zero_division=1)
voting_rec = recall_score(y_test, voting_pred, average='weighted', zero_division=1)
voting_f1 = f1_score(y_test, voting_pred, average='weighted', zero_division=1)

print("\nVoting Classifier:\nAccuracy:", voting_acc, "\nPrecision:", voting_prec, "\nRecall:", voting_rec, "\nF1 Score:", voting_f1)
