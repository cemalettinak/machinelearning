"""
# -*- coding: utf-8 -*-

Created on Sat May 20 13:18:35 2023

@author: cemalettin


"""


import matplotlib.pyplot as plt

train_Accuracy = []
test_Accuracy = []

for train_predict, test_predict in zip(gb_model.staged_predict(X_train), gb_model.staged_predict(X_test)):
    train_Accuracy.append(accuracy_score(y_train, train_predict))
    test_Accuracy.append(accuracy_score(y_test, test_predict))

plt.figure(figsize=(10, 6))
plt.plot(train_Accuracy, label="Train")
plt.plot(test_Accuracy, label="Test")
plt.title("Accuracy per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
