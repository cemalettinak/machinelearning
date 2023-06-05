"""
# -*- coding: utf-8 -*-

Created on Sat May 20 13:18:35 2023

@author: cemalettin


"""


from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt

# Regresyon için Gradient Boosting modeli (Least Squares loss)
gb_model_reg = GradientBoostingRegressor(loss='ls')

# İkili sınıflandırma için Gradient Boosting modeli (Deviance loss)
gb_model_bin = GradientBoostingClassifier(loss='deviance')
 
# Kayıp grafiği
gb_model_Loss = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model_Loss.fit(X_train, y_train)

train_loss = []
test_loss = []
for train_predict_proba, test_predict_proba in zip(gb_model_Loss.staged_decision_function(X_train), gb_model_Loss.staged_decision_function(X_test)):
    train_loss.append(gb_model_Loss.loss_(y_train, train_predict_proba))
    test_loss.append(gb_model_Loss.loss_(y_test, test_predict_proba))

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Train")
plt.plot(test_loss, label="Test")
plt.title("Loss per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()
