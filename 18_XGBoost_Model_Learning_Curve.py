
"""
Accurancy Chart
"""
# Creating empty lists to save success rates for different number of trees
num_trees = [0, 4, 8, 12, 16, 50]
train_scores = []
test_scores = []

# Recording the XGBoost model training and test success rates for each number of trees
for trees in num_trees:
     xgb_model = XGBClassifier(n_estimators=trees, random_state=0)
     xgb_model.fit(X_train, y_train)
     train_pred = xgb_model.predict(X_train)
     test_pred = xgb_model.predict(X_test)
     train_acc = accuracy_score(y_train, train_pred)
     test_acc = accuracy_score(y_test, test_pred)
     train_scores.append(train_acc)
     test_scores.append(test_acc)

from matplotlib import pyplot as plt
# Graph drawing
plt.plot(num_trees, train_scores, label="Training")
plt.plot(num_trees, test_scores, label="Test")
plt.xlabel('Number of Trees')
plt.ylabel('Accurancy')
plt.title('XGBoost Model Performance Chart')
plt.legend()
plt.ylim(0, 1)
plt.show()
