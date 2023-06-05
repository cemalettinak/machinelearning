# -*- coding: utf-8 -*-
"""

@author: cemalettin
"""

import matplotlib.pyplot as plt

# Calculating accuracy values
train_scores = []
test_scores = []

for i in range(1, 6):
     X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
     pipe.fit(X_train_subset, y_train_subset)
     train_pred = pipe.predict(X_train_subset)
     test_pred = pipe.predict(X_test)
     train_scores.append(accuracy_score(y_train_subset, train_pred))
     test_scores.append(accuracy_score(y_test, test_pred))

# Creating a performance chart
epochs = [1, 2, 3, 4, 5]
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_scores, 'o-', label='Training Achievement')
plt.plot(epochs, test_scores, 'o-', label='Test Success')
plt.xlabel('Epochs')
plt.ylabel('Accurancy')
plt.title('Logistics Regression Model Performance Chart')
plt.ylim(0.6, 0.8)
plt.legend()
plt.show()
