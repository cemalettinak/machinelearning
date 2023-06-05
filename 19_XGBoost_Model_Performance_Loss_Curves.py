


# define the datasets to evaluate each iteration

eval_set = [(X_train, y_train), (X_test, y_test)]

# fit the model
xgb_model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=eval_set, verbose=False)


# retrieve performance metrics
results = xgb_model.evals_result()
train_loss = results['validation_0']['mlogloss']
test_loss = results['validation_1']['mlogloss']

from matplotlib import pyplot as plt
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, test_loss, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()
