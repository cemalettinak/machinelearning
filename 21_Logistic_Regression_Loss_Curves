
# -*- coding: utf-8 -*-
"""

@author: cemalettin
"""




from sklearn.metrics import log_loss

epochs = [10, 20, 30, 40, 50]

train_losses = []
test_losses = []

# Eğitim ve test verilerini normalize edin
X_train_normalized = pipe.named_steps['standardscaler'].transform(X_train)
X_test_normalized = pipe.named_steps['standardscaler'].transform(X_test)

for epoch in epochs:
    model = LogisticRegression(max_iter=epoch)
    model.fit(X_train_normalized, y_train)
    train_pred = model.predict_proba(X_train_normalized)
    test_pred = model.predict_proba(X_test_normalized)
    train_loss = log_loss(y_train, train_pred)
    test_loss = log_loss(y_test, test_pred)
    train_losses.append(train_loss)
    test_losses.append(test_loss)


# Kayıp grafiğini çizin
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, 'o-',  label='Training Loss')
plt.plot(epochs, test_losses, 'o-',  label='Test Loss')
plt.title('Lojistik Regresyon Modeli Başarım Kaybı Grafiği')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim(0.6, 0.80)  # Y ekseni aralığını (0, 1) olarak ayarlayın
# plt.grid(True)
plt.show()
