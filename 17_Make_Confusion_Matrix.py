
"""
Tüm grafikler için matplotlib kütüphanesinin pyplot modülü içe aktarılmaktadır. Ardından bu grafik için seaborn kütüphanesi de içe aktarılmıştır. 
Sonrasında gerçek çıktı değeri ile o çıktı değerinin tahmini ve çapraz tabloya ait diğer özellikler crosstab fonksiyonu içerisine yazılmıştır.
Boyut ayarlaması, başlık ve ısı haritası üzerine renklendirme ve yazı yazdırma ayarlamaları yapılarak show fonksiyonu ile grafik çıktısı üretilmiştir.
Random Forest modeli için rf_pred, Gradient Boosting modeli için gb_pred, XGBoost modeli için xgb_pred, Desicion Tree modeli için dt_pred, Logistic Regression modeli için lr_pred, Voting modeli için voting_pred vb. şekilde değiştirerek diğer ısı haritaları oluşturulabilir.
"""


import seaborn as sns
import matplotlib.pyplot as plt

confusion_matrix = pd.crosstab(y_test, voting_pred, rownames=['True'], colnames=['Forecast'], normalize='index')
# Create heatmaps
plt.figure(figsize=(10, 8))
plt.title("Confusion Matrix - Voting")
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".2f")
plt.show()
