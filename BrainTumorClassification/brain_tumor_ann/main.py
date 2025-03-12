# import os  # (Dosya ve klasör işlemleri için)
# Matris işlemleri ve matematiksel hesaplamalar, Veri normalizasyonu, istatistiksel işlemler gibi işlemler.
# import numpy as np
# import matplotlib.pyplot as plt  # Veri görselleştirme için kullanılır.
# from sklearn.model_selection import train_test_split
# Modelin gerçek dünyada nasıl performans gösterdiğini ölçmek için verinin bir kısmını model eğitimi dışında tutar.
# import tensorflow as tf
# to_categorical = tf.keras.utils.to_categorical


import matplotlib.pyplot as plt
from models.model import create_model
from utils.preprocess import load_data

# Veriyi yükle
train_data_path = "dataset/Training/"
X_train, X_test, y_train, y_test = load_data(train_data_path)

# Modeli oluştur
input_shape = X_train.shape[1]  # MLP için giriş şekli tek boyutlu
model = create_model(input_shape)

# Modeli eğit
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Eğitim grafiği çizdirme
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()

# Modeli kaydet
model.save("models/brain_tumor_ann.h5")

