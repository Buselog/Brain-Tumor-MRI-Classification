import numpy as np
from sklearn.metrics import classification_report
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
EarlyStopping = tf.keras.callbacks.EarlyStopping
from utils.preprocess import load_data  # Veri işleme fonksiyonunu içe aktar
from models.model import create_ann_model  # Modeli içe aktar

# Veri setini yükle
dataset_path = "dataset/Training/"  # Kendi veri yolunu kullan
X_train, X_test, y_train, y_test = load_data(dataset_path)

# Modeli oluştur
input_shape = X_train.shape[1]  # Giriş boyutu (düzleştirilmiş)
model = create_ann_model(input_shape)

# Erken durdurma (Overfitting'i önlemek için)
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Modeli eğit
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[early_stopping], verbose=1)

# Modeli değerlendir
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Doğruluk: {test_acc:.4f}")

# Sınıflandırma raporunu yazdır
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("Sınıflandırma Raporu:\n", classification_report(y_true, y_pred, target_names=["No Tumor", "Glioma", "Meningioma", "Pituitary"]))

# Eğitilmiş modeli kaydet
model.save("brain_tumor_ann.h5")
print("Model kaydedildi: brain_tumor_ann.h5")


# import os  # (Dosya ve klasör işlemleri için)
# Matris işlemleri ve matematiksel hesaplamalar, Veri normalizasyonu, istatistiksel işlemler gibi işlemler.
# import numpy as np
# import matplotlib.pyplot as plt  # Veri görselleştirme için kullanılır.
# from sklearn.model_selection import train_test_split
# Modelin gerçek dünyada nasıl performans gösterdiğini ölçmek için verinin bir kısmını model eğitimi dışında tutar.
# import tensorflow as tf
# to_categorical = tf.keras.utils.to_categorical
