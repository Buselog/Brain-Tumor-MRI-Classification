import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from utils.preprocess import load_data  
from models.model import create_ann_model  

# Veri setini yükle
dataset_path = "dataset/Training/"  
X_train, X_test, y_train, y_test = load_data(dataset_path)

# Modeli oluştur
input_shape = X_train.shape[1]  
model = create_ann_model(input_shape)

# Erken durdurma 
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğit ve geçmişi kaydet
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping], verbose=1)

# Modeli değerlendir
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Doğruluk: {test_acc:.4f}")

# Sınıflandırma raporu
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("Sınıflandırma Raporu:\n", classification_report(y_true, y_pred, target_names=["No Tumor", "Glioma", "Meningioma", "Pituitary"]))

# Kayıp ve doğruluk grafiklerini çizme
plt.figure(figsize=(12, 5))

# Kayıp Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.legend()

# Doğruluk Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend()

plt.show()
