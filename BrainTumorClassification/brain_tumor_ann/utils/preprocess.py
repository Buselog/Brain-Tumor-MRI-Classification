import os
import numpy as np
import cv2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as ts
to_categorical = ts.keras.utils.to_categorical


# def load_data(dataset_path, img_size=128):
#     categories = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
#     data = []
#     labels = []
#
#     for i, category in enumerate(categories):
#         folder_path = os.path.join(dataset_path, category)
#         for image_name in os.listdir(folder_path):
#             image_path = os.path.join(folder_path, image_name)
#             image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü griye çevir
#             image = cv2.resize(image, (img_size, img_size))  # Boyutu küçült
#             data.append(image.flatten())  # 2D görüntüyü düzleştir (MLP için)
#             labels.append(i)  # 0: no_tumor, 1: glioma, 2: meningioma, 3: pituitary
#
#     data = np.array(data) / 255.0  # Normalizasyon (0-1 aralığına çekme)
#     labels = np.array(labels)
#
#     labels = to_categorical(labels, num_classes=4)  # One-hot encoding
#
#     return train_test_split(data, labels, test_size=0.2, random_state=42)

# Veri setinin konumu
DATASET_PATH = "dataset/Training/"
CATEGORIES = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]

# Veri artırma (Data Augmentation) işlemleri
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def load_data():
    data, labels = [], []

    for i, category in enumerate(CATEGORIES):
        folder_path = os.path.join(DATASET_PATH, category)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))  # 128x128 boyutuna getir

            # Orijinal görüntüyü ekle
            data.append(image)
            labels.append(i)

            # **Veri artırma işlemi burada uygulanıyor**
            image = np.expand_dims(image, axis=-1)  # (128,128) -> (128,128,1)
            image = np.expand_dims(image, axis=0)  # (128,128,1) -> (1,128,128,1)

            # 3 tane yeni görüntü üret
            aug_iter = data_generator.flow(image, batch_size=1)
            for _ in range(3):  # 3 yeni veri oluştur
                aug_image = next(aug_iter)[0].astype(np.uint8)
                aug_image = aug_image.squeeze()  # Tekrar (128,128) hale getir
                data.append(aug_image)
                labels.append(i)

    # Dizilere çevir ve normalize et (0-1 arasına getir)
    data = np.array(data) / 255.0
    labels = np.array(labels)

    # One-hot encoding
    labels = to_categorical(labels, num_classes=len(CATEGORIES))

    # Eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Modelin kabul etmesi için şekil değiştir
    X_train = X_train.reshape(-1, 128*128)  # ANN için düzleştirildi
    X_test = X_test.reshape(-1, 128*128)    # ANN için düzleştirildi

    return X_train, X_test, y_train, y_test
