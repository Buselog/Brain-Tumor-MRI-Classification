import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as ts
to_categorical = ts.keras.utils.to_categorical
import random


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

# Gürültü ekleme fonksiyonu (ANN için uygun)
def add_noise(image):
    noise = np.random.randint(0, 30, image.shape, dtype='uint8')
    return np.clip(image + noise, 0, 255)


# Parlaklık değiştirme fonksiyonu (ANN için uygun)
def change_brightness(image):
    factor = random.uniform(0.7, 1.3)
    return np.clip(image * factor, 0, 255).astype(np.uint8)


# ANN için uygun veri yükleme fonksiyonu
def load_data(dataset_path, img_size=128, augment_factor=2):
    categories = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    data = []
    labels = []

    for i, category in enumerate(categories):
        folder_path = os.path.join(dataset_path, category)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_size, img_size))

            # Orijinal görüntüyü ekle (1D vektör olarak)
            data.append(image.flatten())
            labels.append(i)

            # Veri artırma işlemleri
            for _ in range(augment_factor):
                if random.random() > 0.5:  # %50 olasılıkla parlaklık değişimi
                    augmented_image = change_brightness(image)
                else:  # %50 olasılıkla gürültü ekleme
                    augmented_image = add_noise(image)

                data.append(augmented_image.flatten())  # ANN için düzleştirme
                labels.append(i)

    data = np.array(data) / 255.0  # Normalizasyon (0-1 aralığı)
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=4)  # One-hot encoding

    return train_test_split(data, labels, test_size=0.2, random_state=42)




