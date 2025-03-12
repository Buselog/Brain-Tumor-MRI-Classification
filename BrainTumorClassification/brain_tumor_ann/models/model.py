import tensorflow as ts
Sequential = ts.keras.models.Sequential
from tensorflow.keras.layers import Dense

def create_model(input_shape):
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_shape,)),  # Giriş katmanı
        Dense(64, activation="relu"),  # Gizli katman
        Dense(4, activation="softmax")  # Çıkış katmanı (4 sınıf)
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

