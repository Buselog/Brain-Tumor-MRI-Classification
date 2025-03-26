import tensorflow as ts
Sequential = ts.keras.models.Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_ann_model(input_shape, num_classes=4):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),  # İlk katman (giriş boyutu)
        Dropout(0.3),  # Overfitting'i önlemek için
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Çıkış katmanı (4 sınıf için softmax)
    ])

    # Modeli derle (Adam optimizer + Categorical Crossentropy Loss)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model



