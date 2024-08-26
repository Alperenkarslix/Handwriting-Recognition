import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import requests

data_gen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = data_gen.flow_from_directory(
    '/kaggle/input/handwrite-turkish-dataset/HandwrittingDataSet/data/train_data',  # Veri setinin yolu
    target_size=(64, 64),  # Tüm resimler bu boyuta getirilecek
    batch_size=32,  # Her bir iterasyonda alınacak örnek sayısı
    subset='training',  # Eğitim verisi
    class_mode='categorical',  # Sınıflandırma türü
    color_mode='grayscale'  # Renk modu
)

val_gen = data_gen.flow_from_directory(
    '/kaggle/input/handwrite-turkish-dataset/HandwrittingDataSet/data/validation_data',  # Veri setinin yolu
    target_size=(64, 64),  # Tüm resimler bu boyuta getirilecek
    batch_size=32,  # Her bir iterasyonda alınacak örnek sayısı
    subset='validation',  # Doğrulama verisi
    class_mode='categorical',  # Sınıflandırma türü
    color_mode='grayscale'  # Renk modu
)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(26, activation='softmax')
])

optimizer = keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

# Görüntüyü yükleme ve gri tonlamaya çevirme
img = cv2.imread('/kaggle/input/predict3/predict3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Görüntüyü ikili biçime (binary image) çevirme
# Eşik değeri belirleme ve uygulama (thresholding)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Karakterleri segmente etme
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Konturları soldan sağa sıralama
bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
(contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                         key=lambda b: b[1][0], reverse=False))

# Karakter tahminleri için döngü
predictions = []
for contour in contours:
    # Kontur etrafında sınırlayıcı dikdörtgen çizme ve segmenti alma
    x, y, w, h = cv2.boundingRect(contour)
    # Dikdörtgen alanı görüntüden kesme
    letter_image = binary[y:y + h, x:x + w]
    # Tahmin için görüntüyü yeniden boyutlandırma ve modelin beklediği şekle getirme
    letter_image = cv2.resize(letter_image, (64, 64))
    letter_image = img_to_array(letter_image)
    letter_image = np.expand_dims(letter_image, axis=0)
    letter_image = np.reshape(letter_image, (1, 64, 64))

    # Tahmin yapma
    prediction = model.predict(letter_image)
    predicted_class = np.argmax(prediction, axis=1)
    predictions.append(predicted_class[0])  # predicted_class'ın ilk elemanını ekle

# Tahmin edilen karakterleri ekrana yazdırma
predicted_text = [chr(prediction + 96) for prediction in predictions]  # Her tahmini string'e çevir
predicted_text = ' '.join(predicted_text)  # Aralarında boşluk bırakarak birleştir
print(f'Tahmin edilen harfler: {predicted_text}')