import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def svd_compress_images(images, k=20):
    compressed_images = []
    for img in images:
        img = img.squeeze() 
        U, S, Vt = np.linalg.svd(img, full_matrices=False)
        S_k = np.diag(S[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]
        img_compressed = np.dot(U_k, np.dot(S_k, Vt_k))
        compressed_images.append(img_compressed)
    compressed_images = np.array(compressed_images).reshape(-1, 28, 28, 1)
    return compressed_images

df = pd.read_csv("C:/Users/airob/OneDrive/Desktop/robo/sign_mnist_train_small.csv")

X = df.drop(columns=['label']).values
y = df['label'].values

X = X / 255.0

X = X.reshape(-1, 28, 28, 1)
print("Shape of X after reshaping:", X.shape)

X = svd_compress_images(X, k=20)
print("Shape after SVD compression:", X.shape)

y[y >= 24] = 23
print("Unique labels after correction:", np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, num_classes=24)
y_test = to_categorical(y_test, num_classes=24)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save("sign_language_cnn_model.h5")
print("Model saved successfully!")
