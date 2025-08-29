import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Reshape
from scipy.fft import fft
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

with open("C:/Users/airob/OneDrive/Desktop/robo/hand_keypoints.json", "r") as f:
    text_to_keypoints = json.load(f)

word_list = list(text_to_keypoints.keys())
word_to_idx = {word: i for i, word in enumerate(word_list)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

X = np.array([word_to_idx[word] for word in text_to_keypoints.keys()])
y = np.array([np.mean(text_to_keypoints[word], axis=0) for word in text_to_keypoints.keys()])

y_fft = np.array([np.abs(fft(pts)) for pts in y])

def admm_optimize(y_fft):
    def loss_fn(params_flat):
        params = params_flat.reshape(y_fft.shape) 
        return np.sum((params - y_fft) ** 2)
    
    initial_params = np.random.randn(y_fft.size)  
    
    result = minimize(loss_fn, initial_params, method="L-BFGS-B")  
    return result.x.reshape(y_fft.shape)  

y_optimized = admm_optimize(y_fft)

X = X.reshape(-1, 1)  

model = Sequential([
    Embedding(input_dim=len(word_list), output_dim=16, input_length=1),  
    Reshape((1, 16)), 
    LSTM(32, return_sequences=False),
    Dense(42, activation="linear")
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y_optimized, epochs=50, batch_size=8, validation_split=0.2)

model.save("text_to_sign_model.h5")
print("Model saved successfully!")

y_pred = model.predict(X)

mae = mean_absolute_error(y_optimized, y_pred)
mse = mean_squared_error(y_optimized, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

def compute_accuracy(y_true, y_pred, tolerance=5):
    
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    
    accuracy = np.mean(distances <= tolerance) * 100  
    return accuracy

accuracy = compute_accuracy(y_optimized, y_pred)
print(f"Custom Accuracy (within tolerance): {accuracy:.2f}%")
