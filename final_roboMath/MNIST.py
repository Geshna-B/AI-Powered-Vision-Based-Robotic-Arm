import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/airob/OneDrive/Desktop/final/final/MNIST/sign_mnist_train.csv")

df_sample = df.sample(frac=0.5, random_state=42)

df_sample.to_csv("sign_mnist_train_small.csv", index=False)
