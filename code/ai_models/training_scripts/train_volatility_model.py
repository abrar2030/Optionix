import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def train_model():
    data = pd.read_csv("../../resources/datasets/historical_volatility.csv")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[["open", "high", "low", "volume"]])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(64, input_shape=(60, 4), return_sequences=True),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.fit(scaled_data, data["volatility"], epochs=50, batch_size=32)
    model.save("../../ai_models/volatility_model.h5")


if __name__ == "__main__":
    train_model()
