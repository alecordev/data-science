import numpy as np
import keras
from keras import layers
from keras import ops


def simple1():
    model = keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )

    print(model.layers)
    x = ops.ones((3, 3))
    y = model(x)

    print(y)
    print(model.summary())


def simple2():
    initial_model = keras.Sequential(
        [
            keras.Input(shape=(250, 250, 3)),
            layers.Conv2D(32, 5, strides=2, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(32, 3, activation="relu"),
        ]
    )
    feature_extractor = keras.Model(
        inputs=initial_model.inputs,
        outputs=[layer.output for layer in initial_model.layers],
    )

    # Call feature extractor on test input.
    x = ops.ones((1, 250, 250, 3))
    features = feature_extractor(x)
    print(features)


if __name__ == "__main__":
    # simple1()
    simple2()
