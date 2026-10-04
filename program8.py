import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

(x,y),(a,b) = mnist.load_data()
x,a = x[...,None]/255.0, a[...,None]/255.0
y,b = to_categorical(y,10),to_categorical(b,10)

m = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32,3,activation="relu"),
    MaxPooling2D(2),
    Dropout(0.25),
    Conv2D(64,3,activation="relu"),
    MaxPooling2D(2),
    Dropout(0.25),
    Flatten(),
    Dense(128,activation="relu"),
    Dense(10,activation="softmax")
])

m.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
h = m.fit(x,y,epochs=10,batch_size=32,validation_data=(a,b))

print("Loss, Accuracy:",m.evaluate(a,b))

plt.plot(h.history["accuracy"],label="Train")
plt.plot(h.history["val_accuracy"],label="Test")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()