import tensorflow as tf
import matplotlib.pyplot as plt

i = tf.keras.Input(shape=(784,))
e = tf.keras.layers.Dense(32,activation="relu")(i)
o = tf.keras.layers.Dense(784,activation="sigmoid")(e)

m = tf.keras.Model(i,o)
m.compile(optimizer="adam",loss="binary_crossentropy")

(x,_),(y,_) = tf.keras.datasets.mnist.load_data()
x = x.reshape(-1,784).astype("float32")/255
y = y.reshape(-1,784).astype("float32")/255

m.fit(x,x,epochs=30,batch_size=256,shuffle=True,validation_data=(y,y))
p = m.predict(y)

plt.figure(figsize=(20,4))
for j in range(10):
    for row,img in enumerate([y[j],p[j]]):
        plt.subplot(2,10,row*10+j+1)
        plt.imshow(img.reshape(28,28),cmap="gray")
        plt.axis("off")
plt.show()