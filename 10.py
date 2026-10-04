import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Input

x = np.tile(np.arange(9),(100,1)).reshape(100,9,1)
y = np.full(100,9)

m = Sequential([Input(shape=(9,1)),LSTM(50,activation="relu"),Dense(1)])
m.compile(optimizer="adam",loss="mse")
m.fit(x,y,epochs=200,verbose=0)

t = np.arange(7,16).reshape(1,9,1)
print("Predicted Value:",m.predict(t,verbose=0)[0][0])
