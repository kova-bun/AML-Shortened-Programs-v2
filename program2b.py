import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

d = pd.read_csv("iris.csv")
x = d[["SepalLengthCm","SepalWidthCm"]]
y = d["PetalLengthCm"]

a,b,c,e = train_test_split(x,y,test_size=0.2,random_state=0)
m = LinearRegression().fit(a,c)
p = m.predict(b)

print("Coefficients:", m.coef_)
print("Intercept:", m.intercept_)
print("MSE:", mean_squared_error(e,p))
print("R2 Score:", r2_score(e,p))
print("Prediction:", m.predict(pd.DataFrame([[5.8,3.0]],columns=x.columns))[0])

plt.scatter(e,p)
plt.plot([y.min(),y.max()],[y.min(),y.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()