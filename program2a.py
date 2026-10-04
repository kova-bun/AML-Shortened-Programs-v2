import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

d = pd.read_csv("salary_data.csv")
x, y = d[["YearsExperience"]], d["Salary"]

m = LinearRegression().fit(x, y)
p = m.predict(x)

print("Coefficient:", m.coef_[0])
print("Intercept:", m.intercept_)
print("MSE:", mean_squared_error(y, p))
print("R2 Score:", r2_score(y, p))
print("Salary for 6 years:", round(m.predict(pd.DataFrame([[6]], columns=x.columns))[0], 2))

plt.scatter(x.iloc[:, 0], y)
plt.plot(x.iloc[:, 0], p, color="red")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()