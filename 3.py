import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(0)
x = np.random.rand(100,2)
y = (x[:,0]+x[:,1]>1).astype(int)

a,b,c,d = train_test_split(x,y,test_size=0.2,random_state=0)
m = LogisticRegression().fit(a,c)
p = m.predict(b)

print("Predicted:", p)
print("Actual:", d)
print("Accuracy:", accuracy_score(d,p))
print("Confusion Matrix:\n", confusion_matrix(d,p))
