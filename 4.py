from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

x,y = load_iris(return_X_y=True)
x,y = x[y<2],y[y<2]

a,b,c,d = train_test_split(x,y,test_size=0.2,random_state=0)
m = LogisticRegression().fit(a,c)
p = m.predict(b)

print("Accuracy:", accuracy_score(d,p))
print("Confusion Matrix:\n", confusion_matrix(d,p))
print("Classification Report:\n", classification_report(d,p))
