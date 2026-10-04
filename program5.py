from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

x,y = load_iris(return_X_y=True)
a,b,c,d = train_test_split(x,y,test_size=0.2,random_state=1)

s = StandardScaler()
a,b = s.fit_transform(a),s.transform(b)

for name,m in [("KNN",KNeighborsClassifier(3)),("GAUSSIAN NB",GaussianNB())]:
    p = m.fit(a,c).predict(b)
    print(name,"CLASSIFICATION")
    print(confusion_matrix(d,p))
    print(classification_report(d,p))