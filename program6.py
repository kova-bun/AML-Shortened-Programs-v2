import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

x,y = load_iris(return_X_y=True)
a,b,c,d = train_test_split(x,y,test_size=0.2,random_state=0)

t = DecisionTreeClassifier().fit(a,c)
print(t)

plt.figure(figsize=(15,10))
plot_tree(t,filled=True)
plt.show()

param = {
    "criterion":["gini","entropy","log_loss"],
    "splitter":["best","random"],
    "max_depth":[1,2,3,4,5],
    "max_features":["sqrt","log2"]
}

g = GridSearchCV(t,param,cv=5,scoring="accuracy").fit(a,c)
p = g.predict(b)

print("Best Parameters:",g.best_params_)
print("Grid Score:",g.best_score_)
print("Confusion Matrix:\n",confusion_matrix(d,p))
print("Classification Report:\n",classification_report(d,p))