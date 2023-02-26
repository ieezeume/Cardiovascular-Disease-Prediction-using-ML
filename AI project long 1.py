import pandas as pd
df = pd.read_csv ("C:\\Users\\Azmat\\OneDrive\\Desktop\\preprocessed_dataset1.csv")
df.head(5)
X = df [['age','gender','height','weight','systolic','diastolic','cholesterol','glucose','smoke','alcohol','active','pulse_pressure']]
y = df ['cardiovascular_disease'].ravel()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=109)
from sklearn.svm import SVC
clf = SVC(kernel='rbf',C=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))





