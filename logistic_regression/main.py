# import dataset
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:, -1].values

# splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_test = sc.fit_transform(X_test)
X_train = sc.transform(X_train)
# import regression model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score

cm = confusion_matrix(y_test, classifier.predict(X_test))
print(cm)
print(accuracy_score(y_test, classifier.predict(X_test)))
