from ucimlrepo import fetch_ucirepo
import pandas as pd

#loading the Wisconsin Breast Cancer dataset
cancer = fetch_ucirepo(id=17)

#features
X = cancer.data.features

#results/classes
y = cancer.data.targets

#just checking the data
# print(X.head())
# print(y.head())
# print(X.shape)
# print(y.value_counts())

#scikit likes the target in 1D so raveling it
y = y.values.ravel()

#check
# print(y.shape)


#splitting the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_Test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% for testing
    random_state=67, #freezes the split so it stays the same
    stratify=y #keeps about the same B and M ratio in both sets
)


#ANNs are sensitive to scaling so gotta scale the features first
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#fit on train, then transform it
X_train_scaled = scaler.fit_transform(X_train)

#only transform test data, don't fit again
X_Test_scaled = scaler.transform(X_Test)


#building the ANN
from sklearn.neural_network import MLPClassifier

ann_model = MLPClassifier(
    hidden_layer_sizes=(16, 8), #2 hidden layers, 16 neurons then 8
    activation="relu", #activation function
    solver="adam", #updates the weights
    max_iter=1000, #max training iterations
    random_state=67
)

#training it
ann_model.fit(X_train_scaled, y_train)

#testing on data it hasn't seen
ann_predictions = ann_model.predict(X_Test_scaled)


#accuracy
from sklearn.metrics import accuracy_score

ann_accuracy = accuracy_score(y_test, ann_predictions)

print("ANN Accuracy:", ann_accuracy)


#confusion matrix
from sklearn.metrics import confusion_matrix

ann_confusion = confusion_matrix(y_test, ann_predictions)

print("\nANN Confusion Matrix:")
print(ann_confusion)


#classification report
from sklearn.metrics import classification_report

print("\nANN Classification Report:")
print(classification_report(y_test, ann_predictions))