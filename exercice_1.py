from ucimlrepo import fetch_ucirepo
import pandas as pd
# Download the Breast Cancer Wisconsin Diagnostic dataset
cancer = fetch_ucirepo(id=17)

# X contains the input features
X = cancer.data.features

# y contains the answers/classes
y = cancer.data.targets

# print(X.head()) #just to see
# print(y.shape)
# print(y.head())
# print(y.value_counts()) to see the results

#I learned that SciKit prefers raveled data so
y = y.values.ravel()
#to check 
# print(y.shape) gold

#Ok so you can split the data using scikit function train_test_split, they make it so easy so you have to split them with pandas
from sklearn.model_selection import train_test_split

X_train, X_Test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2, # emaning 20% of the data
    random_state = 67, #This jawn hdre freezes the state so that everytime I don't get a new set to train and test, it could be any number
    stratify = y # Keeps approx the same values of the different observations in the test adn the train kits , super interesting stuff
)
#Now building the decision tree itself

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    max_depth= None,
    random_state= 67
)

tree_model.fit(X_train, y_train)

#Now we give the model states it has not seen before
tree_predictions = tree_model.predict(X_Test)

#calcualting classification accuracy
from sklearn.metrics import accuracy_score
tree_accuracy = accuracy_score(y_test, tree_predictions)
print("Decision Tree accuracy:", tree_accuracy)

#Now plotting it
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(17,10)) #Random parameters
#I got these from their website
plot_tree(tree_model,
        feature_names= X.columns,
        class_names= tree_model.classes_,
        filled= True,
        rounded = True,
        fontsize=8, #I tried other ones this is is good
          )
plt.title(" Dcision Tree - Wisconsin Breast Cancer Dataset")
plt.show()