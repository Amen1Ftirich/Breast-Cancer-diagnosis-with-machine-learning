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
    test_size = 0.2 # emaning 20% of the data
    random_state = 67 #This jawn hdre freezes the state so that everytime I don't get a new set to train and test, it could be any number
    stratisfy = y # Keeps approx the same values of the different observations in the test adn the train kits , super interesting stuff
)
