from ucimlrepo import fetch_ucirepo

# Download the Breast Cancer Wisconsin Diagnostic dataset
cancer = fetch_ucirepo(id=17)

# X contains the input features
X = cancer.data.features

# y contains the answers/classes
y = cancer.data.targets

