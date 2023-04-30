import pandas as pd 
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def readData():
    X = pd.read_csv(r'Data/labels.csv')
    Y = pd.read_csv(r'Data/priceReport.csv')
    return X, Y

# -------------------------------
# Step 1: Read in all of the data
# -------------------------------
X, Y = readData()
print(X.head())
print(Y.head())

# --------------------------
# Step 2: visualize the data
# --------------------------

# Make a scatter matrix of the data:
attributes = ["Revenue", "Net Income"]
scatter_matrix(X[attributes]);
plt.show()
# Apply transformations to the data to make it easier to work with
