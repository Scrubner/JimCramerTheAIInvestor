import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

def readData():
    '''
    This function reads in the cleaned data sets and returns the raw data matricies X_ and Y_
    '''
    X_ = pd.read_csv(r'Data/cleaned_X.csv', index_col=0)
    Y_ = pd.read_csv(r'Data/cleaned_Y.csv', index_col=0)
    return X_, Y_

# -------------------------------
# Step 1: Read in all of the data
# -------------------------------
X_, Y_ = readData()
print(X_.head())
print(Y_.head())

keyCheckNullList = ["Short Term Debt", "Interest Expense, Net", "Income Tax (Expense) Benefit, Net",\
                    "Cash, Cash Equivalents & Short Term Investments", "Property, Plant, Equipment, Net",\
                    "Revenue", "Gross Profit"]

def fixNansInX():
    '''
    This function will go through the X_ Matrix and will remove any NaN entries by replacing them with 0
    '''
    for key in X_.keys():
        if key in keyCheckNullList:
            X_.loc[X_[key].isnull(), key] = 0

def addColsToX():
    '''
    This function calculates the values:
    EV - Enterprise value - correlates with the value a company has
    EBIT - Earnings before interest and taxes
    '''
    X_["EV"] = X_["Market Cap"] + X_["Long Term Debt"] + X_["Short Term Debt"] - X_["Cash, Cash Equivalents & Short Term Investments"]
    X_["EBIT"] = X_["Net Income"] - X_["Interest Expense, Net"] - X_["Income Tax (Expense) Benefit, Net"]

def getXRatios():
    X=pd.DataFrame()

    # Related to Earnings yield. Think of it as How much money each dollar in the company earns
    X["EV/EBIT"] = X_["EV"] / X_["EBIT"] 
    X["Op. In./(NWC+FA)"] = X_["Operating Income (Loss)"]/ (X_["Total Current Assets"] - \
                           X_["Total Current Liabilities"] + X_["Property, Plant & Equipment, Net"])
    X["P/E"] = X_["Market Cap"] / X_["Net Income"]      # Price to Earnings Ratio
    X["P/B"] = X_["Market Cap"] / X_["Total Equity"]    # Price to Book Value ratio
    X["P/S"] = X_["Market Cap"] / X_["Revenue"]         # Price to sales ratio
    X["Op. In./Interest Expense"] = X_["Operating Income (Loss)"] / X_["Interest Expense, Net"]
    X["Working Capital Ratio"] = X_["Total Current Assets"] / X_["Total Current Liabilities"]
    X["RoE"] = X_["Net Income"] / X_["Total Equity"]    # Return on Equity
    X["ROCE"] = X_["EBIT"] / (X_["Total Assets"] - X_["Total Current Liabilities"]) # Return on Equity Employed
    X["Debt/Equity"] = X_["Total Liabilities"] / X_["Total Equity"]
    X["Debt Ratio"] = X_["Total Assets"] / X_["Total Liabilities"]
    X["Cash Ratio"] = X_["Cash, Cash Equivalents & Short Term Investments"] / X_["Total Current Liabilities"]
    X["Asset Turnover"] = X_["Revenue"] / X_["Property, Plant & Equipment, Net"]
    X["Gross Profit Margin"] = X_["Gross Profit"] / X_["Revenue"]
    
    ### ALTMAN RATIOS ###
    X["(CA-CL)/TA"] = (X_["Total Current Assets"] - X_["Total Current Liabilities"]) / X_["Total Assets"]
    X["RE/TA"] = X_["Retained Earnings"] / X_["Total Assets"]
    X["EBIT/TA"] = X_["EBIT"] / X_["Total Assets"]
    X["Book Equity/TL"] = X_["Total Equity"] / X_["Total Liabilities"]

    return X

def maxMinRatio(m, text, max, min):
    m.loc[X[text]>max, text] = max
    m.loc[X[text]<min, text] = min

# -------------------------------------------------------------------------------------------
# Step 2: We need to clean up any of the outliers that are created by the feature engineering
# -------------------------------------------------------------------------------------------

# k = X.keys()[2] # 14 max
# X[k].hist(bins=100, figsize = (5,5))
# plt.title(k);

def fixXRatios():
    for key in X.keys():
        X[key][X[key].isnull()] = 0
        if (key == "RoE"):
            maxMinRatio(X, key, 5, -5)
        elif (key == "Op. In./(NWC+FA)"):
            maxMinRatio(X, key, 5, -5)
        elif (key == "EV/EBIT"):
            maxMinRatio(X, key, 500, -500)
        elif (key == "P/E"):
            maxMinRatio(X, key, 1000, -1000)
        elif (key == "P/B"):
            maxMinRatio(X, key, 100, -50)
        elif (key == "P/S"):
            maxMinRatio(X, key, 500, 0)
        elif (key == "Op. In./Interest Expense"):
            maxMinRatio(X, key, 800, -200)
        elif (key == "Working Capital Ratio"):
            maxMinRatio(X, key, 30, 0)
        elif (key == "ROCE"):
            maxMinRatio(X, key, 2, -2)
        elif (key == "Debt/Equity"):
            maxMinRatio(X, key, 50, -50)
        elif (key == "Debt Ratio"):
            maxMinRatio(X, key, 50, 0)
        elif (key == "Cash Ratio"):
            maxMinRatio(X, key, 30, 0)
        elif (key == "Gross Profit Margin"):
            maxMinRatio(X, key, 3, -3)
        ### ALTMAN RATIOS ###
        elif (key == "(CA-CL)/TA"):
            maxMinRatio(X, key, 2, -1.5)
        elif (key == "RE/TA"):
            maxMinRatio(X, key, 2, -20)
        elif (key == "EBIT/TA"):
            maxMinRatio(X, key, 1, -2)
        elif (key == "Book Equity/TL"):
            maxMinRatio(X, key, 20, -2)
        else:
            print(key)
            maxMinRatio(X, key, 2000, -2000)
    
    return X

fixNansInX()
addColsToX()
X = getXRatios()
X = fixXRatios()

X.to_csv(r"Data/Stock_Ratios.csv")

# -----------------------------
# Step 3: Fix the Y maxtrix now
# -----------------------------
def getYPerf(Y_):
    Y = pd.DataFrame()
    Y["Ticker"] = Y_["Ticker"]
    Y["Performance"] = (Y_["Open Price2"] - Y_["Open Price"])/Y_["Open Price"]
    Y[Y["Performance"].isnull()]=0

    return Y

Y = getYPerf(Y_)

Y.to_csv(r"Data/Stock_Performances.csv")

# -----------------------------------------------
# Rescale all of the X data to have better inputs
# -----------------------------------------------
transformer = PowerTransformer()
X_T = pd.DataFrame(transformer.fit_transform(X), columns=X.keys())

def plotFunc(n, myDataFrame):
    myKey = myDataFrame.keys()[n]
    plt.hist(myDataFrame[myKey], density = True, bins = 30)
    plt.grid()
    plt.xlabel(myKey)
    plt.ylabel('Probability')

plt.figure(figsize = (13,20))

print(X.keys())

plotsIwant = [0, 1, 2, 3, 4]

j=1
for i in plotsIwant:
    plt.subplot(len(plotsIwant), 2, 2*j-1)
    plotFunc(i,X)
    if j==1:
        plt.title('Before Transformation', fontsize = 17)
    plt.subplot(len(plotsIwant), 2, 2*j)
    plotFunc(i, X_T)
    if j == 1:
        plt.title('After Transformation', fontsize = 17)
    j += 1

plt.savefig('Transformed_Data_1.png', dpi = 300)
