import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

# -------------------------------------------
# The subroutines that make this all possible
# -------------------------------------------

def plotDensityContour(model_name, x_plot, y_plot, ps):
    plt.scatter(x_plot, y_plot, s = 1)
    LinMod = LinearRegression().fit(y_plot.reshape(-1, 1), x_plot.reshape(-1, 1))
    xx = [[-5],[5]]
    yy = LinMod.predict(xx)
    plt.plot(yy, xx, 'g')

    # Formatting **WOOT**
    plt.grid()
    plt.axhline(y = 0, color = 'r', label = '_nolegend_')
    plt.axvline(x = 0, color = 'r', label = '_nolegend_')
    plt.xlabel('Predicted Return')
    plt.ylabel('Actual Return')
    plt.plot([-100, 100], [-100, 100], 'y--')
    plt.xlim([-ps, ps])
    plt.ylim([-ps, ps])
    plt.title('Predicted/Actual density plot for {}'.format(model_name))
    plt.legend(['Linear Fit Line', 'y=x Perfect Prediction Line','Prediction Points'])

def observePredictionAbility(my_pipeline, tests = 0.1, num_straps = 10):
    top10PredRtrns = np.array([])
    top10ActRtrns = np.array([])
    bot10PredRtrns = np.array([])
    bot10ActRtrns = np.array([])

    for i in range (0,num_straps):
        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = tests, random_state = i)

        # Fit the model
        my_pipeline.fit(X_train, Y_train)
        Y_pred = my_pipeline.predict(X_test)

        Y_purchase = pd.DataFrame(Y_pred)
        b1_top10 = (Y_purchase[0] > Y_purchase.nlargest(10, 0).tail(1)[0].values[0])
        Y_test_reindexed = Y_test.reset_index(drop = True)

        Y_purchase = pd.DataFrame(Y_pred)
        b1_bot10 = (Y_purchase[0] < Y_purchase.nsmallest(8, 0).tail(1)[0].values[0])

        top10PredRtrns = np.append(top10PredRtrns, round(np.mean(Y_purchase[b1_top10][0])*100,2))
        top10ActRtrns = np.append(top10ActRtrns, round(np.mean(Y_test_reindexed[b1_top10])*100,2))
        bot10PredRtrns = np.append(bot10PredRtrns, round(np.mean(Y_purchase[b1_bot10][0])*100,2))
        bot10ActRtrns = np.append(bot10ActRtrns, round(np.mean(Y_test_reindexed[b1_bot10])*100,2))

    print('-------------------------------------------------')
    ### IMPORT PERFORMANCE MEASURES HERE ###
    print('\033[4mMean Predicted Performance of Top 10 Return Portfolios:\033[0m', round(top10PredRtrns.mean(),2))
    print('\033[4mMean Actual Performance of Top 10 Return Portfolios:\033[0m', round(top10ActRtrns.mean(),2))
    print('Mean Predicted Performance of Top 10 Return Portfolios:', round(bot10PredRtrns.mean(),2))
    print('Mean Actual Performance of Top 10 Return Portfolios:', round(bot10ActRtrns.mean(),2))

# -----------------------
# Step 1: Import all data
# -----------------------

X = pd.read_csv(r'Data\Stock_Ratios.csv', index_col = 0)
Y = pd.read_csv(r'Data\Stock_Performances.csv', index_col = 0)
Y = Y["Performance"]

# -----------------------------------------------------
# Step 2: Determine the baseline prediction of the data
# -----------------------------------------------------
baseline = Y.mean()
print("The base line prediction is:", baseline)

# -------------------------------------
# Step 3: Build the Test and Train sets
# -------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state = 41)

# ------------------------------------------------------------------------------------------------
# Step 4: Fit the models
# There are some sub steps to this work flow I wanted to outline here for a clearer understanding
# a - Train the model on the training set
# b - Use the model to predict the test set
# c - Evaluate the performance of the model on the test set
# ------------------------------------------------------------------------------------------------

# Part a
pl_linear = Pipeline([('Power Transformer', PowerTransformer()),('linear', LinearRegression())])
pl_linear.fit(X_train, Y_train)

# Part b
Y_train_pred = pl_linear.predict(X_train) 
Y_test_pred = pl_linear.predict(X_test)

# Part c
print("Linear Regression")
print("-Train MSE: ", mean_squared_error(Y_train, Y_train_pred))
print("-Test MSE:  ", mean_squared_error(Y_test, Y_test_pred))

# ---------------------------------
# Step 5: Cross validate the models
# ---------------------------------

# vals = [2, 5, 10, 20, 100, 200]

# print('Linear Regre')
# for i in vals:
#     scores = cross_validate(pl_linear, X, Y, scoring = 'neg_mean_squared_error', cv = i, return_train_score = True)
#     print('K = ', i)
#     print('AVERAGE TEST SCORE:', round(np.sqrt(-scores['test_score']).mean(),4),\
#            'STD. DEV.', round(np.sqrt(-scores['test_score']).std(),4))
#     print('AVERAGE TRAIN SCORE:', round(np.sqrt(-scores['train_score']).mean(),4),\
#            'STD. DEV.', round(np.sqrt(-scores['train_score']).std(),4))
#     print('-------------------------------------------------------')

# ---------------------------------------
# Step 6: Make the final diagnostic plots
# ---------------------------------------
plotDensityContour('pl_linear', Y_test_pred, Y_test.to_numpy(), 2)
plt.savefig('linear_reg_contour_plot.png', dpi = 300)
#plt.show()

# ---------------------------------------------------------
# Step 7: See how the trading strategy would have performed
# ---------------------------------------------------------
Y_purchase = pd.DataFrame(Y_test_pred)
b1_top10 = (Y_purchase[0] > Y_purchase.nlargest(10, 0).tail(1)[0].values[0])
Y_test_reindexed = Y_test.reset_index(drop = True)

print('Predicted Returns: ', Y_purchase[b1_top10][0].values)
print('\nActual Returns: ', Y_test_reindexed[b1_top10].values)

print('Top 10 Predicted Returns: ', round(np.mean(Y_purchase[b1_top10][0])*100,2), '%')
print('Actual Top 10 Returns: ', round(np.mean(Y_test_reindexed[b1_top10])*100,2), '%', '\n')

b1_bot10 = (Y_purchase[0] < Y_purchase.nsmallest(8, 0).tail(1)[0].values[0])

print('Bottom 10 Predicted Returns: ', round(np.mean(Y_purchase[b1_bot10][0])*100,2), '%')
print('Actual Bottom 10 Returns: ', round(np.mean(Y_test_reindexed[b1_bot10])*100,2), '%', '\n')

observePredictionAbility(pl_linear, tests = 0.2, num_straps= 100)












































