import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import pickle

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

def saveModel(my_pipeline)

# -----------------------
# Step 1: Import all data
# -----------------------

X = pd.read_csv(r'Data\Stock_Ratios.csv', index_col = 0)
Y = pd.read_csv(r'Data\Stock_Performances.csv', index_col = 0)
Y = Y["Performance"]

# -------------------
# Build the pipelines
# -------------------
pl_linear = Pipeline([('Power Transformer', PowerTransformer()),('Linear', LinearRegression())])
p1_ENet = Pipeline([('Power Transformer', PowerTransformer()),('E-Net', ElasticNet())])
pl_KNN = Pipeline([('Power Transformer', PowerTransformer()),('K Nearest Neighbors',\
                    KNeighborsRegressor(n_neighbors = 40))])
pl_SVM = Pipeline([('Power Transformer', PowerTransformer()),('Support Vector Machine',\
                    SVR())])
pl_decTree = Pipeline([('Decision Tree', DecisionTreeRegressor())])
pl_RF = Pipeline([('Random Forest', RandomForestRegressor(max_depth = 10))])
pl_ET = Pipeline([('Extra Trees', ExtraTreesRegressor(max_depth=10))])


                

