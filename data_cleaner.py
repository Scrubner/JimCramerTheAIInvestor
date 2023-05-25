import simfin as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix



def getStockData():
    # Set your API-key for downloading data.
    sf.set_api_key('5c12b7f2-512c-480a-ac5f-24447091db0c')

    # Set the local directory where data-files are stored.
    # The directory will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    # Download the data from the SimFin server and load into a Pandas DataFrame.
    ticker_df = sf.load_companies(market='us')
    income_df = sf.load_income(variant='annual')
    balance_df = sf.load_balance(variant='annual')
    cashflow_df = sf.load_cashflow(variant='annual')
    price_df = sf.load_shareprices(variant='daily')

    # Print the first rows of the data.
    # print(ticker_df.head())
    # print(income_df.head())
    # print(balance_df.head())
    # print(cashflow_df.head())
    # print(price_df.head())

    # Save the data into csv files for later use
    ticker_df.to_csv(r'Data\ticker.csv', index=True)
    income_df.to_csv(r'Data\income.csv', index=True)
    balance_df.to_csv(r'Data\balance.csv', index=True)
    cashflow_df.to_csv(r'Data\cashflow.csv', index=True)
    price_df.to_csv(r'Data\price.csv', index=True)
    return 0

def collectRawXData(read_Data = False):
    # Step 0: Get the data from simfin
    if read_Data:
        getStockData()
        print("Stock Data collected!")

    # Step 1: Import the data
    bal_df   = pd.read_csv(r'Data\balance.csv')     # Balance Data
    cash_df  = pd.read_csv(r'Data\cashflow.csv')    # cashflow Data
    inc_df   = pd.read_csv(r'Data\income.csv')      # income Data

    # Step 1b: Check to make sure we got the right data
    print('Balance array dimensions:', bal_df.shape)
    print('Cashflow array dimensions:', cash_df.shape)
    print('Income array dimensions:', inc_df.shape)

    # Step 2: Merge all of the data together
    inc_bal_comb = pd.merge(inc_df, bal_df, on = ['Ticker', 'SimFinId', 'Currency', 'Fiscal Year', 'Report Date', 'Publish Date'])   # Combine the income and Balance data

    X = pd.merge(inc_bal_comb, cash_df, on = ['Ticker', 'SimFinId', 'Currency', 'Fiscal Year', 'Report Date', 'Publish Date'])   # merge cashflow with the rest of the data
    X["Report Date"] = pd.to_datetime(X["Report Date"])
    X["Publish Date"] = pd.to_datetime(X["Publish Date"])
    print('Final merged array dimensions:', X.shape)
    
    return X

def collectRawYData(read_Data = False):
    # Step 0: Get the data from simfin, This step should be completed during "collectRayXData" function
    if read_Data:
        getStockData()

    # Import the data
    d         = pd.read_csv(r'Data\price.csv')     # Price Data
    d["Date"] = pd.to_datetime(d["Date"])
    print('Price array dimensions:', d.shape)
    return d
    

def get_price_data_at(ticker, day_f, num_days, d):
    window_days = 5
    # Step 1: Grab all data about the ticker of interest near the date of interest
    rows = d[(d["Date"].between(pd.to_datetime(day_f) + pd.Timedelta(days=num_days), pd.to_datetime(day_f)\
             + pd.Timedelta(days=num_days+window_days))) & (d["Ticker"]==ticker)]
    
    # Step 2: Grab the data closest to the date of interest
    if rows.empty:   # Edge case where no recent data exists
        return [ticker, np.float64("NaN"), np.datetime64('NaT'), np.float64("NaN")]
    else:   # return the most recent data available
        return [ticker, rows.iloc[0]["Open"], rows.iloc[0]["Date"], rows.iloc[0]["Open"]*rows.iloc[0]["Volume"]]

def get_price_report(x, d, modifier=365):
    y = []
    whichDateCol = 'Publish Date'

    print('Compiling Price Data...')
    for index, row in x.iterrows():
        ticker = row['Ticker']
        day_f = row[whichDateCol]

        price_data_1 = get_price_data_at(ticker, day_f, 0, d)
        price_data_2 = get_price_data_at(ticker, day_f, modifier, d)
        y.append(price_data_1 + price_data_2)

        if index == np.floor(len(x) / 4):
            print('25% complete...')
        elif index == np.floor(len(x) / 2):
            print('50% complete...')
        elif index == np.floor(3 * len(x) / 4):
            print('75% complete...')

    print('Price Data Compiled')

    print('Saving Price Data...')
    column_names = ['Ticker', 'Open Price', 'Date', 'Volume', 'Ticker2', 'Open Price2', 'Date2', 'Volume2']
    Y = pd.DataFrame(y, columns=column_names)
    print("Y matrix dimensions:", Y.shape)
    Y.to_csv('Data/priceReport.csv', index=True)
    print('Price Data Saved')

    return Y

# def getPriceDataAt(ticker, day_f, num_days, d):
#     window_days = 5
#     # Step 1: Grab all data about the ticker of interest near the date of interest
#     rows = d[(d["Date"].between(pd.to_datetime(day_f) + pd.Timedelta(days=num_days), pd.to_datetime(day_f)\
#              + pd.Timedelta(days=num_days+window_days))) & (d["Ticker"]==ticker)]
    
#     # Step 2: Grab the data closest to the date of interest
#     if rows.empty:   # Edge case where no recent data exists
#         return [ticker, np.float64("NaN"), np.datetime64('NaT'), np.float64("Nan")]
#     else:   # return the most recent data available
#         return [ticker, rows.iloc[0]["Open"], rows.iloc[0]["Date"], rows.iloc[0]["Open"]*rows.iloc[0]["Volume"]]
    
# def getPriceReport(x, d, modifier=365):
#     i = 0
#     y = [[None]*8 for i in range(len(x))]
#     whichDateCol = 'Publish Date'

#     print('Compiling Price Data...')
#     for index in range(len(x)):
#         y[i] = getPriceDataAt(x['Ticker'].iloc[index], x[whichDateCol].iloc[index], 0, d) + getPriceDataAt(x['Ticker'].iloc[index], x[whichDateCol].iloc[index], modifier, d)
#         if i == np.floor(len(x)/4):
#             print(r'25% complete...')
#         if i == np.floor(len(x)/2):
#             print(r'50% complete...')
#         if i == np.floor(3*len(x)/4):
#             print(r'75% complete...')
#         i = i+1
#     print('Price Data Compiled')

#     print('Saving Price Data...')
#     Y = pd.DataFrame(y, columns=['Ticker', 'Open Price', 'Date', 'Volume', 'Ticker2', 'Open Price2', 'Date2', 'Volume2'])
#     print("Y matrix dimensions:", Y.shape)
#     Y.to_csv(r'Data\priceReport.csv', index=True)
#     print('Price Data Saved')

    return y
#########################################################################################################
# MAIN SCRIPT
#########################################################################################################

# -------------
# Control Panel
# -------------


# ------------------------
# Step 1: Read in the data
# ------------------------
d = collectRawYData()

# X = collectRawXData(read_Data=True)
# X.to_csv(r'Data\labels.csv', index=True)
# Y = get_price_report(X, d)

X = pd.read_csv(r'Data\labels.csv', index_col = 0)
Y = pd.read_csv(r'Data\priceReport.csv', index_col = 0)

err_count = 0
for i in range(len(X.index)):
    if (Y.loc[i].at["Ticker"] != X.loc[i].at["Ticker"]):
        print( X.loc[i].at["Ticker"], Y.loc[i].at["Ticker"])
        err_count += 1
        print("There is a mismatch at", i)
        if err_count > 10:
            break

# ----------------------------------
# Step 3: Clean out the missing data
# ----------------------------------
# Case: No share price
mask = ~Y['Open Price'].isnull()
Y = Y[mask]
X = X[mask]

# Case: No listed number of shares
mask = ~X['Shares (Diluted)_x'].isnull()
Y = Y[mask]
X = X[mask]

# Case: Volume of shares if less than 10000
mask = ~((Y['Volume']<1e4) | (Y['Volume2']<1e4))
Y = Y[mask]
X = X[mask]

# Case: There is a date missing
# Case: No listed number of shares
mask = ~Y['Date2'].isnull()
Y = Y[mask]
X = X[mask]

Y = Y.reset_index(drop = True)
X = X.reset_index(drop = True)

X["Market Cap"] = Y["Open Price"] * X["Shares (Diluted)_x"]
print(X.shape)
print(Y.shape)

# Save the filtered data so we can avoid this step in the future
X.to_csv(r"Data\cleaned_X.csv")
Y.to_csv(r"Data\cleaned_Y.csv")

# ----------------------------------------
# Step 2: Some Data visuals
# ----------------------------------------
attributes = ["Revenue", "Net Income"]
scatter_matrix(X[attributes])
plt.show()
print("Plot Made")
