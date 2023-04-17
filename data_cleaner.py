import simfin as sf
import numpy as np
import pandas as pd


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
    
def getPriceDataAt(ticker, day_f, num_days, d):
    window_days = 5
    # Step 1: Grab all data about the ticker of interest near the date of interest
    rows = d[(d["Date"].between(pd.to_datetime(day_f) + pd.Timedelta(days=num_days), pd.to_datetime(day_f)\
             + pd.Timedelta(days=num_days+window_days))) & (d["Ticker"]==ticker)]
    
    # Step 2: Grab the data closest to the date of interest
    if rows.empty:   # Edge case where no recent data exists
        return [ticker, np.float64("NaN"), np.datetime64('NaT'), np.float64("Nan")]
    else:   # return the most recent data available
        return [ticker, rows.iloc[0]["Open"], rows.iloc[0]["Date"], rows.iloc[0]["Open"]*rows.iloc[0]["Volume"]]
    
def getPriceReport(x, d, modifier=365):
    i = 0
    y = [[None]*8 for i in range(len(x))]
    whichDateCol = 'Publish Date'

    print('Compiling Price Data...')
    for index in range(len(x)):
        y[i] = getPriceDataAt(x['Ticker'].iloc[index], x[whichDateCol].iloc[index], 0, d) + getPriceDataAt(x['Ticker'].iloc[index], x[whichDateCol].iloc[index], modifier, d)
        if i == np.floor(len(x)/4):
            print(r'25% complete...')
        if i == np.floor(len(x)/2):
            print(r'50% complete...')
        if i == np.floor(3*len(x)/4):
            print(r'75% complete...')
        i = i+1
    print('Price Data Compiled')

    print('Saving Price Data...')
    Y = pd.DataFrame(y, columns=['Ticker', 'Open Price', 'Date', 'Volume', 'Ticker2', 'Open Price2', 'Date2', 'Volume2'])
    Y.to_csv(r'Data\priceReport.csv', index=True)
    print('Price Data Saved')

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
x = collectRawXData()
d = collectRawYData()
Y = getPriceReport(x, d)

# ---------------------------------------------
# Step 2: Clean up the data for easy processing
# ---------------------------------------------

# ------------------ The following are some test cases: confirm data when able -------------------------
print(getPriceDataAt('GOOG','2021-05-12', 0, d))    # currently reporting correct values in csv file - check to make sure csv is correct
print(getPriceDataAt('GOOG','2021-05-12', 30, d))   # Doing the same as the earlier function

