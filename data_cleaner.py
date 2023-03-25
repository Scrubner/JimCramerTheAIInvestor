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
    y         = pd.read_csv(r'Data\price.csv')     # Price Data
    y["Date"] = pd.to_datetime(price_df["Date"])
    print('Price array dimensions:', y.shape)
    return y
    

X = collectRawXData()
y = collectRawYData()
