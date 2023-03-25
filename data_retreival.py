import simfin as sf

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