# data/data_load.py
import pandas as pd
import streamlit as st

def load_raw_data() -> pd.DataFrame:
    '''Download and locally store the accounting dataset in CSV format.'''
    df = pd.read_parquet("https://storage.googleapis.com/sovai-public/accounting/fundamental_quarterly.parq")
    df.to_csv("data/raw_dataset.csv")  # Save the dataset to a local CSV file

def df_info(df):
    '''Display information about the DataFrame, including shape, columns, and a preview of the data.'''
    print("---------------------------------------------")
    print(df.shape)  # Print the dimensions of the DataFrame
    print(df.columns)  # Print the column names
    print(df.head())  # Print the first 5 rows for a quick preview
    print(df.tail())  # Print the last 5 rows for a quick preview
    print("---------------------------------------------")

def remove_ticks(df_accounting, feature):
    '''
    Remove tickers that have a zero mean for the specified feature.
    This helps clean the dataset by filtering out tickers with missing or irrelevant data.
    
    Parameters:
        df_accounting (pd.DataFrame): The input dataset.
        feature (str): The column name representing the feature to check for zero mean values.

    Returns:
        pd.DataFrame: A filtered DataFrame with tickers having non-zero values for the specified feature.
    '''
    # Ensure 'ticker' is a column, not part of the index
    if 'ticker' in df_accounting.index.names:
        df_accounting = df_accounting.reset_index()

    # Group by 'ticker' and calculate the mean for the specified feature
    ticker_revenue_mean = df_accounting.groupby('ticker')[feature].mean()

    # Identify tickers with a mean value of zero for the given feature
    tickers_to_remove = ticker_revenue_mean[ticker_revenue_mean == 0].index
    print(f"Removing {len(tickers_to_remove)} tickers with zero mean {feature}")

    # Filter out tickers with zero mean from the DataFrame
    df_filtered = df_accounting[~df_accounting['ticker'].isin(tickers_to_remove)]
    print(f"Remaining tickers after removing zero {feature}: {df_filtered['ticker'].nunique()}")

    return df_filtered

def singular_feature(df_accounting, feature):
    '''
    Process the dataset for a specific feature by removing irrelevant tickers,
    grouping by date and ticker, and pivoting the data to prepare it for analysis.

    Parameters:
        df_accounting (pd.DataFrame): The input dataset.
        feature (str): The column name representing the feature to analyze.

    Returns:
        pd.DataFrame: A pivoted DataFrame with the feature values for each ticker over time.
    '''
    # Remove tickers with zero mean feature values
    df_filtered = remove_ticks(df_accounting, feature)
    print(f"Tickers remaining after filtering: {df_filtered['ticker'].nunique()}")

    # Select only relevant columns for further processing
    df_total_revenue = df_filtered.reset_index()[['calendardate', 'ticker', feature]]
    
    # Handle duplicate entries by taking the mean of duplicate rows
    df_total_revenue = df_total_revenue.groupby(['calendardate', 'ticker'])[feature].mean().reset_index()

    print(f"Tickers after grouping by 'calendardate' and 'ticker': {df_total_revenue['ticker'].nunique()}")

    # Pivot the DataFrame so that 'ticker' becomes columns and 'calendardate' becomes the index
    df_pivoted = df_total_revenue.pivot(index='calendardate', columns='ticker', values=feature)

    # Drop columns (tickers) with all NaN values
    print(f"Tickers before dropping columns with all NaNs: {df_pivoted.shape[1]}")
    df_pivoted = df_pivoted.dropna(axis=1, how="all")
    print(f"Tickers after dropping columns with all NaNs: {df_pivoted.shape[1]}")

    return df_pivoted

def variable_improvement(df_accounting, variable):
    '''
    Enhance the dataset by cleaning, filling missing data, and preparing it for causal analysis.

    Parameters:
        df_accounting (pd.DataFrame): The input dataset.
        variable (str): The feature to be improved and processed.

    Returns:
        pd.DataFrame: A cleaned and processed DataFrame with the specified variable.
    '''
    df_pivoted = singular_feature(df_accounting, variable)

    # Ensure the index (calendardate) is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_pivoted.index):
        df_pivoted.index = pd.to_datetime(df_pivoted.index)

    # Drop columns where the most recent date (last row) contains NaN values
    print(f"Tickers before dropping columns with NaN in the last row: {df_pivoted.shape[1]}")
    df_pivoted = df_pivoted.dropna(axis=1, subset=[df_pivoted.index[-1]])
    print(f"Tickers after dropping columns with NaN in the last row: {df_pivoted.shape[1]}")

    # Fill any remaining missing values with the mean value of each row
    row_means = df_pivoted.mean(axis=1)
    df_pivoted = df_pivoted.dropna(axis=0, how='all')  # Drop rows where all values are NaN
    df_pivoted = df_pivoted.T.fillna(row_means).T  # Fill missing values with row means

    return df_pivoted

def get_filtered_data(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    '''
    Wrapper function to apply the variable improvement process on the dataset.

    Parameters:
        df (pd.DataFrame): The input dataset.
        variable (str): The feature to be processed.

    Returns:
        pd.DataFrame: The processed and filtered dataset.
    '''
    return variable_improvement(df, variable)

# Cache the data to avoid reloading it multiple times
@st.cache_data
def load_data():
    '''Load and cache the raw dataset for performance improvement.'''
    df = pd.read_csv('data/raw_dataset.csv')
    df['calendardate'] = pd.to_datetime(df['calendardate'], errors='coerce')  # Ensure date is in datetime format
    return df.dropna(subset=['calendardate'])  # Drop rows where 'calendardate' is NaT

if __name__ == "__main__":
    # Load the raw dataset
    load_raw_data()

    # Read the locally saved CSV file into a DataFrame
    df = pd.read_csv('data/raw_dataset.csv')
    
    # Display information about the raw dataset
    df_info(df)

    # Process the dataset for the 'workingcapital' feature
    df_processed = get_filtered_data(df, "workingcapital")

    # Display information about the processed dataset
    df_info(df_processed)
