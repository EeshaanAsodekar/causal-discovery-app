# pages/causal_discovery.py
import streamlit as st
import pandas as pd
from data.data_load import get_filtered_data  # Assuming this function is already defined

# Cache the data to avoid reloading it multiple times
@st.cache_data
def load_data():
    '''Load and cache the raw dataset for performance improvement.'''
    df = pd.read_csv('data/raw_dataset.csv')
    df['calendardate'] = pd.to_datetime(df['calendardate'], errors='coerce')  # Ensure date is in datetime format
    return df.dropna(subset=['calendardate'])  # Drop rows where 'calendardate' is NaT

def causal_discovery_page():
    st.title("Perform Causal Analysis with varLiNGAM")

    # Load and cache the dataset
    df = load_data()

    # Part 1: Firm Selection (All Stocks Automatically Selected)
    st.header("All Firms Selected")

    # Use all stocks (No filtering)
    st.write("Using all stocks for the analysis.")
    st.write(df.head())  # Display some rows from the full dataset
    st.write(df['ticker'].nunique())


    # # Display DataFrame with all stocks
    # st.write("Displaying data for all stocks:")
    # st.write(df.head())

    # Part 2: Apply Market Cap Filter on the Last Row for Each Stock
    st.header("Market Cap Filter")

    # Get the latest market cap (as of the last available row) for each stock
    df_last_row = df.groupby('ticker').last().reset_index()
    
    mcap_threshold = st.slider("Market Cap Filter (in $M):", min_value=250, max_value=500_000, value=2_000)
    df_filtered_mcap = df_last_row[df_last_row['marketcap'] >= mcap_threshold * 1_000_000]

    # Filter original DataFrame based on selected stocks that meet the market cap requirement
    df_firm_filtered = df[df['ticker'].isin(df_filtered_mcap['ticker'])]

    # Display DataFrame after applying market cap filter
    st.write("Data after applying market cap filter:")
    st.write(df_firm_filtered.head())
    st.write(df_firm_filtered['ticker'].nunique())

    # Part 3: Variable Selection
    st.header("Variable Selection")

    # Search and Filter for Variables (Features)
    financial_parameters = df.columns.tolist()
    search_term = st.text_input("Search for a financial parameter:", "")
    filtered_params = [param for param in financial_parameters if search_term.lower() in param.lower()]
    
    selected_features = st.multiselect("Select Financial Parameters for Analysis:", filtered_params, default=["workingcapital", "revenueusd"])

    # Create a dictionary of DataFrames, one for each selected variable
    df_dict = {}
    for feature in selected_features:
        df_dict[feature] = get_filtered_data(df_firm_filtered, feature)  # Use get_filtered_data for each selected feature
    
    # Display each DataFrame for the selected features
    st.header("Data for Selected Variables")
    for feature, df_variable in df_dict.items():
        st.subheader(f"Data for {feature}")
        st.write(df_variable.head())
        st.write(df_variable.shape)

    # Part 4: Timeframe and Frequency Selection
    st.header("Timeframe Selection and Frequency Adjustment")

    # Ensure date range is set using datetime values, replacing NaT with a default min/max date
    min_date = df_firm_filtered['calendardate'].min().date() if not df_firm_filtered.empty else pd.to_datetime('2000-01-01').date()
    max_date = df_firm_filtered['calendardate'].max().date() if not df_firm_filtered.empty else pd.to_datetime('2020-01-01').date()

    date_range = st.date_input(
        "Select Date Range:",
        [min_date, max_date]
    )

    # Frequency Adjustment
    frequency = st.selectbox("Choose Data Frequency:", ["Quarterly", "Annually"], index=0)

    # Apply date filtering and frequency adjustment to all DataFrames in the dictionary
    for feature in selected_features:
        df_variable = df_dict[feature]

        # Filter data based on the selected date range
        df_variable = df_variable[
            (df_variable.index >= pd.to_datetime(date_range[0])) &
            (df_variable.index <= pd.to_datetime(date_range[1]))
        ]

        # Apply frequency resampling
        if frequency == "Quarterly":
            df_variable = df_variable.resample('Q').last()
        elif frequency == "Annually":
            df_variable = df_variable.resample('A').last()

        # Update the dictionary with the processed DataFrame
        df_dict[feature] = df_variable

    # Display the filtered and resampled DataFrames
    st.header("Filtered and Resampled DataFrames")
    for feature, df_variable in df_dict.items():
        st.subheader(f"Data for {feature} after timeframe and frequency adjustment")
        st.write(df_variable.head())
        st.write(df_variable.shape)

    return df_dict

# Call this function in your app.py
if __name__ == "__main__":
    causal_discovery_page()
