# pages/dataset_overview.py
import streamlit as st
import pandas as pd
import sys
import os
from data.data_load import load_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def dataset_overview():
    st.header("Equity Quarterly Dataset Overview")

    # Load the dataset
    df = load_data()
    
    # 1. High-Level Dataset Summary
    st.subheader("Dataset Summary")
    st.write(f"**Total Number of Rows:** {df.shape[0]:,}")
    st.write(f"**Number of Financial Parameters:** {df.shape[1]:,}")
    st.write(f"**Number of Unique Tickers:** {df['ticker'].nunique():,}")
    st.write(f"**Data Date Range:** {df['calendardate'].min()} to {df['calendardate'].max()}")
    
    # 2. Feature Groups - Grouping the features for better understanding
    st.subheader("Feature Groups")
    
    # Grouping similar features together for a structured overview
    feature_groups = {
        "Financial Metrics": [
            'assets', 'assetsavg', 'assetsc', 'assetsnc', 'liabilities', 'liabilitiesc', 
            'liabilitiesnc', 'equity', 'equityavg', 'revenue', 'netinc', 'cashneq', 'cashnequsd', 
            'workingcapital', 'gp', 'intangibles', 'tangibles', 'debt', 'debtc', 'debtnc', 'debtusd'
        ],
        "Ratios & Margins": [
            'grossmargin', 'currentratio', 'de', 'pb', 'pe', 'pe1', 'bvps', 'ps', 'ps1', 
            'roe', 'roa', 'roic', 'ros', 'assetturnover', 'payoutratio', 'netmargin'
        ],
        "Performance Indicators": [
            'marketcap', 'eps', 'epsdil', 'epsusd', 'fcf', 'fcfps', 'price', 'divyield', 'ev', 
            'evebit', 'evebitda', 'sps', 'ppnenet', 'retearn'
        ],
        "Expenses": [
            'capex', 'opex', 'rnd', 'sgna', 'taxexp', 'depamor', 'sbcomp', 'payables', 
            'taxliabilities', 'intexp'
        ],
        "Cash Flow Components": [
            'ncf', 'ncfbus', 'ncfcommon', 'ncfdebt', 'ncfdiv', 'ncff', 'ncfi', 'ncfinv', 
            'ncfo', 'ncfx'
        ],
        "Debt & Investments": [
            'deferredrev', 'investments', 'investmentsc', 'investmentsnc', 'invcap', 
            'invcapavg', 'cor', 'deposits'
        ],
        "Earnings & Profits": [
            'ebit', 'ebitda', 'ebitdamargin', 'ebitdausd', 'ebitusd', 'ebt', 'opinc', 
            'netinccmn', 'netinccmnusd', 'netincdis', 'netincnci'
        ],
        "Other": [
            'sharesbas', 'shareswa', 'shareswadil', 'sharefactor', 'datemonth', 'dimension', 
            'delay', 'id', 'datekey', 'reportperiod'
        ]
    }
    for group_name, features in feature_groups.items():
        with st.expander(group_name):
            num_columns = 4  # Customize the number of columns for better aesthetics
            cols = st.columns(num_columns)
            for idx, feature in enumerate(features):
                with cols[idx % num_columns]:
                    st.write(f"• {feature}")

    # 5. Interactive Filtering Section (Optional)
    st.subheader("Explore Specific Ticker Data")

    # Use session state to preserve the selected ticker without full page reload
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = df['ticker'].iloc[0]

    selected_ticker = st.selectbox("Select a Ticker to Explore:", df['ticker'].unique(), 
                                   index=df['ticker'].tolist().index(st.session_state.selected_ticker))

    # Save the current selection in session state
    st.session_state.selected_ticker = selected_ticker

    # Display data for the selected ticker
    ticker_data = df[df['ticker'] == selected_ticker]
    st.subheader(f"Data for {selected_ticker}")
    st.write(ticker_data)

    st.header("Fixed Income and Commodity Dataset Overview")
    st.write("Currently under development!")

# Call this function in app.py
if __name__ == "__main__":
    dataset_overview()
