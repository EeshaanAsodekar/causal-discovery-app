# pages/causal_discovery.py
import streamlit as st
import pandas as pd
from data.data_load import get_filtered_data  # Assuming this function is already defined
import lingam
import matplotlib.pyplot as plt
import networkx as nx

from data.data_load import load_data

# # Cache the data to avoid reloading it multiple times
# @st.cache_data
# def load_data():
#     '''Load and cache the raw dataset for performance improvement.'''
#     df = pd.read_csv('data/raw_dataset.csv')
#     df['calendardate'] = pd.to_datetime(df['calendardate'], errors='coerce')  # Ensure date is in datetime format
#     return df.dropna(subset=['calendardate'])  # Drop rows where 'calendardate' is NaT

def apply_varlingam(df_dict, lags=1, top_n=10, bootstrap=False, n_sampling=5):
    """
    Apply the VARLiNGAM model to the dataframes in df_dict.
    
    Parameters:
        df_dict (dict): A dictionary where keys are financial parameters and values are dataframes.
        lags (int): The number of lags for the VARLiNGAM model.
        top_n (int): Number of top nodes to visualize based on connection strength.
        bootstrap (bool): Whether to use bootstrap sampling.
        n_sampling (int): Number of bootstrap samples if using bootstrap.
        
    Returns:
        varlingam_results (dict): A dictionary storing the results for each financial parameter.
    """
    varlingam_results = {}  # To store results for each parameter
    
    for feature, df_variable in df_dict.items():
        if df_variable.shape[1] < 2:
            print(f"Skipping {feature} because it has less than 2 valid stocks for analysis.")
            continue
        
        print(f"Applying VARLiNGAM for {feature} with {lags} lags...")
        
        # Initialize and fit the VARLiNGAM model
        model = lingam.VARLiNGAM(lags=lags)
        
        if bootstrap:
            result = model.bootstrap(df_variable, n_sampling=n_sampling)
            adjacency_matrices = result.adjacency_matrices_
            causal_order = result.causal_order_
        else:
            model.fit(df_variable)
            adjacency_matrices = model.adjacency_matrices_
            causal_order = model.causal_order_

        # Get the original labels for the stocks (columns in df_variable)
        org_labels = df_variable.columns.to_list()
        order_labels = df_variable.iloc[:, causal_order].columns.to_list()

        # Store results in varlingam_results
        varlingam_results[feature] = {
            "causal_order": causal_order,
            "ordered_labels": order_labels,
            "original_labels": org_labels,
            "adjacency_matrices": adjacency_matrices,
            "model": model
        }

        # Visualize the causal graph for the current financial parameter
        print(f"Visualizing the causal graph for {feature}")
        visualize_causal_graph(adjacency_matrices, org_labels, feature,lags, top_n=top_n)
        
    return varlingam_results

def visualize_causal_graph(adjacency_matrices, labels, fin_param, lags, top_n=10):
    """
    Visualize the causal relationships using a directed graph.

    Parameters:
        adjacency_matrices (list): A list of adjacency matrices from the VARLiNGAM model.
        labels (list): A list of stock ticker labels.
        top_n (int): Number of top nodes to visualize based on connection strength.
    """
    G = nx.DiGraph()
    num_vars = len(labels)
    
    # Build the graph
    for i in range(num_vars):
        for j in range(num_vars):
            if i != j and any(adj[i, j] != 0 for adj in adjacency_matrices):
                weight = sum(adj[i, j] for adj in adjacency_matrices)
                G.add_edge(labels[i], labels[j], weight=weight)

    # Calculate total connection strength for each node
    node_strength = {node: sum(G.edges[node, n]['weight'] for n in G[node]) + 
                            sum(G.edges[n, node]['weight'] for n in G.pred[node])
                     for node in G.nodes()}
    
    # Sort nodes by strength and select top_n nodes
    top_nodes = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_labels = [node for node, _ in top_nodes]
    
    # Create a subgraph for the top_n nodes
    G_top = G.subgraph(top_labels)
    
    pos = nx.spring_layout(G_top, seed=42)  # For consistent layout
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Adjust node size and edge width
    nx.draw_networkx_nodes(G_top, pos, node_size=1500, node_color='skyblue', alpha=0.8, ax=ax)
    
    # Draw edges with bold arrows
    edges = G_top.edges(data=True)
    nx.draw_networkx_edges(G_top, pos, edgelist=edges, width=2, edge_color='black', alpha=0.7, arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax)
    
    # Draw edge labels with the weight
    edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in edges}
    nx.draw_networkx_edge_labels(G_top, pos, edge_labels=edge_labels, font_size=15, font_color='red', ax=ax)
    
    # Draw node labels
    nx.draw_networkx_labels(G_top, pos, font_size=12, font_weight='bold', ax=ax)
    
    ax.set_title(f"Top {top_n} Causally Related Stocks on {fin_param} with lag of {lags} quarter(s)", fontsize=16)
    ax.axis('off')
    
    # Display the graph in Streamlit
    st.pyplot(fig)

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
    
    mcap_threshold = st.slider("Market Cap Filter (in $M):", min_value=250, max_value=500_000, value=250_000)
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
    filtered_params = [param for param in financial_parameters]
    
    selected_features = st.multiselect("Select Financial Parameters for Analysis:", filtered_params, default=["revenueusd", "ebitdausd","capex"])

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

    # st.write(df_dict)

    # Part 5: Lag Selection for VARLiNGAM
    st.subheader("VARLiNGAM Model Parameter Selection")

    # Add a slider to allow the user to select the number of lags for VARLiNGAM
    selected_lags = st.slider("Select the number of lags for VARLiNGAM:", min_value=1, max_value=10, value=2)

    # Add a slider to allow the user to select the topn for VARLiNGAM
    selected_topn = st.slider("Select the top N causally related names to display:", min_value=5, max_value=15, value=10)

    # calling the varlingam model:
    varlingam_results = apply_varlingam(df_dict, lags=selected_lags, top_n=selected_topn)

    return varlingam_results

# Call this function in your app.py
if __name__ == "__main__":
    causal_discovery_page()
