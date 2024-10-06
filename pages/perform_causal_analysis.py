# pages/causal_discovery.py
import streamlit as st
import pandas as pd
from data.data_load import get_filtered_data  # Assuming this function is already defined
import lingam
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
from data.data_load import load_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        st.subheader(f"Causal Analysis: {feature}")
        visualize_causal_graph(adjacency_matrices, org_labels, feature,lags, top_n=top_n)
        visualize_adjacency_matrix(adjacency_matrices, org_labels, feature,lags)
    return varlingam_results

def visualize_causal_graph(adjacency_matrices, labels, fin_param, lags, top_n=10):
    """
    Visualize the causal relationships using an interactive graph.

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

    # Calculate total connection strength for each node (sum of weights of edges in and out of the node)
    node_strength = {node: sum(abs(G.edges[node, n]['weight']) for n in G[node]) + 
                            sum(abs(G.edges[n, node]['weight']) for n in G.pred[node])
                     for node in G.nodes()}
    
    # Sort nodes by strength and select top_n nodes
    top_nodes = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_labels = [node for node, _ in top_nodes]
    
    # Create a subgraph for the top_n nodes
    G_top = G.subgraph(top_labels)
    
    # Spring layout for better visualization
    pos = nx.spring_layout(G_top, seed=42)

    # Extract node positions
    node_x = [pos[node][0] for node in G_top.nodes()]
    node_y = [pos[node][1] for node in G_top.nodes()]

    # Define node trace (increase size and font for better visibility)
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y, 
        mode='markers+text',
        text=[f'{node}' for node in G_top.nodes()],
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=[30 for _ in G_top.nodes()],  # Increase node size
            color=[node_strength[node] for node in G_top.nodes()],
            colorbar=dict(
                thickness=15,
                title='Node Influence',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=3)  # Slightly thicker node borders
        ),
        textfont=dict(
            size=20,  # Increase text size for better readability
            color='green'
        ),
        textposition="bottom center",
        hoverinfo='text'
    )

    # Extract edge traces (adjusting the width to reflect strength of relationships)
    edge_traces = []
    for edge in G_top.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = abs(edge[2]['weight'])  # Ensure the edge width is positive
        
        # Define a minimum width for better visibility
        width = max(weight * 3.5, 1)  # Increased scaling factor for clearer differentiation

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None], 
                y=[y0, y1, None],
                line=dict(width=width, color='black'),  # Thicker line for higher weights
                mode='lines',
                hoverinfo='none'
            )
        )

    # Combine traces
    fig = go.Figure(data=edge_traces + [node_trace])

    # Add layout details
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        title=f"Top {top_n} Causally Related Stocks on {fin_param} with lag of {lags} quarter(s)",
        titlefont_size=20,  # Slightly larger title
        margin=dict(b=0, l=0, r=0, t=60),  # More space for title
        annotations=[dict(
            text=f"",
            showarrow=False,
            xref="paper", 
            yref="paper",
            x=0.005, 
            y=-0.002
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def visualize_adjacency_matrix(adjacency_matrices, labels, fin_param, lags):
    """
    Visualize the adjacency matrix as a heatmap to show causal relationships.
    
    Parameters:
        adjacency_matrices (list): A list of adjacency matrices from the VARLiNGAM model.
        labels (list): A list of stock ticker labels.
        fin_param (str): The financial parameter being analyzed.
        lags (int): The number of lags in the model.
    """
    # Aggregate adjacency matrices if there are multiple (e.g., from bootstrapping)
    aggregated_matrix = np.mean(adjacency_matrices, axis=0)

    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 10))
    
    # Create a mask to only show the upper triangle since the graph is directed and some entries may be symmetric
    mask = np.triu(np.ones_like(aggregated_matrix, dtype=bool))

    # Generate a custom diverging colormap (e.g., red for negative, blue for positive)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        aggregated_matrix, 
        mask=mask, 
        cmap=cmap, 
        annot=True,  # Annotate with the values
        fmt=".2f",  # Limit to two decimal places
        linewidths=0.5,  # Add lines between cells
        square=True,  # Keep the cells square
        cbar_kws={"shrink": .75, "label": "Causal Coefficients"},  # Colorbar options
        xticklabels=labels, 
        yticklabels=labels
    )

    # Title and axis labels
    plt.title(f"Causal Relationships Adjacency Matrix for {fin_param} with Lag of {lags} quarter(s)", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(plt)



def causal_discovery_page():
    st.title("Equity Causal Discovery")

    # Load and cache the dataset
    df = load_data()

    # # Part 1: Firm Selection (All Stocks Automatically Selected)
    # st.header("All Firms Selected")

    # # Use all stocks (No filtering)
    # st.write("Using all stocks for the analysis.")
    # st.write(df.head())  # Display some rows from the full dataset
    # st.write(df['ticker'].nunique())


    # # Display DataFrame with all stocks
    # st.write("Displaying data for all stocks:")
    # st.write(df.head())

    # Part 2: Apply Market Cap Filter on the Last Row for Each Stock
    st.subheader("Market Cap Filter")

    # Get the latest market cap (as of the last available row) for each stock
    df_last_row = df.groupby('ticker').last().reset_index()
    
    mcap_threshold = st.slider("Market Cap Filter (in $M):", min_value=250, max_value=500_000, value=250_000)
    df_filtered_mcap = df_last_row[df_last_row['marketcap'] >= mcap_threshold * 1_000_000]

    # Filter original DataFrame based on selected stocks that meet the market cap requirement
    df_firm_filtered = df[df['ticker'].isin(df_filtered_mcap['ticker'])]

    # # Display DataFrame after applying market cap filter
    # st.write("Data after applying market cap filter:")
    # st.write(df_firm_filtered.head())
    # st.write(df_firm_filtered['ticker'].nunique())

    # Part 3: Variable Selection
    st.subheader("Variable Selection")

    # Search and Filter for Variables (Features)
    financial_parameters = df.columns.tolist()
    filtered_params = [param for param in financial_parameters]
    
    selected_features = st.multiselect("Select Financial Parameters for Analysis:", filtered_params, default=["revenueusd", "ebitdausd","capex"])

    # Create a dictionary of DataFrames, one for each selected variable
    df_dict = {}
    for feature in selected_features:
        df_dict[feature] = get_filtered_data(df_firm_filtered, feature)  # Use get_filtered_data for each selected feature
    
    # Display each DataFrame for the selected features
    # st.subheader("Data for Selected Variables")
    # for feature, df_variable in df_dict.items():
    #     st.subheader(f"Data for {feature}")
    #     st.write(df_variable.head())
    #     st.write(df_variable.shape)

    # Part 4: Timeframe and Frequency Selection
    st.subheader("Timeframe Selection and Frequency Adjustment")

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
    st.subheader("Filtered and Resampled Data")
    for feature, df_variable in df_dict.items():
        st.write(f"**{feature} Dataset**")
        # st.write(df_variable.head(3))
        st.write(df_variable.tail(5))
        st.write("Dataset Shape: ", df_variable.shape)

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
