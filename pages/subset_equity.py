# pages/causal_discovery.py
import streamlit as st
import pandas as pd
from data.data_load import get_filtered_data
import lingam
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
from data.data_load import load_data
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# def apply_varlingam(df_dict, lags=1, top_n=10, bootstrap=False, n_sampling=5):
#     """
#     Apply the VARLiNGAM model to the dataframes in df_dict.
    
#     Parameters:
#         df_dict (dict): A dictionary where keys are financial parameters and values are dataframes.
#         lags (int): The number of lags for the VARLiNGAM model.
#         top_n (int): Number of top nodes to visualize based on connection strength.
#         bootstrap (bool): Whether to use bootstrap sampling.
#         n_sampling (int): Number of bootstrap samples if using bootstrap.
        
#     Returns:
#         varlingam_results (dict): A dictionary storing the results for each financial parameter.
#     """
#     varlingam_results = {}  # To store results for each parameter
    
#     for feature, df_variable in df_dict.items():
#         if df_variable.shape[1] < 2:
#             print(f"Skipping {feature} because it has less than 2 valid stocks for analysis.")
#             continue
        
#         print(f"Applying VARLiNGAM for {feature} with {lags} lags...")
        
#         # Initialize and fit the VARLiNGAM model
#         model = lingam.VARLiNGAM(lags=lags)
        
#         if bootstrap:
#             result = model.bootstrap(df_variable, n_sampling=n_sampling)
#             adjacency_matrices = result.adjacency_matrices_
#             causal_order = result.causal_order_
#         else:
#             model.fit(df_variable)
#             adjacency_matrices = model.adjacency_matrices_
#             causal_order = model.causal_order_

#         # Get the original labels for the stocks (columns in df_variable)
#         org_labels = df_variable.columns.to_list()
#         order_labels = df_variable.iloc[:, causal_order].columns.to_list()

#         # Store results in varlingam_results
#         varlingam_results[feature] = {
#             "causal_order": causal_order,
#             "ordered_labels": order_labels,
#             "original_labels": org_labels,
#             "adjacency_matrices": adjacency_matrices,
#             "model": model
#         }

#         # Visualize the causal graph for the current financial parameter
#         print(f"Visualizing the causal graph for {feature}")
#         st.subheader(f"Causal Analysis: {feature}")
#         visualize_causal_graph(adjacency_matrices, org_labels, feature,lags, top_n=top_n)
#         visualize_interactive_adjacency_matrix(adjacency_matrices, org_labels, feature,lags)
#     return varlingam_results

# def visualize_causal_graph(adjacency_matrices, labels, fin_param, lags, top_n=10):
#     """
#     Visualize the causal relationships using an interactive graph.

#     Parameters:
#         adjacency_matrices (list): A list of adjacency matrices from the VARLiNGAM model.
#         labels (list): A list of stock ticker labels.
#         top_n (int): Number of top nodes to visualize based on connection strength.
#     """
#     G = nx.DiGraph()
#     num_vars = len(labels)
    
#     # Build the graph
#     for i in range(num_vars):
#         for j in range(num_vars):
#             if i != j and any(adj[i, j] != 0 for adj in adjacency_matrices):
#                 weight = sum(adj[i, j] for adj in adjacency_matrices)
#                 G.add_edge(labels[i], labels[j], weight=weight)

#     # Calculate total connection strength for each node (sum of weights of edges in and out of the node)
#     node_strength = {node: sum(abs(G.edges[node, n]['weight']) for n in G[node]) + 
#                             sum(abs(G.edges[n, node]['weight']) for n in G.pred[node])
#                      for node in G.nodes()}
    
#     # Sort nodes by strength and select top_n nodes
#     top_nodes = sorted(node_strength.items(), key=lambda x: x[1], reverse=True)[:top_n]
#     top_labels = [node for node, _ in top_nodes]
    
#     # Create a subgraph for the top_n nodes
#     G_top = G.subgraph(top_labels)
    
#     # Spring layout for better visualization
#     pos = nx.spring_layout(G_top, seed=42)

#     # Extract node positions
#     node_x = [pos[node][0] for node in G_top.nodes()]
#     node_y = [pos[node][1] for node in G_top.nodes()]

#     # Define node trace (increase size and font for better visibility)
#     node_trace = go.Scatter(
#         x=node_x, 
#         y=node_y, 
#         mode='markers+text',
#         text=[f'{node}' for node in G_top.nodes()],
#         marker=dict(
#             showscale=True,
#             colorscale='Viridis',
#             size=[30 for _ in G_top.nodes()],  # Increase node size
#             color=[node_strength[node] for node in G_top.nodes()],
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Influence',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line=dict(width=3)  # Slightly thicker node borders
#         ),
#         textfont=dict(
#             size=20,  # Increase text size for better readability
#             color='green'
#         ),
#         textposition="bottom center",
#         hoverinfo='text'
#     )

#     # Extract edge traces (adjusting the width to reflect strength of relationships)
#     edge_traces = []
#     for edge in G_top.edges(data=True):
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         weight = abs(edge[2]['weight'])  # Ensure the edge width is positive
        
#         # Define a minimum width for better visibility
#         width = max(weight * 3.5, 1)  # Increased scaling factor for clearer differentiation

#         edge_traces.append(
#             go.Scatter(
#                 x=[x0, x1, None], 
#                 y=[y0, y1, None],
#                 line=dict(width=width, color='black'),  # Thicker line for higher weights
#                 mode='lines',
#                 hoverinfo='none'
#             )
#         )

#     # Combine traces
#     fig = go.Figure(data=edge_traces + [node_trace])

#     # Add layout details
#     fig.update_layout(
#         showlegend=False,
#         hovermode='closest',
#         title=f"Top {top_n} Causally Related Stocks on {fin_param} with lag of {lags} quarter(s)",
#         titlefont_size=20,  # Slightly larger title
#         margin=dict(b=0, l=0, r=0, t=60),  # More space for title
#         annotations=[dict(
#             text=f"",
#             showarrow=False,
#             xref="paper", 
#             yref="paper",
#             x=0.005, 
#             y=-0.002
#         )],
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
#     )

#     # Display in Streamlit
#     st.plotly_chart(fig, use_container_width=True)
# import plotly.graph_objs as go
# import numpy as np

# def visualize_interactive_adjacency_matrix(adjacency_matrices, labels, fin_param, lags):
#     """
#     Visualize the adjacency matrix as an interactive heatmap to show causal relationships.

#     Parameters:
#         adjacency_matrices (list): A list of adjacency matrices from the VARLiNGAM model.
#         labels (list): A list of stock ticker labels.
#         fin_param (str): The financial parameter being analyzed.
#         lags (int): The number of lags in the model.
#     """
#     # Aggregate adjacency matrices if there are multiple (e.g., from bootstrapping)
#     aggregated_matrix = np.mean(adjacency_matrices, axis=0)

#     # Create hover text for each cell
#     hover_text = [[f"{labels[i]} -> {labels[j]}<br>Causal Coefficient: {aggregated_matrix[i][j]:.2f}"
#                    for j in range(len(labels))] for i in range(len(labels))]

#     # Create a Plotly heatmap
#     heatmap = go.Heatmap(
#         z=aggregated_matrix,  # The values (causal coefficients)
#         x=labels,  # x-axis labels (stock tickers)
#         y=labels,  # y-axis labels (stock tickers)
#         colorscale='RdBu',  # Diverging color scale (red-blue)
#         zmin=-np.max(np.abs(aggregated_matrix)),  # Ensure that both sides of the color scale are symmetric
#         zmax=np.max(np.abs(aggregated_matrix)),
#         hoverinfo="text",  # Display hover text
#         text=hover_text,  # Hover text showing stock pairs and values
#         colorbar=dict(
#             title="Causal Coefficient",
#             thickness=15,
#             titleside='right'
#         )
#     )

#     # Determine the size based on the number of labels for a square layout
#     size_per_label = 55  # Size per label in pixels
#     matrix_size = len(labels) * size_per_label

#     layout = go.Layout(
#         title=f"Causal Relationships Adjacency Matrix for {fin_param} with Lag of {lags} quarter(s)",
#         xaxis=dict(ticks='',showgrid=False, side="bottom"),
#         yaxis=dict(ticks='',showgrid=False, autorange="reversed"),
#         # width=matrix_size,  # Ensure the width matches the height for a square
#         # height=matrix_size,  # Ensure the height matches the width for a square
#         hovermode='closest',
#         autosize=False  # Ensure the plot maintains the specified width and height
#     )

#     # Create the figure and display it
#     fig = go.Figure(data=[heatmap], layout=layout)
    
#     # Display in Streamlit
#     st.plotly_chart(fig, use_container_width=True)

# def apply_varlingam_rolling_window(df_dict, lags=1, window_size=5, top_n=10, bootstrap=False, n_sampling=5):
#     """
#     Apply the VARLiNGAM model to the dataframes in df_dict over rolling windows and extract time-varying coefficients.

#     Parameters:
#         df_dict (dict): A dictionary where keys are financial parameters and values are dataframes.
#         lags (int): The number of lags for the VARLiNGAM model.
#         window_size (int): The size of the rolling window in years.
#         top_n (int): Number of top nodes to visualize based on connection strength.
#         bootstrap (bool): Whether to use bootstrap sampling.
#         n_sampling (int): Number of bootstrap samples if using bootstrap.

#     Returns:
#         rolling_coeff_results (dict): A dictionary storing the rolling coefficients for each stock pair over time.
#     """
#     rolling_coeff_results = {}  # To store time-varying causal coefficients for each stock pair

#     for feature, df_variable in df_dict.items():
#         if df_variable.shape[1] < 2:
#             st.warning(f"Skipping {feature} because it has less than 2 valid stocks for analysis.")
#             continue

#         st.info(f"Applying VARLiNGAM for {feature} over rolling windows...")

#         # Ensure df_variable has a 'year' column
#         df_variable['year'] = df_variable.index.year
#         unique_years = df_variable['year'].unique()

#         rolling_windows = []  # To store results for each rolling window

#         # Apply VARLiNGAM for each rolling window
#         for start_year in range(unique_years.min(), unique_years.max() - window_size + 1):
#             end_year = start_year + window_size
#             df_window = df_variable[(df_variable['year'] >= start_year) & (df_variable['year'] < end_year)].drop(columns='year')

#             # Ensure df_window has enough data
#             if df_window.empty or df_window.shape[0] <= lags or df_window.shape[1] < 2:
#                 st.warning(f"Skipping window {start_year}-{end_year} for {feature} due to insufficient data.")
#                 continue

#             # Initialize and fit the VARLiNGAM model
#             model = lingam.VARLiNGAM(lags=lags)
#             try:
#                 if bootstrap:
#                     result = model.bootstrap(df_window, n_sampling=n_sampling)
#                     adjacency_matrices = result.adjacency_matrices_
#                 else:
#                     model.fit(df_window)
#                     adjacency_matrices = model.adjacency_matrices_

#                 # Store the adjacency matrix for the rolling window
#                 rolling_windows.append((start_year, end_year, adjacency_matrices[0]))  # Store the first lag only

#             except np.linalg.LinAlgError as e:
#                 st.error(f"Error fitting VARLiNGAM model for {feature} in window {start_year}-{end_year}: {e}")
#                 continue

#         # Store the rolling coefficients for the feature
#         rolling_coeff_results[feature] = {
#             'rolling_windows': rolling_windows,  # Store the adjacency matrices over the rolling windows
#             'original_labels': df_variable.columns.to_list()
#         }

#         # Ask the user to pick two stocks and show coefficient evolution over time
#         st.subheader(f"Rolling Coefficients for {feature}")
#         org_labels = df_variable.columns.drop('year').to_list()
#         selected_pair = st.multiselect(f"Select two stocks for {feature} to view their rolling causal relationship:",
#                                        org_labels, default=org_labels[:2])

#         if len(selected_pair) == 2:
#             # Extract and plot the rolling coefficients between the two selected stocks
#             plot_rolling_coefficient_time_series(rolling_windows, org_labels, selected_pair)

#     return rolling_coeff_results


# def plot_rolling_coefficient_time_series(rolling_windows, labels, selected_pair):
#     """
#     Plot the time series of rolling causal coefficients between two selected stocks.

#     Parameters:
#         rolling_windows (list): A list of tuples (start_year, end_year, adjacency_matrix) for each rolling window.
#         labels (list): A list of stock ticker labels.
#         selected_pair (list): The selected pair of stocks to plot the rolling causal coefficients.
#     """
#     # Find the index of the selected pair in the adjacency matrix
#     idx_1 = labels.index(selected_pair[0])
#     idx_2 = labels.index(selected_pair[1])

#     # Extract the rolling coefficients between the two stocks
#     coefficients_1_to_2 = [adj_matrix[idx_1, idx_2] for _, _, adj_matrix in rolling_windows]
#     coefficients_2_to_1 = [adj_matrix[idx_2, idx_1] for _, _, adj_matrix in rolling_windows]

#     # Extract the years for the x-axis
#     years = [f"{start}-{end}" for start, end, _ in rolling_windows]

#     # Plot the rolling coefficients
#     plt.figure(figsize=(10, 5))
#     plt.plot(years, coefficients_1_to_2, label=f'{selected_pair[0]} -> {selected_pair[1]}', color='blue', marker='o')
#     plt.plot(years, coefficients_2_to_1, label=f'{selected_pair[1]} -> {selected_pair[0]}', color='green', marker='o')
#     plt.xlabel('Time Windows')
#     plt.ylabel('Causal Coefficient')
#     plt.xticks(rotation=45)
#     plt.title(f'Rolling Causal Coefficients Between {selected_pair[0]} and {selected_pair[1]}')
#     plt.legend()
#     plt.grid(True)
#     st.pyplot(plt)


# def causal_discovery_page():
#     st.header("Equity Causal Discovery Exploration")

#     # Load and cache the dataset
#     df = load_data()

#     # Part 1: Ticker Selection (Custom Selection of Stocks)
#     st.subheader("Select Firms (Tickers) for Analysis")

#     # Extract unique tickers from the dataset
#     tickers = df['ticker'].unique()

#     # Magnificent 7 stock tickers
#     magnificent_7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
#     # Multi-select for users to choose specific tickers, defaulting to the Magnificent 7
#     selected_tickers = st.multiselect(
#         "Select Firms (Tickers) for Analysis:", 
#         tickers, 
#         default=[ticker for ticker in magnificent_7 if ticker in tickers]
#     )

#     if len(selected_tickers) < 3:
#         st.warning("Please select at least three tickers.")
#         return

#     # Filter the dataset based on the selected tickers
#     df_firm_filtered = df[df['ticker'].isin(selected_tickers)]

#     # Part 2: Variable Selection
#     st.subheader("Variable Selection")

#     # Search and Filter for Variables (Features)
#     financial_parameters = df.columns.tolist()
#     filtered_params = [param for param in financial_parameters]

#     # Multi-select for financial parameters to analyze
#     selected_features = st.multiselect("Select Financial Parameters for Analysis:", filtered_params, default=["revenueusd", "ebitdausd", "capex"])

#     # Create a dictionary of DataFrames, one for each selected variable
#     df_dict = {}
#     for feature in selected_features:
#         df_dict[feature] = get_filtered_data(df_firm_filtered, feature)

#     # Part 3: Timeframe and Frequency Selection
#     st.subheader("Timeframe Selection and Frequency Adjustment")

#     # Ensure date range is set using datetime values, replacing NaT with a default min/max date
#     min_date = df_firm_filtered['calendardate'].min().date() if not df_firm_filtered.empty else pd.to_datetime('2000-01-01').date()
#     max_date = df_firm_filtered['calendardate'].max().date() if not df_firm_filtered.empty else pd.to_datetime('2020-01-01').date()

#     date_range = st.date_input("Select Date Range:", [min_date, max_date])

#     # Frequency Adjustment
#     frequency = st.selectbox("Choose Data Frequency:", ["Quarterly", "Annually"], index=0)

#     # Apply date filtering and frequency adjustment to all DataFrames in the dictionary
#     for feature in selected_features:
#         df_variable = df_dict[feature]

#         # Filter data based on the selected date range
#         df_variable = df_variable[
#             (df_variable.index >= pd.to_datetime(date_range[0])) &
#             (df_variable.index <= pd.to_datetime(date_range[1]))
#         ]

#         # Apply frequency resampling
#         if frequency == "Quarterly":
#             df_variable = df_variable.resample('Q').last()
#         elif frequency == "Annually":
#             df_variable = df_variable.resample('A').last()

#         # Update the dictionary with the processed DataFrame
#         df_dict[feature] = df_variable

#     # Display the filtered and resampled DataFrames
#     st.subheader("Filtered and Resampled Data")
#     for feature, df_variable in df_dict.items():
#         st.write(f"**{feature} Dataset**")
#         st.write(df_variable.tail(5))
#         st.write("Dataset Shape: ", df_variable.shape)

#     # Part 4: Lag Selection for VARLiNGAM
#     st.subheader("VARLiNGAM Model Parameter Selection")

#     # Add a slider to allow the user to select the number of lags for VARLiNGAM
#     selected_lags = st.slider("Select the number of lags for VARLiNGAM:", min_value=1, max_value=10, value=2)

#     # Add a slider to allow the user to select the topn for VARLiNGAM
#     selected_topn = st.slider("Select the top N causally related names to display:", min_value=5, max_value=15, value=10)

#     # # Add a checkbox for bootstrapping option
#     # bootstrap_enabled = st.checkbox("Enable Bootstrapping for VARLiNGAM?", value=False)

#     # # If bootstrapping is enabled, allow user to select number of bootstrap samples
#     # if bootstrap_enabled:
#     #     n_sampling = st.slider("Select Number of Bootstrap Samples:", min_value=1, max_value=100, value=5)
#     # else:
#     #     n_sampling = 5  # Default value, not used unless bootstrap is enabled

#     # calling the varlingam model:
#     varlingam_results = apply_varlingam(df_dict, lags=selected_lags, top_n=selected_topn)
#     varlingam_results = apply_varlingam_rolling_window(df_dict, lags=selected_lags)

#     return varlingam_results

# Call this function in your app.py
if __name__ == "__main__":
    st.subheader("Equity Causal Relations: Evolution over the years")
    st.write("Currently under developement!")
    # causal_discovery_page()
