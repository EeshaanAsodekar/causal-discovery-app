import streamlit as st

def main():
    st.title("Overview: Causal Discovery and Dataset Insights")

    # Detailed explanation of causal discovery
    st.header("What is Causal Discovery?")
    st.write("""
        Causal discovery is a statistical method used to uncover the underlying cause-and-effect relationships between variables in a dataset.
        In finance, it helps determine how certain variables like revenue, cash flow, assets, liabilities, and other metrics
        influence each other over time. Causal discovery goes beyond correlation analysis, where two variables might move together without any causal relationship.
        
        By uncovering causal relationships, we can better understand which variables drive changes in others, enabling more informed decision-making 
        for forecasting, risk management, and investment strategy development.
        """)

    st.subheader("Mathematical Representation of Time Series Relationships:")
    st.write("""
        In the context of time series data, causal discovery focuses on analyzing how past values of variables affect their future states.
        Consider a set of variables (e.g., revenue, net income, etc.) at time t. We can model these variables as a 
        linear combination of their past values (lags), represented by matrices A and B, along with noise terms. 
        This results in the following system of equations:
        """)
    
    st.latex(r'''
        \mathbf{X}_t = A \mathbf{X}_{t-1} + B \mathbf{X}_{t-2} + \dots + \epsilon_t
        ''')
    
    st.write("""
        In this equation:
        - X_t is the vector of variables at time t.
        - A, B, etc., are matrices that represent the strength of the causal influence of previous time steps.
        - epsilon_t is the noise term, capturing unobserved factors or random effects.
        
        The goal of causal discovery is to estimate the structure of the matrices (e.g., which elements of A, B are non-zero)
        and to quantify the strength of those relationships. This helps us identify the variables that drive others over time.
        """)

    # Detailed explanation of VARLiNGAM
    st.header("What is VARLiNGAM?")
    st.write("""
        VARLiNGAM (Vector AutoRegressive Linear Non-Gaussian Acyclic Model) is an advanced statistical technique designed to discover causal 
        relationships between time series data. Unlike traditional models, VARLiNGAM focuses on uncovering the underlying causal structure 
        between variables while allowing for non-Gaussian noise. Non-Gaussian noise is common in financial datasets, making VARLiNGAM a better 
        fit for analyzing economic data where distributions often deviate from normality.

        The core idea of VARLiNGAM is to apply a Vector AutoRegressive (VAR) model but with additional constraints that allow us to uncover 
        the directionality and structure of causal relationships.
        """)
    
    st.subheader("Mathematical Formulation of VARLiNGAM:")
    st.write("""
        In VARLiNGAM, we extend the basic VAR model to include multiple time lags and impose a causal structure on the relationships 
        between variables. The equation becomes:
        """)
    
    st.latex(r'''
        \mathbf{X}_t = \sum_{i=1}^{k} A_i \mathbf{X}_{t-i} + \epsilon_t
        ''')
    
    st.write("""
        Here:
        - X_t is the vector of variables (e.g., revenue, net income) at time t.
        - A_i are the matrices representing the causal relationship between variables at time lag i.
        - k is the number of time lags considered.
        - epsilon_t is the error or noise term, assumed to be non-Gaussian in VARLiNGAM.

        The key difference between a standard VAR model and VARLiNGAM is that VARLiNGAM leverages non-Gaussianity to detect 
        causal directions. In other words, VARLiNGAM does not just capture correlations but also uncovers which variables are 
        causing changes in others.
        """)

    st.subheader("Identifying Causality in Financial Data")
    st.write("""
        In financial datasets, variables often exhibit complex interdependencies. For example, changes in market conditions (e.g., price, revenue)
        may affect a company's cash flow, while changes in interest rates may influence liabilities. Using VARLiNGAM, we can identify such causal 
        links and quantify the extent to which one variable drives another over time.

        The model estimates the causal ordering of variables by analyzing the structure of the data matrix and leveraging non-Gaussian assumptions. 
        By understanding the direction of causality, we can build better predictive models and make more informed financial decisions.
        """)
    
if __name__ == "__main__":
    main()
