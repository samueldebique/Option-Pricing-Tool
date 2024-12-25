import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

# Black-Scholes formula
def black_scholes(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call and put option prices.

    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility of the underlying asset

    Returns:
    call_price, put_price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return round(call_price, 2), round(put_price, 2)

# Heatmap generator
def generate_heatmap(S_range, sigma_range, K, T, r):
    call_prices = []
    for sigma in sigma_range:
        row = []
        for S in S_range:
            call_price, _ = black_scholes(S, K, T, r, sigma)
            row.append(call_price)
        call_prices.append(row)
    return np.array(call_prices)

# Streamlit App
def main():
    st.title("Black-Scholes Option Pricing Tool")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    S = st.sidebar.number_input("Current Stock Price (S)", min_value=0.0, value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0, step=1.0)
    T = st.sidebar.number_input("Time to Maturity (T in years)", min_value=0.01, value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=0.2, step=0.01)

    # Calculate option prices
    call_price, put_price = black_scholes(S, K, T, r, sigma)

    # Display results
    st.subheader("Option Prices")
    st.write(f"**Call Option Price**: ${call_price}")
    st.write(f"**Put Option Price**: ${put_price}")

    # Heatmap visualization
    st.subheader("Call Option Price Heatmap")
    min_S = st.slider("Minimum Spot Price", 50, 150, 80)
    max_S = st.slider("Maximum Spot Price", 50, 150, 120)
    min_sigma = st.slider("Minimum Volatility", 0.1, 0.5, 0.1)
    max_sigma = st.slider("Maximum Volatility", 0.1, 0.5, 0.4)

    S_range = np.linspace(min_S, max_S, 50)
    sigma_range = np.linspace(min_sigma, max_sigma, 50)
    heatmap = generate_heatmap(S_range, sigma_range, K, T, r)

    # Plot heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(heatmap, extent=[min_S, max_S, min_sigma, max_sigma], origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Spot Price (S)")
    ax.set_ylabel("Volatility (σ)")
    fig.colorbar(cax, label="Call Option Price")
    st.pyplot(fig)

if __name__ == "__main__":
    main()