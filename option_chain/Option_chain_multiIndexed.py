# import streamlit as st
# import requests
# import pandas as pd
# import time
# from streamlit_autorefresh import st_autorefresh
# import plotly.express as px

# def run_option_chain_app():
#     st.set_page_config(page_title="NSE Option Chain Dashboard", layout="wide", page_icon="📈")

#     # Custom CSS for styling
#     st.markdown("""
#         <style>
#         .title {
#             font-size: 3rem;
#             color: #4CAF50;
#             text-align: center;
#             font-family: 'Arial Black', sans-serif;
#         }
#         .subtitle {
#             font-size: 1.5rem;
#             color: #4CAF50;
#             text-align: center;
#             margin-bottom: 10px;
#             font-family: 'Arial', sans-serif;
#         }
#         .metrics {
#             font-size: 1.2rem;
#             font-family: 'Arial', sans-serif;
#             text-align: center;
#         }
#         .dataframe {
#             border: 2px solid #4CAF50;
#             border-radius: 10px;
#             padding: 10px;
#         }
#         </style>
#     """, unsafe_allow_html=True)

#     # Function to fetch option chain data from NSE
#     def fetch_nse_data(symbol="NIFTY"):
#         url = f'https://www.nseindia.com/api/option-chain-indices?symbol={symbol}'
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#             'Accept-Language': 'en-US,en;q=0.9',
#             'Accept-Encoding': 'gzip, deflate, br',
#             'Connection': 'keep-alive',
#             'Referer': 'https://www.nseindia.com/option-chain',
#             'DNT': '1'  # Do Not Track header
#         }

#         session = requests.Session()
#         response = session.get(url, headers=headers)

#         if response.status_code == 200:
#             return response.json()
#         else:
#             print(f"Failed to fetch data from NSE: {response.status_code}")
#             return None
        
#     # Function to find ATM strike price
#     def find_atm_strike(data):
#         nifty_price = data['records']['underlyingValue']
#         strikes = [record['strikePrice'] for record in data['records']['data'] if 'CE' in record and 'PE' in record]
#         atm_strike = min(strikes, key=lambda x: abs(x - nifty_price))  # Closest strike to index value
#         return atm_strike, nifty_price, strikes

#     # Function to calculate the PCR for all strikes and display selected ones
#     def display_atm_otm_data(data, selected_strike):
#         option_data = []
#         pcr_total_calls = 0
#         pcr_total_puts = 0
#         strikes = []

#         for record in data['records']['data']:
#             if 'CE' in record and 'PE' in record:
#                 strikes.append({
#                     'Strike Price': record['strikePrice'],
#                     'Call Open Interest': record['CE']['openInterest'],
#                     'Put Open Interest': record['PE']['openInterest']
#                 })
#                 pcr_total_calls += record['CE']['openInterest']
#                 pcr_total_puts += record['PE']['openInterest']

#         pcr_all = pcr_total_puts / pcr_total_calls if pcr_total_calls != 0 else 0
#         strikes = sorted(strikes, key=lambda x: x['Strike Price'])
#         selected_index = next(i for i, strike in enumerate(strikes) if strike['Strike Price'] == selected_strike)
#         selected_strikes = strikes[max(0, selected_index - 10): selected_index + 11]

#         return pcr_all, pd.DataFrame(selected_strikes)

#     # Beautified Streamlit app layout
#     def main():
#         # Sidebar for input options
#         st.sidebar.header("Options")
#         st.sidebar.write("Use the controls below to configure the app.")
        
#         # Select index symbol
#         index_symbol = st.sidebar.selectbox("Select Index", options=["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
        
#         refresh_time = st.sidebar.slider("Auto-refresh interval (minutes):", min_value=1, max_value=60, value=5, step=1)
        
#         # Title and intro
#         st.markdown(f"<h1 class='title'>📈 NSE {index_symbol} Option Chain Dashboard</h1>", unsafe_allow_html=True)
#         st.markdown("""
#         <p class='subtitle'>
#         Track real-time data from the NSE Option Chain with key insights such as the Put-Call Ratio (PCR) and ATM strike prices.
#         </p>
#         """, unsafe_allow_html=True)

#         st.write("---")
        
#         if st.sidebar.button('Fetch NSE Data'):
#             with st.spinner("Fetching data..."):
#                 try:
#                     # Fetch data from NSE
#                     nse_data = fetch_nse_data(index_symbol)

#                     if nse_data:
#                         # Find the ATM strike price and all strikes
#                         atm_strike, index_price, strikes = find_atm_strike(nse_data)

#                         st.subheader("📊 Key Metrics")
#                         col1, col2, col3 = st.columns(3)
#                         col1.metric(label=f"{index_symbol} Index Value", value=f"{index_price:.2f}", delta_color="normal")
#                         col2.metric(label="ATM Strike Price", value=f"{atm_strike:.2f}", delta_color="normal")
                        
#                         # Calculate PCR for all strikes
#                         pcr_all, strike_data = display_atm_otm_data(nse_data, atm_strike)

#                         col3.metric(label="Put Call Ratio (PCR)", value=f"{pcr_all:.2f}", delta_color="normal")

#                         # Data Table for selected and nearby strikes
#                         st.write("---")
#                         st.subheader("🔍 Selected and Nearby Strike Prices (10 ATM + 10 OTM)")

#                         # Plot Open Interest Data
#                         fig = px.bar(
#                             strike_data, 
#                             x="Strike Price", 
#                             y=["Call Open Interest", "Put Open Interest"], 
#                             barmode="group",
#                             labels={'value': 'Open Interest', 'variable': 'Type'},
#                             height=400, width=800
#                         )
#                         fig.update_layout(
#                             title=f"Open Interest for Call & Put at Selected Strikes for {index_symbol}",
#                             title_font_size=20
#                         )
#                         st.plotly_chart(fig)

#                         # Display DataFrame for strikes
#                         st.dataframe(strike_data.style.format({'Call Open Interest': '{:.0f}', 'Put Open Interest': '{:.0f}'}), height=300)

#                         st.success(f"Data fetched successfully for {index_symbol}! Auto-refresh will occur every {refresh_time} minutes.")

#                 except Exception as e:
#                     st.error(f"Error: {e}")

#         # Set up auto-refresh (refreshes the app every 'refresh_time' minutes)
#         st_autorefresh(interval=refresh_time * 60 * 1000)  # refresh_time in milliseconds







import streamlit as st
import requests
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# Set up page config at the top
# st.title("NSE Option Chain Dashboard")

def run_option_chain_app():
    # Custom CSS for styling
    st.markdown("""
        <style>
        .title {
            font-size: 3rem;
            color: #4CAF50;
            text-align: center;
            font-family: 'Arial Black', sans-serif;
        }
        .subtitle {
            font-size: 1.5rem;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 10px;
            font-family: 'Arial', sans-serif;
        }
        .metrics {
            font-size: 1.2rem;
            font-family: 'Arial', sans-serif;
            text-align: center;
        }
        .dataframe {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Function to fetch option chain data from NSE
    def fetch_nse_data(symbol="NIFTY"):
        url = f'https://www.nseindia.com/api/option-chain-indices?symbol={symbol}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/option-chain',
            'DNT': '1'  # Do Not Track header
        }

        session = requests.Session()
        response = session.get(url, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data from NSE: {response.status_code}")
            return None
        
    # Function to find ATM strike price
    def find_atm_strike(data):
        nifty_price = data['records']['underlyingValue']
        strikes = [record['strikePrice'] for record in data['records']['data'] if 'CE' in record and 'PE' in record]
        atm_strike = min(strikes, key=lambda x: abs(x - nifty_price))  # Closest strike to index value
        return atm_strike, nifty_price, strikes

    # Function to calculate the PCR for all strikes and display selected ones
    def display_atm_otm_data(data, selected_strike):
        option_data = []
        pcr_total_calls = 0
        pcr_total_puts = 0
        strikes = []

        for record in data['records']['data']:
            if 'CE' in record and 'PE' in record:
                strikes.append({
                    'Strike Price': record['strikePrice'],
                    'Call Open Interest': record['CE']['openInterest'],
                    'Put Open Interest': record['PE']['openInterest']
                })
                pcr_total_calls += record['CE']['openInterest']
                pcr_total_puts += record['PE']['openInterest']

        pcr_all = pcr_total_puts / pcr_total_calls if pcr_total_calls != 0 else 0
        strikes = sorted(strikes, key=lambda x: x['Strike Price'])
        selected_index = next(i for i, strike in enumerate(strikes) if strike['Strike Price'] == selected_strike)
        selected_strikes = strikes[max(0, selected_index - 10): selected_index + 11]

        return pcr_all, pd.DataFrame(selected_strikes)

    # Beautified Streamlit app layout
    def main():
        # Sidebar for input options
        st.sidebar.header("Options")
        st.sidebar.write("Use the controls below to configure the app.")
        
        # Select index symbol
        index_symbol = st.sidebar.selectbox("Select Index", options=["NIFTY", "BANKNIFTY", "FINNIFTY"], index=0)
        
        refresh_time = st.sidebar.slider("Auto-refresh interval (minutes):", min_value=1, max_value=60, value=5, step=1)
        
        # Title and intro
        st.markdown(f"<h1 class='title'>📈 NSE {index_symbol} Option Chain Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p class='subtitle'>
        Track real-time data from the NSE Option Chain with key insights such as the Put-Call Ratio (PCR) and ATM strike prices.
        </p>
        """, unsafe_allow_html=True)

        st.write("---")
        
        if st.sidebar.button('Fetch NSE Data'):
            with st.spinner("Fetching data..."):
                try:
                    # Fetch data from NSE
                    nse_data = fetch_nse_data(index_symbol)

                    if nse_data:
                        # Find the ATM strike price and all strikes
                        atm_strike, index_price, strikes = find_atm_strike(nse_data)

                        st.subheader("📊 Key Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric(label=f"{index_symbol} Index Value", value=f"{index_price:.2f}", delta_color="normal")
                        col2.metric(label="ATM Strike Price", value=f"{atm_strike:.2f}", delta_color="normal")
                        
                        # Calculate PCR for all strikes
                        pcr_all, strike_data = display_atm_otm_data(nse_data, atm_strike)

                        col3.metric(label="Put Call Ratio (PCR)", value=f"{pcr_all:.2f}", delta_color="normal")

                        # Data Table for selected and nearby strikes
                        st.write("---")
                        st.subheader("🔍 Selected and Nearby Strike Prices (10 ATM + 10 OTM)")

                        # Plot Open Interest Data
                        fig = px.bar(
                            strike_data, 
                            x="Strike Price", 
                            y=["Call Open Interest", "Put Open Interest"], 
                            barmode="group",
                            labels={'value': 'Open Interest', 'variable': 'Type'},
                            height=400, width=800
                        )
                        fig.update_layout(
                            title=f"Open Interest for Call & Put at Selected Strikes for {index_symbol}",
                            title_font_size=20
                        )
                        st.plotly_chart(fig)

                        # Display DataFrame for strikes
                        st.dataframe(strike_data.style.format({'Call Open Interest': '{:.0f}', 'Put Open Interest': '{:.0f}'}), height=300)

                        st.success(f"Data fetched successfully for {index_symbol}! Auto-refresh will occur every {refresh_time} minutes.")

                except Exception as e:
                    st.error(f"Error: {e}")

        # Set up auto-refresh (refreshes the app every 'refresh_time' minutes)
        st_autorefresh(interval=refresh_time * 60 * 1000)  # refresh_time in milliseconds
    
    main()


# Call the main function to run the app
