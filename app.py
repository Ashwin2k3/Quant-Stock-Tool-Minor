# # smartetrade/app.py
# import streamlit as st
# from sentiment_analysis.sentiment_analysis import show_sentiment_analysis
# from option_chain.Option_chain_multiIndexed import run_option_chain_app
# from stock_prediction.stock import run_stock_prediction_app

# # Sidebar navigation
# st.sidebar.title('Navigation')
# app_selection = st.sidebar.radio('Go to', ('Sentiment Analysis', 'Option Chain', 'Stock Prediction'))

# # Main app logic
# if app_selection == 'Sentiment Analysis':
#     show_sentiment_analysis()  # Function to run sentiment analysis app
# elif app_selection == 'Option Chain':
#     run_option_chain_app()  # Function to run option chain app
# elif app_selection == 'Stock Prediction':
#     run_stock_prediction_app()  # Function to run stock prediction app

import streamlit as st
from sentiment_analysis.sentiment_analysis import show_sentiment_analysis
from option_chain.Option_chain_multiIndexed import run_option_chain_app
from stock_prediction.stock import run_stock_prediction_app






# Sidebar navigation
st.sidebar.title('Navigation')
app_selection = st.sidebar.radio('Go to', ('Sentiment Analysis', 'Option Chain', 'Stock Prediction'))


# Main app logic
if app_selection == 'Sentiment Analysis':

    show_sentiment_analysis()  # Function to run sentiment analysis app
elif app_selection == 'Option Chain':

    run_option_chain_app()  # Function to run option chain app
elif app_selection == 'Stock Prediction':

    run_stock_prediction_app()  # Function to run stock prediction app
