# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# from tensorflow.keras.models import load_model
# import streamlit as st
# from datetime import datetime, timedelta
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Set start and end dates
# start = '2018-01-01'
# end = datetime.today().strftime('%Y-%m-%d')  # Get today's date

# st.title("STOCK TREND PREDICTION")
# user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# # Download the stock data
# df = yf.download(user_input, start=start, end=end)

# # Describing data
# st.subheader('Data from 2018 to today')
# st.write(df.describe())

# # Visualizations
# st.subheader('Closing Price vs. Time Chart')
# fig = plt.figure(figsize=(12, 6))
# plt.plot(df['Close'])
# st.pyplot(fig)

# st.subheader('Closing Price vs. Time Chart with 100 MA')
# ma100 = df['Close'].rolling(100).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100, label='100 MA', color='orange')
# plt.plot(df['Close'], label='Close Price', color='blue')
# plt.legend()
# st.pyplot(fig)

# st.subheader('Closing Price vs. Time Chart with 100 MA & 200 MA')
# ma200 = df['Close'].rolling(200).mean()
# fig = plt.figure(figsize=(12, 6))
# plt.plot(ma100, 'r', label='100 MA')
# plt.plot(ma200, 'g', label='200 MA')
# plt.plot(df['Close'], 'b', label='Close Price')
# plt.legend()
# st.pyplot(fig)

# # Split data into training and testing
# data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
# data_test = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

# scaler = MinMaxScaler(feature_range=(0, 1))
# data_train_array = scaler.fit_transform(data_train)

# # Splitting data into x_train and y_train
# x_train = []
# y_train = []
# for i in range(100, data_train_array.shape[0]):
#     x_train.append(data_train_array[i-100:i])
#     y_train.append(data_train_array[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)

# # Load the model (LSTM)
# model = load_model('keras_model.h5')

# # Testing part
# past_100_days = data_train.tail(100)
# final_df = pd.concat([past_100_days, data_test], ignore_index=True)
# input_data = scaler.transform(final_df)

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100:i])
#     y_test.append(input_data[i, 0])

# x_test, y_test = np.array(x_test), np.array(y_test)
# y_predicted = model.predict(x_test)

# # Scale back the predicted and actual prices
# y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))  # Reshape to 2D array
# y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape to 2D array

# # Final graph
# st.subheader('Predictions vs Original ')
# fig2 = plt.figure(figsize=(12, 6))
# plt.plot(y_test, 'b', label='Original Price')
# plt.plot(y_predicted, 'r', label='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)

# # Calculate and display performance metrics
# rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
# mae = mean_absolute_error(y_test, y_predicted)

# # Display metrics in the app
# st.subheader('Performance Metrics')
# st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
# st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

# # Ensure y_test and y_predicted are one-dimensional
# y_test_flat = y_test.flatten()
# y_predicted_flat = y_predicted.flatten()

# # Get the last 100 values with dates
# last_100_actual = y_test_flat[-100:]
# last_100_predicted = y_predicted_flat[-100:]
# last_100_dates = df.index[-100:].strftime('%Y-%m-%d')  # Get the last 100 dates

# # Display predicted and actual prices with dates
# st.subheader('Actual Prices vs Predictions for the Last 100 Days')
# comparison_df = pd.DataFrame({
#     'Date': last_100_dates,
#     'Actual Price': last_100_actual, 
#     'Predicted Price': last_100_predicted
# })
# st.write(comparison_df)
# # ........................................................................................................................
# # Predict next 5 days (excluding weekends and starting from today or next business day)
# st.subheader('Predicted Prices for the Next 5 Days')

# def get_next_business_day(date):
#     """ Return the next business day, skipping weekends """
#     while date.weekday() >= 5:  # If it's Saturday (5) or Sunday (6), go to next Monday
#         date += timedelta(1)
#     return date

# # Get today's date and ensure it's a business day
# next_date = get_next_business_day(datetime.today())

# next_5_days = []
# last_15_days = input_data[-15:]  # Get the last 15 days of data

# for i in range(5):
#     # Predict the next day
#     next_day_pred = model.predict(np.array([last_15_days]))
#     next_day_pred = scaler.inverse_transform(next_day_pred)[0, 0]  # Scale back to original value
    
#     # Append the prediction to the list with the corresponding date
#     next_5_days.append({
#         'Date': next_date.strftime('%Y-%m-%d'),
#         'Predicted Price': next_day_pred
#     })
    
#     # Update the next date (skip weekends)
#     next_date = get_next_business_day(next_date + timedelta(1))
    
#     # Update the last_20_days with the predicted value for further predictions
#     next_day_scaled = scaler.transform([[next_day_pred]])  # Rescale for the model
#     last_15_days = np.append(last_15_days[1:], next_day_scaled, axis=0)  # Append the predicted day and remove the first day

# # Convert the predictions to a DataFrame and display
# next_5_days_df = pd.DataFrame(next_5_days)
# st.write(next_5_days_df)










# # test the model by entering ticker like 
# # googl 
# # aapl
# # amzn
# # 
# # kotakbank.ns
# # hdfcbank.ns
# # ongc.ns
# # lici.ns
# # vedl.ns
# # hindunilvr.ns

# stock_prediction.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_stock_prediction_app():
    # Set start and end dates
    start = '2018-01-01'
    end = datetime.today().strftime('%Y-%m-%d')  # Get today's date

    st.title("STOCK TREND PREDICTION")
    user_input = st.text_input('Enter Stock Ticker', 'AAPL')

    # Download the stock data
    df = yf.download(user_input, start=start, end=end)

    # Describing data
    st.subheader('Data from 2018 to today')
    st.write(df.describe())

    # Visualizations
    st.subheader('Closing Price vs. Time Chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'])
    st.pyplot(fig)

    st.subheader('Closing Price vs. Time Chart with 100 MA')
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100 MA', color='orange')
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Closing Price vs. Time Chart with 100 MA & 200 MA')
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'r', label='100 MA')
    plt.plot(ma200, 'g', label='200 MA')
    plt.plot(df['Close'], 'b', label='Close Price')
    plt.legend()
    st.pyplot(fig)

    # Split data into training and testing
    data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
    data_test = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_array = scaler.fit_transform(data_train)

    # Splitting data into x_train and y_train
    x_train = []
    y_train = []
    for i in range(100, data_train_array.shape[0]):
        x_train.append(data_train_array[i-100:i])
        y_train.append(data_train_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Load the model (LSTM)
    model = load_model('stock_prediction/keras_model.h5')

    # Testing part
    past_100_days = data_train.tail(100)
    final_df = pd.concat([past_100_days, data_test], ignore_index=True)
    input_data = scaler.transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)

    # Scale back the predicted and actual prices
    y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1))  # Reshape to 2D array
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape to 2D array

    # Final graph
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label='Original Price')
    plt.plot(y_predicted, 'r', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    # Calculate and display performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    mae = mean_absolute_error(y_test, y_predicted)

    # Display metrics in the app
    st.subheader('Performance Metrics')
    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    st.write(f'Mean Absolute Error (MAE): {mae:.2f}')

    # Ensure y_test and y_predicted are one-dimensional
    y_test_flat = y_test.flatten()
    y_predicted_flat = y_predicted.flatten()

    # Get the last 100 values with dates
    last_100_actual = y_test_flat[-100:]
    last_100_predicted = y_predicted_flat[-100:]
    last_100_dates = df.index[-100:].strftime('%Y-%m-%d')  # Get the last 100 dates

    # Display predicted and actual prices with dates
    st.subheader('Actual Prices vs Predictions for the Last 100 Days')
    comparison_df = pd.DataFrame({
        'Date': last_100_dates,
        'Actual Price': last_100_actual, 
        'Predicted Price': last_100_predicted
    })
    st.write(comparison_df)

    # Predict next 5 days
    st.subheader('Predicted Prices for the Next 5 Days')

    def get_next_business_day(date):
        """ Return the next business day, skipping weekends """
        while date.weekday() >= 5:  # If it's Saturday (5) or Sunday (6), go to next Monday
            date += timedelta(1)
        return date

    # Get today's date and ensure it's a business day
    next_date = get_next_business_day(datetime.today())

    next_5_days = []
    last_15_days = input_data[-15:]  # Get the last 15 days of data

    for i in range(5):
        # Predict the next day
        next_day_pred = model.predict(np.array([last_15_days]))
        next_day_pred = scaler.inverse_transform(next_day_pred)[0, 0]  # Scale back to original value
        
        # Append the prediction to the list with the corresponding date
        next_5_days.append({
            'Date': next_date.strftime('%Y-%m-%d'),
            'Predicted Price': next_day_pred
        })
        
        # Update the next date (skip weekends)
        next_date = get_next_business_day(next_date + timedelta(1))
        
        # Update the last_15_days with the predicted value for further predictions
        next_day_scaled = scaler.transform([[next_day_pred]])  # Rescale for the model
        last_15_days = np.append(last_15_days[1:], next_day_scaled, axis=0)  # Append the predicted day and remove the first day

    # Convert the predictions to a DataFrame and display
    next_5_days_df = pd.DataFrame(next_5_days)
    st.write(next_5_days_df)
