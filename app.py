import sklearn.preprocessing
import  streamlit as st
import numpy as np
import pandas as pd
import datetime as datetime
import pandas_datareader
import plotly.graph_objects as go
import sklearn.linear_model
import sklearn.model_selection
from PIL import Image
import yfinance as yf

yf.pdr_override()

st.title('Stock Price Predict APP with AI')
st.write('Let\'s predict StokPrice')

image = Image.open('stock_predict.png')
st.image(image, use_container_width=True)

st.write('It\'s Just a predict.')

st.header('Input a  ticker symbol')
stock_name = st.text_input('ex: AAPL, FB, SFTBY',"AAPL")

stock_name = stock_name.upper()

link='https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html'
st.markdown(link)
st.write('Could you  prease reffer the link above.')

try:

    df_stock = pandas_datareader.data.get_data_yahoo(stock_name, '2021-01-05')
    st.header(stock_name + 'StockPrice(USD) from 20222Jan15 til today.')
    st.write(df_stock)

    st.header(stock_name + 'final price and average 14days')
    df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
    df_stock2 = df_stock[['Close','SMA']]
    st.line_chart(df_stock2)

    st.header(stock_name + 'price movements(USD)')
    df_stock['change'] = (((df_stock['Close'] - df_stock['Open'])) / (df_stock['Open'])* 100)
    st.line_chart(df_stock['change'].tail(100))

    fig = go.Figure(
        data=[go.Candlestick(
            x = df_stock.index,
            open = df_stock['Open'],
            high = df_stock['High'],
            low = df_stock['Low'],
            close = df_stock['Close'],
            increasing_line_color = 'green',
            decreasing_line_color = 'red',

        )
        ]
    )

    st.header(stock_name + 'CandleStick')
    st.plotly_chart(fig, use_column_width=True)


    df_stock['label'] = df_stock['Close'].shift(-30)

    st.header(stock_name + 'Let\'s predict the price of 1 month leter.')
    def stock_predict():
        X = np.array(df_stock.drop(['label','SMA'],axis=1))
        X = sklearn.preprocessing.scale(X)
        predict_data = X[-30:]
        X = X[:-30]
        y = np.array(df_stock['label'])
        y = y[:-30]

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X,y, test_size= 0.2)

        model = sklearn.linear_model.LinearRegression()
        model.fit(X_train,y_train)

        accuracy = model.score(X_test,y_test)

        st.write(f'The percentage of hit is{round(accuracy) * 100,1}% .')

        if accuracy > 0.75:
            st.write('Trustworthy:High')
        elif accuracy > 0.5:
            st.write('Trustworthy:Moderate')
        else:
            st.write('Trustworthy:Low')

        st.write('The predict price is Orange line.')

        predicted_data = model.predict(predict_data)
        df_stock['Predict']= np.nan
        last_date = df_stock.iloc[-1].name
        one_day= 86400
        next_unix = last_date.timestamp()+ one_day

        for data in predicted_data:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df_stock.loc[next_date] = np.append([np.nan] * (len(df_stock.columns)-1),data)
        
        df_stock['Close'].plot(figsize=(15, 6), color='green')
        df_stock['Predict'].plot(figsize=(15, 6), color='orange')

        df_stock3 = df_stock[['Close','Predict']]
        st.line_chart(df_stock3)

    if st.button('Predict'):
        stock_predict()

except:
    st.error(
        'There are some errors'
    )

st.write('copyright &copy; 2021 Tomoyuki Yoshikawa. All Rights Reserved.')

