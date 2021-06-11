numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
from sklearn.linear_model import Ridge
def picture(df):
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df.index[:100],
                    open=df['Open'][:100],
                    high=df['High'][:100],
                    low=df['Low'][:100],
                    close=df['Close'][:100])])
    fig.update_layout(
        title= {
            'text': '',
          'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
          font=dict(
            family="Courier New, monospace",
            size=20,
            color="#7f7f7f"
        )
        )
    fig.show()
def data_points(df):
    df['Date'] = df.index
    df = df[[c for c in df.columns if 'prev_date' not in c]]
    df_all_dates = df
    for i in range (1 , 20):
        df_past = df.copy()
        df_past['Date_past'+ str(i)] = df_past['Date'] + i
        df_past = df_past.set_index('Date_past' + str(i))
        df_past.columns = [c + '_past' + str(i) for c in df_past.columns]
        df_all_dates = df_all_dates.set_index('Date').join(df_past).reset_index()
    return df_all_dates.dropna()

# def get_xy(df):
#     df = data_points(df)
#     past_cols = [[c + '_past' + str(days_past) for c in numeric_cols] for days_past in range(1, 14)]
#     past_cols = [c for cols in past_cols for c in cols]
#     current_cols = numeric_cols
#     x = df[past_cols]
#     y = df[current_cols]
#     return x, y

def get_xandychange(df,weeks_past = 7):
    current_cols = numeric_cols
    past_cols = [[c + '_past' + str(days_past) for c in numeric_cols] for days_past in range(1, 14)]
    past_cols = [c for cols in past_cols for c in cols]
    y_diff = ((df['High'] + df['Low']).iloc[1:].values/2 - df['Close'].iloc[:-1].values)/df['Close'].iloc[:-1].values
    x_with_current = df[current_cols + past_cols].iloc[:-1]
    return x_with_current, y_diff
#     mdl = Ridge().fit(x_with_current, y_diff)
#     yhat = mdl.predict(x_with_current)
def training_values(x, y, percent_train, percent_val):
    num_train = round(x.shape[0] * percent_train)
    num_val = round(x.shape[0] * percent_val)
    x_trn, y_trn = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:], y[num_train:]
    return x_trn, y_trn, x_val, y_val
def predict(x, y):
    mdl= Ridge().fit(x_trn,y_trn)
    yhat = mdl.predict(x_trn)
    return yhat
def validation(mdl,x_val,y_val):
    money = 100
    predictions = mdl.predict(x_val)
    assignment = predictions > predictions.mean()
    for i, y in zip (assignment, y_val):
        money = money * (1 - i) + i * money * (1 + y)
    return money