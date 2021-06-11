import numpy as np

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
    

def dataShift(df, numeric_cols):
    df['Date'] = df.index
    for days_past in range(1, 7):
        df['prev_date' + str(days_past)] = df['Date'] - days_past#pd.DateOffset(days=days_past)

    for days_past in range(1, 7):
        df_past = df[numeric_cols + ['Date']].set_index('Date')
        df_past.columns = [c + '_past' + str(days_past) for c in df_past.columns]
        df = df.set_index('prev_date' + str(days_past)).join(df_past).reset_index(drop=True)
    df = df[[c for c in df.columns if 'prev_date' not in c]]
    df = df.dropna()
    return df

def get_x_y(numeric_cols, cleaner):
    past_cols = [[c + '_past' + str(days_past) for c in numeric_cols] for days_past in range(1, 7)]
    past_cols = [c for cols in past_cols for c in cols]
    x = cleaner[past_cols]
    y = cleaner[numeric_cols]
    return x, y, past_cols

def get_mdl_inputs(df, numeric_cols, past_cols):
    y_diff = ((df['High'] + df['Low']).iloc[1:].values/2 - df['Close'].iloc[:-1].values)/df['Close'].iloc[:-1].values
    x_with_current = df[numeric_cols + past_cols].iloc[:-1]
    return y_diff, x_with_current

def calc_y(df):
    y_future = df[['High', 'Low']].mean(1).iloc[1:].values
    y_past = df['Close'].iloc[:-1].values
    y_diff = (y_future - y_past)/y_past
    return y_diff

def get_trn_val(x, numeric_cols, trn_percent = 0.8, val_percent = 0.2):
    n_date = x.shape[0]
    num_trn = np.floor(n_date * trn_percent).astype(int)
    num_val = np.floor(n_date * val_percent).astype(int)
    x_trn, x_val = x.iloc[:num_trn+1], x.iloc[num_trn:num_trn+num_val]
    y_trn, y_val = [calc_y(x_set) for x_set in (x_trn, x_val)]
    mean_vals = x_trn[numeric_cols].mean().mean()
    x_trn = x_trn / mean_vals
    x_val = x_val / mean_vals
    x_trn, x_val = x_trn[:-1], x_val[:-1]
    x_trn, x_val = x_trn.values, x_val.values
    return x_trn, y_trn, x_val, y_val