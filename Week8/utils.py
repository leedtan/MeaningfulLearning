
def viz(df):
    # Let's visualize some of our data.

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