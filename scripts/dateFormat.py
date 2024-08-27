import pandas as pd
def formatDate(data, stock=False):
    date = 'Date'if stock else 'date'
    data = pd.to_datetime(data[date],format='ISO8601')
    return data