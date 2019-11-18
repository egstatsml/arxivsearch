import pandas as pd


def read_clean_csv(path, sep='|'):
  """will read and clean a csv file'"""
  df = pd.read_csv(path, sep=sep)
  if('Unnamed: 0' in df.columns.values):
    df = remove_unnamed(df)
  return(df)

def remove_unnamed(df):
  df = df.drop('Unnamed: 0', 1)
  return df
