import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import seaborn as sns

import squarify #for visualizing treemap layout
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer


def create_rfm_table(df, cols=None, checker_date=None, tenure=None):
  '''
  Returns a formatted RFM table ready for aggregations

  Parameters:
      df (df): The raw user transaction dataframe.
      cols (list) : List containing 4 columns to be used for transformation. List should contain [customer_id, invoice_date, invoice_id, cost]
      checker_date (date): The date you want to reference from for the RFM metrics
      tenure (str): Name of the tenure column if it exists

  Returns:
      DataFrame
  '''
  customer_id = cols[0]
  invoice_date = cols[1]
  invoice_id = cols[2]
  cost = cols[3]

  if checker_date:
    checker_date = checker_date
  else:
    checker_date = df[invoice_date].max() + timedelta(days=1)

  if tenure:
    rfm_df = df.groupby(customer_id).agg({
      invoice_date : lambda x: (checker_date - x.max()).days,
      invoice_id : 'count',
      cost : 'sum',
      tenure: 'mean'
    }).rename(columns={
      invoice_date : 'Recency',
      invoice_id : 'Frequency',
      cost : 'Monetary',
      tenure: 'Tenure'
    })
  else:
    rfm_df = df.groupby(customer_id).agg({
      invoice_date : lambda x: (checker_date - x.max()).days,
      invoice_id : 'count',
      cost : 'sum'
    }).rename(columns={
      invoice_date : 'Recency',
      invoice_id : 'Frequency',
      cost : 'Monetary'
    })

  return rfm_df

def apply_rfm_score(rfm_df):
  '''
  Returns RFM table with RFM Score columns

  Parameters:
      rfm_df (df): RFM DataFrame

  Returns:
      DataFrame with additional columns for rfm scores
  '''

  rfm_df['r_score'] = pd.qcut(rfm_df['Recency'], 4, ['1', '2', '3', '4'])
  rfm_df['f_score'] = pd.qcut(rfm_df['Frequency'], 4, ['4', '3', '2', '1'])
  rfm_df['m_score'] = pd.qcut(rfm_df['Monetary'], 4, ['4', '3', '2', '1'])

  rfm_df['rfm_total'] = rfm_df['r_score'].astype(str) + rfm_df['f_score'].astype(str) + rfm_df['m_score'].astype(str)
  rfm_df['rfm_sum'] = rfm_df['r_score'].astype(int) + rfm_df['f_score'].astype(int) + rfm_df['m_score'].astype(int)

  return rfm_df

def cluster_df(rfm_df):
  '''
  Returns RFM table with its cluster

  Parameters:
      rmf_df (df): RFM DataFrame

  Returns:
      DataFrame with additional columns for clusters
  '''
  if 'Tenure' in rfm_df.columns:
    X = StandardScaler().fit_transform(rfm_df.loc[:, ['Recency', 'Frequency', 'Monetary', 'Tenure']])
  else:
    X = StandardScaler().fit_transform(rfm_df.loc[:, ['Recency', 'Frequency', 'Monetary']])

  # NOTE: Find optimal no. of k
  k = find_optimal_k(KMeans(), 12, X)

  k_means = KMeans(n_clusters=k)
  model = k_means.fit(X)
  y_hat = k_means.predict(X)
  labels = k_means.labels_

  rfm_df['clusters'] = labels

  return rfm_df


def find_optimal_k(model, k_fold, X):
  '''
  Returns optimal no. of k clusters

  Parameters:
      model (object): Algorithm object (e.g. KMeans)
      k_fold (int): Check up to how many k

  Returns:
     The optimal no. of k clusters
  '''
  visualizer = KElbowVisualizer(model, k=(2,k_fold))
  visualizer.fit(X)

  return visualizer.elbow_value_



def describe_cluster(rfm_df, col):
  '''
  Returns RFM table that is grouped by according to col showing the mean/average of the RFM metrics

  Parameters:
      rfm_df (df): RFM DataFrame
      col (string): Column to use for groupby. Ideally, this is the cluster name column

  Returns:
      DataFrame grouped by cluster name showing th mean/average of the RFM metrics
  '''  

  if 'Tenure' in rfm_df.columns: 
    cluster_df = rfm_df.groupby(col).agg({
      'Recency' : 'mean',
      'Frequency' : 'mean',
      'Monetary' : 'mean',
      'Tenure': ['mean', 'count']
    })
  else:
    cluster_df = rfm_df.groupby(col).agg({
      'Recency' : 'mean',
      'Frequency' : 'mean',
      'Monetary' : ['mean', 'count']
    })

  return cluster_df


def dist_plot_rfm(rfm_df):
  '''
  Returns Distribution plot for each RFM metrics

  Parameters:
      rfm_df (df): RFM DataFrame

  Returns:
      Dist plot for RFM metrics
  '''    
  sns.distplot(rfm_df['Recency'])
  plt.show()
  sns.distplot(rfm_df['Frequency'])
  plt.show()
  sns.distplot(rfm_df['Monetary'])
  plt.show()



