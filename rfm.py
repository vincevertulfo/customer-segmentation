import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from clustering_utilities import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

class RFM:

    def __init__(self,df, rfm_column_names, reference_date="DAYAFTER"):

        '''
        Returns a formatted RFM table ready for aggregations

        Parameters:
            df (df): The raw user transaction dataframe.
            cols (dict) : {"customer_id": colname, "invoice_date": colname, "invoice_id": colname, "cost": colname}
            ## TODO: add reference date format
            reference_date (date): The date you want to reference from for the RFM metrics.
                                   You can put a specific reference date in this string format "YYYY/MM/DD"
                                   or use "DAYAFTER" which stands for day after last invoice date
            tenure (str): Name of the tenure column if it exists

        Returns:
            DataFrame
        '''

        self.df = df

        customer_id = rfm_column_names["customer_id"]
        invoice_date = rfm_column_names["invoice_date"]
        invoice_id = rfm_column_names["invoice_id"]
        cost = rfm_column_names["cost"]

        self.df[invoice_date]=pd.to_datetime(self.df[invoice_date])


        if reference_date == "DAYAFTER":
            reference_date = self.df[invoice_date].max() + timedelta(days=1)
        else:
            reference_date = datetime.strptime(reference_date, "%Y/%M/%d")
        
        self.df = self.df[self.df[invoice_date]<reference_date]

        rfm_df = self.df.groupby(customer_id).agg({
            invoice_date : lambda x: (reference_date - x.max()).days,
            invoice_id : 'count',
            cost : 'sum'
            }).rename(columns={
            invoice_date : 'Recency',
            invoice_id : 'Frequency',
            cost : 'Monetary'
            })

        self.rfm_df =rfm_df

    def dist_plot_rfm(self):
        '''
        Returns Distribution plot for each RFM metrics

        Parameters:
            rfm_df (df): RFM DataFrame

        Returns:
            Dist plot for RFM metrics
        '''    
        sns.distplot(self.rfm_df['Recency'])
        plt.show()
        sns.distplot(self.rfm_df['Frequency'])
        plt.show()
        sns.distplot(self.rfm_df['Monetary'])
        plt.show()

    def rfm_scores(self):
        '''
            Returns RFM table with RFM Score columns

            Returns:
                DataFrame with additional columns for rfm scores
        '''
        rfm_score_df = self.rfm_df
        rfm_score_df['r_score'] = pd.qcut(rfm_score_df['Recency'], 4, ['1', '2', '3', '4'])
        rfm_score_df['f_score'] = pd.qcut(rfm_score_df['Frequency'], 4, ['4', '3', '2', '1'])
        rfm_score_df['m_score'] = pd.qcut(rfm_score_df['Monetary'], 4, ['4', '3', '2', '1'])

        rfm_score_df['rfm_total'] = rfm_score_df['r_score'].astype(str) + rfm_score_df['f_score'].astype(str) + rfm_score_df['m_score'].astype(str)
        rfm_score_df['rfm_sum'] = rfm_score_df['r_score'].astype(int) + rfm_score_df['f_score'].astype(int) + rfm_score_df['m_score'].astype(int)

        return rfm_score_df

    def rfm_kmeans_clustering(self, scale = True, no_of_cluster= "OPTIMIZE"):
        '''
            Returns RFM table with its cluster

            Returns:
                DataFrame with additional columns for clusters
        '''
        rfm_df=self.rfm_df

        if scale:
            features_rfm_df = StandardScaler().fit_transform(rfm_df.loc[:, ['Recency', 'Frequency', 'Monetary']])
        else:
            features_rfm_df = rfm_df.loc[:, ['Recency', 'Frequency', 'Monetary']]

        if no_of_cluster == "OPTIMIZE":
            k = find_optimal_k(KMeans(), 12, features_rfm_df)
        else:
            k = no_of_cluster

        k_means = KMeans(n_clusters=k)
        model = k_means.fit(features_rfm_df)
        y_hat = k_means.predict(features_rfm_df)
        labels = k_means.labels_

        rfm_df['clusters'] = labels
        clustered_rfm_df = rfm_df

        self.clustered_rfm_df = clustered_rfm_df

        return clustered_rfm_df

    def rfm_cluster_summary(self):
        '''
        Returns RFM table that is grouped by according to col showing the mean/average of the RFM metrics

        Parameters:
            rfm_df (df): RFM DataFrame
            col (string): Column to use for groupby. Ideally, this is the cluster name column

        Returns:
            DataFrame grouped by cluster name showing th mean/average of the RFM metrics
        '''  

        cluster_summary = self.clustered_rfm_df.groupby("clusters").agg({
            'Recency' : 'mean',
            'Frequency' : 'mean',
            'Monetary' : ['mean', 'count']
        })

        return cluster_summary
