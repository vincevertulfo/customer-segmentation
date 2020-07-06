import pandas as pd 
import numpy as np 
from rfm import RFM

dataset_link= "dataset/Online Retail.csv"
df=pd.read_csv(dataset_link)
df["TotalCost"]= df["Quantity"]*df["UnitPrice"]


rfm_col_names={"customer_id": "CustomerID", 
               "invoice_date": "InvoiceDate", 
               "invoice_id": "InvoiceNo", 
               "cost":"TotalCost"}



rfm_table=RFM(df,rfm_col_names, reference_date="2020/03/21")

rfm_table.dist_plot_rfm()
x=rfm_table.rfm_kmeans_clustering(scale = True, no_of_cluster= "OPTIMIZE")
print(rfm_table.rfm_cluster_summary())