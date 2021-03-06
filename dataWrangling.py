#Clean the data and generate sample data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlrd
from sklearn.cluster import KMeans
import pylab as pl
import sys

df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx")
print("Frames\n",df.head())

#DATA WRANGLING

#dropping NaN data rows
df = df.dropna()
#dropping description column
df=df.drop(columns=['Description'])
df['Amount'] = df.UnitPrice * df.Quantity
#removing unit price
df=df.drop(columns=['UnitPrice'])

df['date'] = [d.date() for d in df['InvoiceDate']]
df['time'] = [d.time() for d in df['InvoiceDate']]

#CLEANING CUSTOMERS
print("\n\nCleaning")

#dropping NaN values in CustomerID
df = df[np.isfinite(df['CustomerID'])]      #df.loc[df['InvoiceNo']==573174]  reference for NaN value

#removing customers with less than 1Re of transaction
df=df[(df['Amount'] >=1)]

#grouping by countries and adding purchasing percentage on basis of quantity
group_country = df.groupby(['Country'],as_index=False).sum()
group_country = group_country.drop(columns=['CustomerID'])
group_country.sort_values('Quantity',ascending=False,inplace=True)

total_purchased = group_country['Quantity'].sum()
group_country['Buy_perc']=(group_country['Quantity']/total_purchased)*100


print("\n\nBuyers % Plotting")
#Buy_Perc plotting

country=list(group_country['Country'])
Cust_id=list(group_country['Buy_perc'])
plt.figure(figsize=(12,8))
sns.barplot(country, Cust_id, alpha=1)
plt.xticks(rotation='60')
plt.show()


#analysing country with highest purchasing
x=group_country.where(group_country['Buy_perc']==group_country['Buy_perc'].max())
x=x['Country'].dropna().get_values().tolist()[0]

dfcur = df.where(df.Country == x)
dfcur=dfcur.dropna()

#first time buyers
x=dfcur.groupby('CustomerID',).count()
no_customers= x['InvoiceNo'].count()
no_first_timers=x.where(x['InvoiceNo']==1).dropna().count()['InvoiceNo']
print('Percentage of new customers purchasing for the first time',(no_first_timers*100/no_customers))

objects = ('First/New Customers', 'Existing')
y_pos = np.arange(len(objects))
perc = [no_first_timers,no_customers]
plt.bar(y_pos, perc, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Percentage')
plt.title('Customers')

plt.show()

#Converting to date to shortened string formate %y(2)%m
dfcur['dateS']=dfcur['date'].apply(lambda x: x.strftime('%y%m'))
#print('Far date: ',dfcur['dateS'].min(),"\tRecent: ",dfcur['dateS'].max())


#RMF MODEL
print("\n\nRecency Analysis")
#---- Recency Analysis------
def recency(row):
    if int(row['dateS']) > 1109:
        val = 5
            elif int(row['dateS']) <= 1109 and int(row['dateS']) > 1106:
                val = 4
                    elif int(row['dateS']) <= 1106 and int(row['dateS']) > 1103:
                        val = 3
                            elif int(row['dateS']) <= 1103 and int(row['dateS']) > 1101:
                                val = 2
                                    else:
val = 1
    return val

dfcur['Recency_val'] = dfcur.apply(recency, axis=1)

dfcur.head()

#table with just recency values
rec_df = dfcur
rec_df = rec_df.drop(columns=['Quantity','InvoiceNo','StockCode','InvoiceDate','Country','Amount','date','time','dateS'])
recencyTable = rec_df.drop_duplicates( keep=False)
rec_df.head()

plt.figure(figsize=(12,8))
sns.countplot(x="Recency_val", data=rec_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Recency_Value', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Recency_Flag", fontsize=15)
plt.show()

print("\tRecency Analysis\n",rec_df.groupby('Recency_val',as_index=False).count())

print("\n\nFrequency Analysis")
#------Frequency Analysis-------

freq_df = dfcur[['Country','InvoiceNo','CustomerID']].drop_duplicates()
freq_count = freq_df.groupby(['Country','CustomerID'],as_index=False)[['InvoiceNo']].count()
freq_count.head()

unique_invoice=freq_df[['InvoiceNo']].drop_duplicates()
unique_invoice.head()

#Dividing the dataframe into 5 bands
unique_invoice['fband']=pd.qcut(unique_invoice['InvoiceNo'],5)
unique_invoice.head()
freqBandTable = unique_invoice[['fband']].drop_duplicates().reset_index()
freqBandTable

def frequ(row):
    if row['InvoiceNo'] <= 13:
        val = 1
            elif row['InvoiceNo'] > 13 and row['InvoiceNo']<=24:
                val =2
                    elif row['InvoiceNo']>24 and row['InvoiceNo']<=35:
                        val = 3
                            elif row['InvoiceNo']>35 and row['InvoiceNo']<=60:
                                val = 4
                                    else:
val = 5
    return val

freq_count['freq_val'] = freq_count.apply(frequ,axis=1)
freq_count.head()

plt.figure(figsize=(12,8))
sns.countplot(x="freq_val", data=freq_count)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Frequency_Value', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency Analysis", fontsize=15)
plt.show()

print("\tFrequency Analysis\n",freq_count.groupby('freq_val',as_index=False).count())

print("\n\nMoentary Analysis")
#----MonetaryValue Analysis------

#monetary value for each country
monetary_df_countries = df.groupby(['Country','CustomerID'],as_index=False)['Amount'].sum()
monetary_df_countries.head()

ctry=list(monetary_df_countries['Country'])
amt=list(monetary_df_countries['Amount'])
plt.figure(figsize=(12,8))
sns.barplot(ctry, amt, alpha=1)
plt.xticks(rotation='60')
plt.show()

print("\tMonetary Analysis Countrywise\n",monetary_df_countries.groupby('Country',as_index=False)[['Amount']].sum().sort_values('Amount', ascending=False).reset_index(drop=True).head())

#getting monetary band
monprice_df = dfcur[['CustomerID','Amount']].drop_duplicates()
monprice_df = monprice_df.groupby(['CustomerID'],as_index=False)[['Amount']].sum()
monprice_df['monetary'] = pd.qcut(monprice_df['Amount'],5)
monprice_df=monprice_df.sort_values(by=['monetary'])
monetary_band = monprice_df[['monetary']].drop_duplicates().reset_index(drop=True)
monetary_band

x=monetary_band['monetary'].get_values().tolist()
x

def mon(row):
    if row['Amount'] <= x[0].right:
        val = 1
            elif row['Amount']> x[0].right and row['Amount']<=x[1].right:
                val = 2
                    elif row['Amount']> x[1].right and row['Amount']<=x[2].right:
                        val = 3
                            elif row['Amount']> x[2].right and row['Amount']<=x[3].right:
                                val = 4
                                    else:
val = 5
    return val

monprice_df['Mont_val'] = monprice_df.apply(mon,axis=1)

monprice_df.head()

plt.figure(figsize=(12,8))
sns.countplot(x="Mont_val", data=monprice_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Monetary_Value', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Monetary Analysis", fontsize=15)
plt.show()

print("\tMonetary Analysis\n",monprice_df.groupby(['Mont_val'],as_index=False)['monetary'].count())

#fetching the sample with RMF values
uk_df=pd.merge(rec_df,freq_count)
uk_df=pd.merge(uk_df,monprice_df)
sample_df=uk_df
sample_df=sample_df.drop_duplicates().reset_index(drop=True)
sample_df = sample_df.drop(columns=['Country','InvoiceNo','monetary'])

sample_df.head()
sample_df.to_csv('sample_data.csv')
