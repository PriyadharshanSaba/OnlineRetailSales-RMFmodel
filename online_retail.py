import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns

!pip install xlrd

import xlrd

df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx")
pdf = df
#df.head()

#DATA WRANGLING

df['Amount'] = df.UnitPrice * df.Quantity
#df.head()

df['date'] = [d.date() for d in df['InvoiceDate']]
df['time'] = [d.time() for d in df['InvoiceDate']]
#print(df.head())

#CLEANING CUSTOMERS

#dropping NaN values in CustomerID
df = df[np.isfinite(df['CustomerID'])]      #df.loc[df['InvoiceNo']==573174]  reference for NaN value

#removing customers with less than 1Re of transaction
df=df[(df['Amount'] >=1)]
df.head()

#grouping by countries and adding purchasing percentage on basis of quantity
group_country = df.groupby(['Country']).sum()
group_country = group_country.drop(columns=['UnitPrice','CustomerID'])
group_country.sort_values('Quantity',ascending=False,inplace=True)

total_purchased = group_country['Quantity'].sum()
group_country['Buy_perc']=(group_country['Quantity']/total_purchased)*100
group_country.head()

#Buy_Perc plotting

country=list(group_country.index.values)
Cust_id=list(group_country['Buy_perc'])
plt.figure(figsize=(12,8))
sns.barplot(country, Cust_id, alpha=1)
plt.xticks(rotation='60')
plt.show()

x=group_country.where(group_country['Buy_perc']==group_country['Buy_perc'].max())
x=x.dropna().index.tolist()[0]

#analysing country with highest purchasing 
dfcur = df.where(df.Country == x)

#each_cust = dfcur.groupby(['CustomerID']).sum()

#first time buyers
x=dfcur.groupby('CustomerID').count()
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

