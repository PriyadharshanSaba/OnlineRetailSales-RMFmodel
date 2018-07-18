#Cluster the sample data
#RMF analysis has been used
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
import os

script_dir = os.path.dirname(__file__)  # Script directory
full_path = os.path.join(script_dir, '../DATA_SET/sample_data.csv')
df = pd.read_csv(full_path)


#----CLUSTERING----

#Outliers in Monetary Amount
amountRange = sample_df.as_matrix(columns=['Amount'])

freqRange=sample_df.as_matrix(columns=['freq_val'])
plt.scatter(amountRange.tolist(),freqRange.tolist(),8)
plt.xlabel('Monetary Amount')
plt.ylabel('Frequency')

recRange = sample_df.as_matrix(columns=['freq_val'])

recRange=sample_df.as_matrix(columns=['Recency_val'])
plt.scatter(amountRange.tolist(),recRange.tolist(),8)
plt.xlabel('Monetary Amount')
plt.ylabel('Recency')

#Standard Deviation approach
x = np.array(amountRange)
xmean = np.mean(x,axis=0)
xsd = np.std(x,axis=0)
sample_df.shape
#amountRange = [ i for i in range(0,sample_df.shape[0]) if(sample_df.iloc[i]['Amount'] > xmean-2*xsd)]
#amountRange = [ y for y in amountRange if(y['Amount'] <xmean+2*xsd)]

sample_df = sample_df[sample_df['Amount'] >= xmean[0]-2*xsd[0]]
sample_df = sample_df[sample_df['Amount'] <= xmean[0]+2*xsd[0]]
sample_df.head()

amountRange = sample_df.as_matrix(columns=['Amount'])
recRange=sample_df.as_matrix(columns=['Recency_val'])
plt.scatter(amountRange.tolist(),recRange.tolist(),8)
plt.xlabel('Monetary Amount')
plt.ylabel('Recency')

#kmeans = KMeans(n_clusters=4)
#x = sample_df.as_matrix()
#kmeans.fit(x)
#print(kmeans.cluster_centers_)

#print(kmeans.labels_)
#plt.scatter(x[:,0],x[:,1], c=kmeans.labels_, cmap='rainbow')
sample_df.head()

x = sample_df[['freq_val']].as_matrix()
y = sample_df[['Amount']].as_matrix()
z_df = sample_df[['freq_val','Amount','Recency_val']]
z=z_df.as_matrix()

nc = range(2, 20)
kmeans = [KMeans(n_clusters=i) for i in nc]
kmeans
score = [kmeans[i].fit(z).score(z) for i in range(len(kmeans))]

#elbow point
pl.plot(nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()

kmeans=KMeans(n_clusters=6)
kmeansoutput=kmeans.fit(z)
#kmeansoutput

"""pca = PCA(n_components=1).fit(y)
    pca_y = pca.transform(y)
    pca_x = pca.transform(x)"""

pl.scatter(x[:, 0],y[:, 0], c=kmeansoutput.labels_)
pl.show()

C = kmeans.cluster_centers_
print(C)
labels = kmeans.labels_

print(sample_df.head())

#----fetching customers---

color=["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059","#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87","#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80","#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100","#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F","#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09","#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66","#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C","#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81","#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00","#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700","#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329","#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C","#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800","#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51","#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58","#000000"]

fig = plt.figure()
ax = Axes3D(fig)
for i in range(len(z)):
    ax.scatter(z[i][0], z[i][1],z[i][2],c=color[labels[i]])
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
ax.set_xlabel('R')
ax.set_ylabel('F')
ax.set_zlabel('Monetary Amount')

for angle in range(0, 360):
    ax.view_init(45,angle)
    plt.draw()
    plt.pause(.1)
plt.close()

C = kmeans.cluster_centers_
print(C)

sample_df.head()

print(set(kmeansoutput.labels_)  )

print('Preparing Mailing list...')
l = kmeans.predict(z)
lab_df = l

sample_df['cLabel']=lab_df

mailList = sample_df[sample_df['cLabel'] == 5]

mailList['CustomerID'].to_csv('mailingList.csv')
print(mailList['CustomerID'])

