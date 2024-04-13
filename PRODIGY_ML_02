
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans


data = pd.read_csv(r"C:\Users\Dell\Downloads\ProdigyMaterial\Task2-K-means-clustering\features.csv")
data.head()
data['Gender'] = data['Gender'].replace({'Female': 1, 'Male': 2}) #numeric values wanted

numeric_col = data.select_dtypes(include=['float64', 'int64']).columns
print (data.head(200))

wcss = [] # Within-cluster sum of squares

for k in range (1, 201): # 1 to 200 as #customers = 200
    k_means = KMeans (n_clusters=k, random_state=0)
    k_means.fit(data)
    wcss.append(k_means.inertia_)

plt.plot(range(1, 201), wcss) # elbow curve to check the number of clusters required 
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=10, random_state=0)
cluster_labels = kmeans.fit_predict(data)


data['Cluster'] = cluster_labels

plt.scatter(data['CustomerID'], data['Spending Score (1-100)'], c=data['Cluster'])
plt.xlabel('CustomerID')
plt.ylabel('Spending Score (1-100)')
plt.show()
