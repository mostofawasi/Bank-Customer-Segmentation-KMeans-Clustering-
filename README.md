# **Bank Churn Customer Analysis**: Customer Segmentation
|| **Segmenting Bank Customers and Recommend Potential New Products or Services for each Segment** ||
## **Objective 1: Preparing the Data for Modeling**

Our First Objective is to Prepare the Data for Modeling by Selecting a Subset of Fields, making sure they are Numeric, looking at their Distributions, and Engineering a new feature.
# Importing the Required Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r"I:\Python\Projects\Bank Churned Segmentation\Bank_Churn.csv")
data.head()
data.info()
### **Let's keep a subset of the Data**

Create a DataFrame containing all fields except "CustomerId", "Surname" and "Exited".
data_subset = data [['CreditScore', 'Geography', 'Gender',
                     'Age', 'Tenure', 'Balance', 'NumOfProducts',
                     'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]

data_subset.head()
### **Making Text Fields Numeric**
data_clean = data_subset.copy()
data_clean.head()
data_clean.Geography.value_counts()
data_clean.Gender.value_counts()
data_clean.Gender = np.where(data_clean.Gender == 'Female', 1, 0)
data_clean.head()
data_clean = pd.get_dummies(data_clean, columns = ['Geography'], dtype = 'int', prefix = '', prefix_sep = '')
data_clean.head()
### **Exploring the Data**

Exploring the Data by looking at the Min/Max values and the Distribution of each Column.
data_clean.describe().round()
sns.pairplot(data_clean, corner = True);
### **Let's Engineer a New Feature**

Engineer a new feature called "ProductsPerYear".
data_clean['ProductsPerYear'] = np.where(data_clean.Tenure == 0, data_clean.NumOfProducts, data_clean.NumOfProducts / data_clean.Tenure)
data_clean.head()
data_clean.describe().round()
## **Objective 2: Cluster the Customers (Round 1)**

Our Second Objective is to Segment the Customers using K-Means Clustering, including Standardizing the Data, Creating an Inertia Plot, and Interpreting the Clusters.
### **Scale the Data using Stardardization**

Standardize the data so that each column has a Mean of 0 and Standard Deviation of 1.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(data_clean), columns = data_clean.columns)
df_scaled.head()
df_scaled.describe().round()
### **Fit K-Means Models with 2-15 Clusters**

Fit K-Means Clustering Models on the Standardized Data with 2-15 Clusters to create an Inertia Plot.
# Import kmeans and write a loop to fit models with 2 to 15 clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create an empty list to hold many Inertia and Silhouette Values
inertia_values = []
silhouette_scores = []

# Creat 2-15 clusters, and add the Intertia Scores and Silhouette Scores to the Lists

for k in range(2, 16):
  kmeans = KMeans(n_clusters = k, n_init= 10, random_state = 42) #changed from auto to 10
  kmeans.fit(df_scaled)
  inertia_values.append(kmeans.inertia_)
  silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_, metric= 'euclidean', sample_size= None))
## Plot the Inertia Plot
# Turn the List into a Series of Plotting
inertia_series = pd.Series(inertia_values, index = range(2, 16))

# Plot the Data
inertia_series.plot(marker= 'o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Number of Clusters vs. Inertia')
plt.show()
So, looking at this we need to find the Elbow of this Inertia Plot.

The Inertia drops significantly at K=5. So, we choose k=5 for our KMeans Clustering.
### **Plot the Intertia Values and Find the Elbow**

Identify the Elbow of the Inertia Plot and Fit a K-Means Model using that value of k.
kmeans5 = KMeans(n_clusters= 5, n_init= 10, random_state= 42)
kmeans5.fit(df_scaled)
### **Check the Number of Customers in each Cluster**
from collections import Counter

Counter(kmeans5.labels_)
### **Create a Heat Map of the Cluster Centers and Interprete the Cluster**

cluster_centers5 = pd.DataFrame(kmeans5.cluster_centers_, columns = df_scaled.columns)

plt.figure(figsize = (10, 2))
sns.heatmap(cluster_centers5, annot = True, cmap="RdBu", fmt=".1f", linewidths= 0.5);
* 0: Many Products in a Short Time
* 1: French Customers with Few Products and High Balance
* 2: German Customers with a High Balance
* 3:French Customers with More Products and Low Balance
* 4: Spanish Customers
## **Objective 3: Cluster the Customers (Round 2)**

Our Third Objective is to Segment the Customers using K-Means Clustering using a different Subset of Fields and Compare the Model Results.
### **Updating the Model Dataset**

We will look at the Summary Stats by Country, and Exclude the Country Field as currently the clusters are dominated by the Country Field. Then we use the updated dataset for KMeans Clustering same way as the previous steps.

data_subset.head()
data_geo = data_subset.copy()
data_geo.Gender = np.where(data_geo.Gender == 'Female', 1, 0)
data_geo.head()
data_geo.groupby('Geography').mean().round()
data_geo[data_geo.Geography == 'France'].Balance.round(-5).value_counts()
data_geo[data_geo.Geography == 'Spain'].Balance.round(-5).value_counts()
data_geo[data_geo.Geography == 'Germany'].Balance.round(-5).value_counts()
**Customers from French, Spain and Germany have pretty similar attributes except in Balance. German customoers have higher Balance than the other two countries.**
So, we will exclude the Country Field.
df_scaled.head()
df_scaled_no_geo = df_scaled.drop(columns = ['France', 'Germany', 'Spain'])
df_scaled_no_geo.head()
### **Now, fit the Clustering Model with the Updated Data**
# Import kmeans and write a loop to fit models with 2 to 15 clusters
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create an empty list to hold many Inertia and Silhouette Values
inertia_values = []
silhouette_scores = []

# Creat 2-15 clusters, and add the Intertia Scores and Silhouette Scores to the Lists

for k in range(2, 16):
  kmeans = KMeans(n_clusters = k, n_init= 10, random_state = 42) #changed from auto to 10
  kmeans.fit(df_scaled_no_geo)
  inertia_values.append(kmeans.inertia_)
  silhouette_scores.append(silhouette_score(df_scaled_no_geo, kmeans.labels_, metric= 'euclidean', sample_size= None))
## Plot the Inertia Plot
# Turn the List into a Series of Plotting
inertia_series = pd.Series(inertia_values, index = range(2, 16))

# Plot the Data
inertia_series.plot(marker= 'o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Number of Clusters vs. Inertia')
plt.show()
The Inertia drops significantly at K=4. So, we choose k=5 for our KMeans Clustering.
kmeans4 = KMeans(n_clusters= 4, n_init= 10, random_state= 42)
kmeans4.fit(df_scaled_no_geo)
Counter(kmeans4.labels_)
### **Create the Heatmap again to Find the Cluster Centers**
cluster_centers4 = pd.DataFrame(kmeans4.cluster_centers_, columns = df_scaled_no_geo.columns)

plt.figure(figsize = (10, 2))
sns.heatmap(cluster_centers4, annot = True, cmap="RdBu", fmt=".1f", linewidths= 0.5);
* 0: Customers who don't have a Credit Card, but with a relatively high Tenure.
* 1: High Balance with Few Products and Have Credit Card.
* 2: Low Balance, More Products and Have Credit Card.
* 3: Customers with Many Products in a very Short period of Time.
## **Objective 4: Explore the Clusters and Make Recommendations**

Our Final Objective is to further Explore Our K-Means Clusters by looking at their Churn Rate and Country Breakdown, then make Recommendations for how to Cater to each Customer Segment.
###**Let's create a DataFrame that combines the data set from the end of Objective 1, the "Exited" field, and the cluster labels.**
data_clean.head()
data.Exited.head()
kmeans4.labels_
**Let's merge those together:**
data_final = pd.concat([data_clean, data.Exited, pd.Series(kmeans4.labels_, name = 'Cluster')], axis = 1)
data_final.head()
### **View the Exited Percent for Each Cluster**
data_final.groupby('Cluster').mean().round(2)
**So, cluster 1 has the highest churn rate (23%) and cluster 2 has the lowest churn rate (16%)**
### **View the Geography Breakdown for Each Cluster**
For Cluster 2 there are many French Customers and very few German Customers.
### **Making Recommendations**

We have to make recommendations for how to Cater to each Customer Segments.
**These are the characteristics of the Clusters we found earlier:**

* 0: Customers who don't have a Credit Card.
* 1: High Balance with Few Products and Have Credit Card.
* 2: Low Balance, More Products and Have Credit Card.
* 3: Customers with Many Products in a very Short period of Time.
***Recommendations:***

*   **Clsuter 0:** Create an Entry-level Credit Card; Also, do some research on their Demographic Information.

*   **Cluster 1:** They have very high Balance, but likely to leave. So,  try to keep them by offering them financial seminars or advisors or, maybe suggest investment opportunities.

* **Cluster 2:** They are less likely to leave. We can reward them (French and Spanish Customers) for staying. We can introduce reward programs to encourage them to invest in more products.

* **Cluster 3:** They have many products in a very short time. Churn rate is also relatively high. Since, they like products we may offer them products with higher tenure.

