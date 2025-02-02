# I. INTRODUCTION
This is a project on customer segmentation, using a dataset from UC Irvine which containing transactions occuring at a UK based online retail store - https://tinyurl.com/UK-Store-Dataset.
The main idea for this project is to find out the different types of customers that are purchasing from the store and split them into different groups to learn their purchase patterns using KMeans, finally suggesting actionable insights based on the results.


# I.I Importing data
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

pd.options.display.float_format = '{:20.2f}'.format
pd.set_option('display.max_columns', 999)

df = pd.read_excel('ONR.xlsx', sheet_name = 0)

df.head(10)

df.info()
```
We start with importing all the necessary libraries for analysis, visualization and clustering. I made adjustments to the display option to make the results more readable and easy to understand. 
Then we import the excel dataset, but we only used the first sheet of the file. After that, the dataset's first 10 rows and the dataset's types and structure.

![image](https://github.com/user-attachments/assets/b75ded92-b828-4b89-94b2-a240dbeacfba)


![image](https://github.com/user-attachments/assets/f3ce23be-aa1d-4f79-acc3-2aed25dcdcf3)

from the above information, we can see that "customer id" is having missing data. The "invoicedate" is already in datetime format, so there wont be issues in converting. 

```python
df.describe()
df.describe(include='O')
```
![image](https://github.com/user-attachments/assets/90925197-cf26-4e89-8583-06fce73b3aa6)
![image](https://github.com/user-attachments/assets/ccb4c02e-4d6f-42b8-b48f-e2fd12eb96ca)

Describe showed us that there's negative values in quantity and price, which is not possible?!

```python
df[df["Customer ID"].isna()].head(10)

df[df["Quantity"] < 0].head(10)
```
![image](https://github.com/user-attachments/assets/9354c985-cb31-4904-aa2f-26599c98a149)


```python
df["Invoice"] = df["Invoice"].astype("str")
df[df["Invoice"].str.match("^\\d{6}$") == False]
```

```python
df["Invoice"].str.replace("[0-9]", "", regex=True).unique()
df[df["Invoice"].str.startswith("A")]
```

```python
df["StockCode"] = df["StockCode"].astype("str")
df[(df["StockCode"].str.match("^\\d{5}$")== False) & (df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$")== False)]
```

```python
cleaned_df = df.copy()
cleaned_df["Invoice"] = cleaned_df["Invoice"].astype("str")

mask = (
     cleaned_df["Invoice"].str.match("^\\d{6}$") == True
    )

cleaned_df = cleaned_df[mask]
cleaned_df
```

```python
cleaned_df["StockCode"] = cleaned_df["StockCode"].astype("str")

mask = (
    (cleaned_df["StockCode"].str.match("^\\d{5}$") == True)
    | (cleaned_df["StockCode"].str.match("^\\d{5}[a-zA-Z]+$") == True)
    | (cleaned_df["StockCode"].str.match("^PADS$") == True)
)

cleaned_df = cleaned_df[mask]

cleaned_df
```

```python
cleaned_df.describe()
cleaned_df = cleaned_df.dropna(subset=["Customer ID"])
cleaned_df.describe()
```

```python
cleaned_df[cleaned_df["Price"]==0]
len(cleaned_df[cleaned_df["Price"]==0])
```

```python
cleaned_df["Price"].min()

cleaned_df = (cleaned_df[cleaned_df["Price"]>0.0])
len(cleaned_df)/len(df)
```

```python
cleaned_df["SalesLineTotal"] = cleaned_df["Quantity"] * cleaned_df["Price"]

cleaned_df
```

```python
aggregated_df = cleaned_df.groupby(by = "Customer ID", as_index = False) \
    .agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency = ("Invoice", "nunique"),
        LastInvoiceDate = ("InvoiceDate", "max")
    )
aggregated_df.head(5)
```

```python
aggregated_df["LastInvoiceDate"] = pd.to_datetime(aggregated_df["LastInvoiceDate"])

max_invoice_date = aggregated_df["LastInvoiceDate"].max()

aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days

aggregated_df.head(5)
```

```python
cleaned_df.describe()
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(aggregated_df['MonetaryValue'], bins=10, color='skyblue', edgecolor='black')
plt.title('Monetary Value Distribution')
plt.xlabel('Monetary Value')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
plt.hist(aggregated_df['Frequency'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
plt.hist(aggregated_df['Recency'], bins=20, color='salmon', edgecolor='black')
plt.title('Recency Distribution')
plt.xlabel('Recency')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data=aggregated_df['MonetaryValue'], color='skyblue')
plt.title('Monetary Value Boxplot')
plt.xlabel('Monetary Value')

plt.subplot(1, 3, 2)
sns.boxplot(data=aggregated_df['Frequency'], color='lightgreen')
plt.title('Frequency Boxplot')
plt.xlabel('Frequency')

plt.subplot(1, 3, 3)
sns.boxplot(data=aggregated_df['Recency'], color='salmon')
plt.title('Recency Boxplot')
plt.xlabel('Recency')

plt.tight_layout()
plt.show()
```

```python
M_Q1 = aggregated_df["MonetaryValue"].quantile(0.25)
M_Q3 = aggregated_df["MonetaryValue"].quantile(0.75)
M_IQR = M_Q3 - M_Q1

monetary_outliers_df = aggregated_df[(aggregated_df["MonetaryValue"] > (M_Q3 + 1.5 * M_IQR)) | (aggregated_df["MonetaryValue"] < (M_Q1 - 1.5 * M_IQR))].copy()

monetary_outliers_df.describe()
```

```python
F_Q1 = aggregated_df['Frequency'].quantile(0.25)
F_Q3 = aggregated_df['Frequency'].quantile(0.75)
F_IQR = F_Q3 - F_Q1

frequency_outliers_df = aggregated_df[(aggregated_df['Frequency'] > (F_Q3 + 1.5 * F_IQR)) | (aggregated_df['Frequency'] < (F_Q1 - 1.5 * F_IQR))].copy()

frequency_outliers_df.describe()
```

```python
non_outliers_df = aggregated_df[(~aggregated_df.index.isin(monetary_outliers_df.index)) & (~aggregated_df.index.isin(frequency_outliers_df.index))]
non_outliers_df.describe()
```

```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(data=non_outliers_df['MonetaryValue'], color='skyblue')
plt.title('Monetary Value Boxplot')
plt.xlabel('Monetary Value')

plt.subplot(1, 3, 2)
sns.boxplot(data=non_outliers_df['Frequency'], color='lightgreen')
plt.title('Frequency Boxplot')
plt.xlabel('Frequency')

plt.subplot(1, 3, 3)
sns.boxplot(data=non_outliers_df['Recency'], color='salmon')
plt.title('Recency Boxplot')
plt.xlabel('Recency')

plt.tight_layout()
plt.show()
```

```python
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(projection="3d")

scatter = ax.scatter(non_outliers_df["MonetaryValue"], non_outliers_df["Frequency"], non_outliers_df["Recency"])

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data')

plt.show()
```

```python
scaler = StandardScaler()

scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

scaled_data
```

```python
scaled_data_df = pd.DataFrame(scaled_data, index=non_outliers_df.index, columns=("MonetaryValue", "Frequency", "Recency"))

scaled_data_df
```

```python
fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(projection="3d")

scatter = ax.scatter(scaled_data_df["MonetaryValue"], scaled_data_df["Frequency"], scaled_data_df["Recency"])

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data')

plt.show()
```

```python
max_k = 12

inertia = []
silhoutte_scores = []
k_values = range(2, max_k + 1)

for k in k_values:

    kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000)

    cluster_labels = kmeans.fit_predict(scaled_data_df)

    sil_score = silhouette_score(scaled_data_df, cluster_labels)

    silhoutte_scores.append(sil_score)

    inertia.append(kmeans.inertia_)
```

```python
plt.figure(figsize =(14, 6))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o', color="red")
plt.title("KMeans Inertia for Different Values of k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_values, silhoutte_scores, marker='o', color="purple")
plt.title("Silhouette Scores for Different Values of k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.grid(True)

plt.tight_layout()
plt.show()
```

```python
kmeans = KMeans(n_clusters = 3, random_state=42, max_iter=1000)

cluster_labels = kmeans.fit_predict(scaled_data_df)

cluster_labels
```

```python
non_outliers_df["Cluster"] = cluster_labels

non_outliers_df
```

```python
cluster_colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728'}

colors = non_outliers_df['Cluster'].map(cluster_colors)

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(projection="3d")

scatter = ax.scatter(non_outliers_df["MonetaryValue"], non_outliers_df["Frequency"], non_outliers_df["Recency"], c=colors, marker="o")

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data by cluster')

plt.show()
```

```python
cluster_colors = {0:'#1f77b4', 1:'#ff7f0e', 2:'#2ca02c', 3:'#d62728'}  # 3 clusters

plt.figure(figsize=(12, 18))

plt.subplot(3, 1, 1)
sns.violinplot(x="Cluster", y="MonetaryValue", data=non_outliers_df, c=colors)
plt.title('Monetary Value by Cluster')
plt.ylabel('Monetary Value')

plt.subplot(3, 1, 2)
sns.violinplot(x="Cluster", y="Frequency", data=non_outliers_df, c=colors)
plt.title('Frequency by Cluster')
plt.ylabel('Frequency')

plt.subplot(3, 1, 3)
sns.violinplot(x="Cluster", y="Recency", data=non_outliers_df, c=colors)
plt.title('Recency by Cluster')
plt.ylabel('Recency')

plt.tight_layout()
plt.show()
```

```python
cluster_labels = {0:"Nurture", 1:"Reward", 2:"Retain"}

new_df = non_outliers_df

new_df["ClusterLabel"] = new_df["Cluster"].map(cluster_labels)

new_df
```

```python
sns.countplot(x="ClusterLabel", data=new_df)
plt.title("Customer Distribution by Cluster")
plt.show()
```
