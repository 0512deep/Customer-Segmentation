# I.PROLOGUE
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
