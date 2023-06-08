#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# for plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
sns.set_style('darkgrid')

# Silhouette analysis
from sklearn.metrics import silhouette_score

# To perform KMeans clustering 
from sklearn.cluster import KMeans

# for scaling
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


# In[2]:


A1 = pd.read_excel("C:\\Users\hp\Downloads\Online Retail.xlsx")


# In[3]:


A1.head()
#looking at top 5 rows


# In[4]:


A1.shape
#checking the no of rows and columns


# In[5]:


A1.info()
#looking out overall information of the data


# In[6]:


A1.isnull().sum()
# checking the number of missing values in each column


# In[7]:


A1.duplicated().sum()
# count of duplicated rows in the data


# In[8]:


A1 = A1[~A1.duplicated()]
# removing the duplicate rows


# In[9]:


A1.shape


# In[10]:


A1[A1['InvoiceNo'].str.startswith('C')==True]
# these are the transactions that have negative quantity which indicates returned or cancelled orders


# In[11]:


A1 = A1[A1['InvoiceNo'].str.startswith('C')!=True]
A1.shape
# removing all the invoice number who starts with 'C' as they are returned orders


# In[12]:


A1.InvoiceNo.nunique()
# checking the number of unique transactions
# though there are more than 5 lakh entries but the number of transaction happened is 21892


# In[13]:


A1.StockCode.nunique()
# checking the unique stock ids in the data or number of unqiue item sold by retailer


# In[14]:


A1.StockCode.value_counts().head(10)
# top 10 stock ids that sold the most


# In[15]:


A1.Quantity.describe()
# looking at the distribution of the quantity
# we seen that there is negative value which might indicate return orders


# In[16]:


A1[A1['Quantity']<0]
# looking at the data where quantity is negative and possible explanation is these are return orders or cancelled order


# In[17]:


A1 = A1[A1['Quantity']>=0]
A1.shape
# keeping only those transactions that have successfully ordered


# In[18]:


print('The minimum date is:',A1.InvoiceDate.min())
print('The maximum date is:',A1.InvoiceDate.max())


# In[19]:


A1.UnitPrice.describe()
# checking the distribution of unit price


# In[20]:


A1.Country.value_counts(normalize=True)
# we see that more than 90% have country as UK which is obvious as the retailer is UK based


# In[21]:


A1['Country'] = A1['Country'].apply(lambda x:'United Kingdom' if x=='United Kingdom' else 'Others')
A1.Country.value_counts(normalize=True)
# putting UK as one country and combine rest countries into one category


# In[22]:


# checking the number of unique item list
A1.Description.nunique()


# In[23]:


# there are cases where the descriptions contains some code/name which are not directly refers to sales
# checking the data where description = ? and it is noted that customerid is NaN and unit price is 0
A1[A1['Description'].str.startswith('?')==True]


# In[24]:


# removing all the above entries
A1 = A1[A1['Description'].str.startswith('?')!=True]
A1.shape


# In[25]:


# checking the data where description = * and it is noted that customerid is NaN
A1[A1['Description'].str.startswith('*')==True]


# In[26]:


# replacing with appropriate name
A1['Description'] = A1['Description'].replace(('*Boombox Ipod Classic','*USB Office Mirror Ball'),
                                             ('BOOMBOX IPOD CLASSIC','USB OFFICE MIRROR BALL'))


# In[27]:


# Description have actual entries in uppercase words and those who don't have are some of the noises in the dataset
A1[A1['Description'].str.islower()==True]['Description'].value_counts()


# In[28]:


# removing all the above noises
A1 = A1[A1['Description'].str.islower()!=True]
A1.shape


# In[29]:


# Description have actual entries in uppercase words and those who don't have are some of the noises in the dataset
A1[A1['Description'].str.istitle()==True]['Description'].value_counts()


# In[30]:


# removing all the above listed noises
A1 = A1[A1['Description'].str.istitle()!=True]
A1.shape


# In[31]:


A1['Description'] = A1['Description'].str.strip()


# In[32]:


# count of unique customer
A1.CustomerID.nunique()


# In[33]:


# checking where customer id is null
A1[A1.CustomerID.isnull()]


# In[34]:


# removing entries where customer id is null
A1 = A1[~A1.CustomerID.isnull()]
A1.shape


# In[35]:


A1.info()


# In[36]:


A1.isnull().sum()


# 
# # EDA

# In[37]:


# creating some columns for exploratory

A1['Amount'] = A1['Quantity']*A1['UnitPrice']
A1['year'] = A1['InvoiceDate'].dt.year
A1['month'] = A1['InvoiceDate'].dt.month
A1['day'] = A1['InvoiceDate'].dt.day
A1['hour'] = A1['InvoiceDate'].dt.hour
A1['day_of_week'] = A1['InvoiceDate'].dt.dayofweek


# In[38]:


A1.head()


# In[39]:


column = ['InvoiceNo','Amount']

plt.figure(figsize=(15,5))
for i,j in enumerate(column):
    plt.subplot(1,2,i+1)
    sns.barplot(x = A1[A1['Country']=='United Kingdom'].groupby('Description')[j].nunique().sort_values(ascending=False).head(10).values,
                y = A1[A1['Country']=='United Kingdom'].groupby('Description')[j].nunique().sort_values(ascending=False).head(10).index,
                color='blue')
    plt.ylabel('')
    if i==0:
        plt.xlabel('Sum of quantity')
        plt.title('Top 10 products purchased by customers in UK',size=15)
    else:
        plt.xlabel('Total Sales')
        plt.title('Top 10 products with most sales in UK',size=15)
        
plt.tight_layout()
plt.show()


# In[40]:


column = ['Others','United Kingdom']

plt.figure(figsize=(15,5))
for i,j in enumerate(column):
    plt.subplot(1,2,i+1)
    sns.barplot(x = A1[A1['Country']==j].groupby('Description')['UnitPrice'].mean().sort_values(ascending=False).head(10).values,
                y = A1[A1['Country']==j].groupby('Description')['UnitPrice'].mean().sort_values(ascending=False).head(10).index,
                color='blue')
    plt.ylabel('')
    if i==0:
        plt.xlabel('Unit Price')
        plt.title('Top 10 high value products outside UK',size=15)
    else:
        plt.xlabel('Unit Price')
        plt.title('Top 10 high value products in UK',size=15)
        
plt.tight_layout()
plt.show()


# In[41]:


# Looking the distribution of column Quantity
plt.figure(figsize=(10,7))

skewness = round(A1.Quantity.skew(),2)
kurtosis = round(A1.Quantity.kurtosis(),2)
mean = round(np.mean(A1.Quantity),0)
median = np.median(A1.Quantity)

plt.subplot(2,2,1)
sns.boxplot(y=A1.Quantity)
plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.subplot(2,2,2)
sns.boxplot(y=A1[A1.Quantity<5000]['Quantity'])
plt.title('Distribution when Quantity<5000')

plt.subplot(2,2,3)
sns.boxplot(y=A1[A1.Quantity<200]['Quantity'])
plt.title('Distribution when Quantity<200')

plt.subplot(2,2,4)
sns.boxplot(y=A1[A1.Quantity<50]['Quantity'])
plt.title('Distribution when Quantity<50')

plt.show()


# In[42]:


# removing the expectional case where quantity > 70000
A1 = A1[A1['Quantity']<70000]


# In[43]:


# Looking the distribution of column Unit Price
plt.figure(figsize=(10,7))

skewness = round(A1.UnitPrice.skew(),2)
kurtosis = round(A1.UnitPrice.kurtosis(),2)
mean = round(np.mean(A1.UnitPrice),0)
median = np.median(A1.UnitPrice)

plt.subplot(2,2,1)
sns.boxplot(y=A1.UnitPrice)
plt.title('Boxplot\n Mean:{}\n Median:{}\n Skewness:{}\n Kurtosis:{}'.format(mean,median,skewness,kurtosis))

plt.subplot(2,2,2)
sns.boxplot(y=A1[A1.UnitPrice<300]['UnitPrice'])
plt.title('Distribution when Unit Price<300')

plt.subplot(2,2,3)
sns.boxplot(y=A1[A1.UnitPrice<50]['UnitPrice'])
plt.title('Distribution when Unit Price<50')

plt.subplot(2,2,4)
sns.boxplot(y=A1[A1.UnitPrice<10]['UnitPrice'])
plt.title('Distribution when Unit Price<10')

plt.show()


# In[44]:


plt.figure(figsize=(12,5))
A1[A1['Country']=='United Kingdom'].groupby(['year','month'])['Amount'].sum().plot(kind='line',label='UK',color='blue')
A1[A1['Country']=='Others'].groupby(['year','month'])['Amount'].sum().plot(kind='line',label='Other',color='grey')
plt.xlabel('Year-Month',size=12)
plt.ylabel('Total Sales', size=12)
plt.title('Sales in each month for an year', size=15)
plt.legend(fontsize=12)
plt.show()


# In[45]:


plt.figure(figsize=(12,5))
A1[A1['Country']=='United Kingdom'].groupby(['day'])['Amount'].sum().plot(kind='line',label='UK',color='blue')
A1[A1['Country']=='Others'].groupby(['day'])['Amount'].sum().plot(kind='line',label='Other',color='grey')
plt.xlabel('Day',size=12)
plt.ylabel('Total Sales', size=12)
plt.title('Sales on each day of a month', size=15)
plt.legend(fontsize=12)
plt.show()


# In[46]:


plt.figure(figsize=(12,5))
A1[A1['Country']=='United Kingdom'].groupby(['hour'])['Amount'].sum().plot(kind='line',label='UK',color='blue')
A1[A1['Country']=='Others'].groupby(['hour'])['Amount'].sum().plot(kind='line',label='Other',color='grey')
plt.xlabel('Hours',size=12)
plt.ylabel('Total Sales', size=12)
plt.title('Sales in each hour in a day', size=15)
plt.legend(fontsize=12)
plt.show()


# # Cohort Analysis
# ![image.png](attachment:image.png)
# 
# An analytical techniques that focuses on analyzing the behavior of a group of users/customers over time, thereby uncovering insights about the experiences of those customers, and what companies can do to better those experiences.

# In[47]:


# copying the data into new df
A1_cohort = A1.copy()
# select only limited columns
A1_cohort = A1_cohort.iloc[:,:9]
A1_cohort.head()


# #For cohort analysis, there are a few labels that we have to create:
# #Invoice Month: A string representation of the year and month of a single transaction/invoice.
# #Cohort Month: A string representation of the the year and month of a customer’s first purchase. This label is common across all invoices for a particular customer.
# #Cohort period: A integer representation a customer’s stage in its “lifetime”. The number represents the number of months passed since the first purchase.

# In[48]:


# creating the first variable 'Invoice Month'
# extracting only year-month from Invoice Date and day will be 1 automatically

A1_cohort['InvoiceMonth'] = A1_cohort['InvoiceDate'].dt.strftime('%Y-%m')
# converting the variable to datetime format
A1_cohort['InvoiceMonth'] = pd.to_datetime(A1_cohort['InvoiceMonth'])


# In[49]:


# creating the second variable 'Cohort Month'
# getting the first time purchase date for each customer

A1_cohort['CohortMonth'] = A1_cohort.groupby('CustomerID')['InvoiceMonth'].transform('min')
# converting the variable to datetime format
A1_cohort['CohortMonth'] = pd.to_datetime(A1_cohort['CohortMonth'])


# In[50]:


A1_cohort.info()


# In[51]:


# creating the third variable 'Cohort Period'
# for this we create a function which calculates the number of month between their first purchase date and Invoice date

def diff_month(d1, d2):
    return((d1.dt.year - d2.dt.year) * 12 + d1.dt.month - d2.dt.month)

A1_cohort['CohortPeriod'] = diff_month(A1_cohort['InvoiceMonth'], A1_cohort['CohortMonth'])


# In[52]:


A1_cohort.sample(5)


# In[53]:


customer_cohort = A1_cohort.pivot_table(index='CohortMonth', columns='CohortPeriod', values='CustomerID', aggfunc='nunique')
customer_cohort

Observations:
The above table show retention and acquistion of customers.
Vertically i.e. the first column '0' tells how many new customers the business acquired in a particular month. ex: 884 is the number of customers business acquired in Dec'201
Horizontally i.e the first row tells the number of customers who is continuing to be part of business since their first purchase i.e. Dec'2010. ex: 323 is the number of customers out of 884 that continue to purchase one month after their first purchase, 286 is the number of customers that continue to purchase two months after their first purchase, and so on.
# In[54]:


# Retention table

cohort_size = customer_cohort.iloc[:,0]
retention = customer_cohort.divide(cohort_size,axis=0) #axis=0 to ensure the divide along the row axis
retention.index = pd.to_datetime(retention.index).date
retention.round(3) * 100 #to show the number as percentage


# Observations:
# The above table is nothing but showing value in percentages.

# In[55]:


#Build the heatmap or pictorial representation of above table

plt.figure(figsize=(15, 8))
plt.title('Retention Rates(in %) over one year period', size=15)
sns.heatmap(data=retention, annot = True, fmt = '.0%', cmap="summer_r")
plt.show()
    


# In[56]:


amount_cohort = A1_cohort.pivot_table(index='CohortMonth', columns='CohortPeriod', values='Amount', aggfunc='mean').round(2)
amount_cohort


# Observation:
# The above table shows the average amount spent by the group of customers over the period of time.

# In[57]:


#Build the heatmap or pictorial representation of above table

amount_cohort.index = pd.to_datetime(amount_cohort.index).date
plt.figure(figsize=(15, 8))
plt.title('Average Spending Over Time', size=15)
sns.heatmap(data = amount_cohort, annot = True, cmap="summer_r")
plt.show()


# # RFM Analysis
Recency: How much time has elapsed since a customer’s last activity or transaction with the brand? Activity is usually a purchase, although variations are sometimes used, e.g., the last visit to a website or use of a mobile app. In most cases, the more recently a customer has interacted or transacted with a brand, the more likely that customer will be responsive to communications from the brand.
Frequency: How often has a customer transacted or interacted with the brand during a particular period of time? Clearly, customers with frequent activities are more engaged, and probably more loyal, than customers who rarely do so. And one-time-only customers are in a class of their own.
Monetary: Also referred to as “monetary value,” this factor reflects how much a customer has spent with the brand during a particular period of time. Big spenders should usually be treated differently than customers who spend little. Looking at monetary divided by frequency indicates the average purchase amount – an important secondary factor to consider when segmenting customers.
# In[ ]:


# copying the data in other df
A1_rfm = A1.copy()
# keeping only desired columns
A1_rfm = A1_rfm.iloc[:,:9]
A1_rfm.head()


# In[ ]:


# extracting the RECENCY

recency = pd.DataFrame(A1_rfm.groupby('CustomerID')['InvoiceDate'].max().reset_index())
recency['InvoiceDate'] = pd.to_datetime(recency['InvoiceDate']).dt.date
recency['MaxDate'] = recency['InvoiceDate'].max()
recency['recency'] = (recency['MaxDate'] - recency['InvoiceDate']).dt.days + 1
recency = recency[['CustomerID','recency']]
recency.head()


# In[ ]:


# extracting the FREQUENCY

frequency = pd.DataFrame(A1_rfm.groupby('CustomerID')['InvoiceNo'].nunique().reset_index())
frequency.columns = ['fCustomerID','frequency']
frequency.head()


# In[ ]:


# extracting the MONETARY

monetary = pd.DataFrame(A1_rfm.groupby('CustomerID')['Amount'].sum().reset_index())
monetary.columns = ['mCustomerID','monetary']
monetary.head()


# In[ ]:


# combining the three into one table

rfm = pd.concat([recency,frequency,monetary], axis=1)
rfm.drop(['fCustomerID','mCustomerID'], axis=1, inplace=True)
rfm.head(10)


# In[ ]:


# checking the overall highlights. The number of distinct customers are 4334
rfm.info()


# In[ ]:


# checking the summary
rfm.describe()


# In[ ]:


# assigning the numbers to RFM values. The better the RFM value higher the number
# note that this process is reverse for R score as lower the value the better it is

rfm['recency_score'] = pd.cut(rfm['recency'], bins=[0,18,51,143,264,375], labels=[5,4,3,2,1])
rfm['recency_score'] = rfm['recency_score'].astype('int')
rfm['frequency_score'] = pd.cut(rfm['frequency'], bins=[0,1,2,5,9,210], labels=[1,2,3,4,5])
rfm['frequency_score'] = rfm['frequency_score'].astype('int')
rfm['monetary_score'] = pd.cut(rfm['monetary'], bins=[-1,306,667,1650,3614,290000], labels=[1,2,3,4,5])
rfm['monetary_score'] = rfm['monetary_score'].astype('int')


# In[ ]:


rfm.info()


# In[ ]:


# summing the R,F,M score to make a one single column that has value range from 3-15

def score_rfm(x) : return (x['recency_score']) + (x['frequency_score']) + (x['monetary_score'])
rfm['score'] = rfm.apply(score_rfm,axis=1 )
rfm.head()


# In[ ]:


rfm.score.describe()


# In[ ]:


# assigning the customers into one of the category Bad, Bronze, Silver, Gold and Platinum based upon the score they get
# we make cuts using percentiles. It can be done in many other ways

rfm['customer_type'] = pd.cut(rfm['score'], bins=[0,6,8,11,13,16], labels=['Bad','Bronze','Silver','Gold','Platinum'])
rfm.head()


# In[ ]:


round(rfm.customer_type.value_counts(normalize=True)*100,0)

Observations:
We see that around 9% of customers are in platinum category and these are the customers who score is best in all the three RFM. Combining with the gold 19% customers are those who are genuine and honest with the business.
Silver category are those where the business can target to convert them into gold category by rolling out offers and new strategies for them.
Bad category are those who are less concerned for the business and does not put much efforts to bring them back.
# In[ ]:


# looking the RFM value for each of the category
rfm.groupby('customer_type')['recency','frequency','monetary'].mean().round(0)


# In[ ]:


column = ['recency','frequency','monetary']
plt.figure(figsize=(15,4))
for i,j in enumerate(column):
    plt.subplot(1,3,i+1)
    rfm.groupby('customer_type')[j].mean().round(0).plot(kind='bar', color='pink')
    plt.title('What is the {} of each customer type'.format(j), size=13)
    plt.xlabel('')
    plt.xticks(rotation=45)

plt.show()


# # k-Means Clustering

# In[ ]:


# copying the data into new variable
A1_kmeans = rfm.copy()
# taking only relevant columns
A1_kmeans = A1_kmeans.iloc[:,:4]
A1_kmeans.head()


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(A1_kmeans.recency, A1_kmeans.frequency, color='grey', alpha=0.3)
plt.title('Recency vs Frequency', size=15)
plt.subplot(1,3,2)
plt.scatter(A1_kmeans.monetary, A1_kmeans.frequency, color='grey', alpha=0.3)
plt.title('Monetary vs Frequency', size=15)
plt.subplot(1,3,3)
plt.scatter(A1_kmeans.recency, A1_kmeans.monetary, color='grey', alpha=0.3)
plt.title('Recency vs Monetary', size=15)
plt.show()


# In[ ]:


# checking the distribution of the variables

column = ['recency','frequency','monetary']
plt.figure(figsize=(15,5))
for i,j in enumerate(column):
    plt.subplot(1,3,i+1)
    sns.boxplot(A1_kmeans[j], color='skyblue')
    plt.xlabel('')
    plt.title('{}'.format(j.upper()), size=13)
plt.show()


# In[ ]:


# Removing outliers for Monetary
Q1 = A1_kmeans.monetary.quantile(0.05)
Q3 = A1_kmeans.monetary.quantile(0.95)
IQR = Q3 - Q1
A1_kmeans = A1_kmeans[(A1_kmeans.monetary >= Q1 - 1.5*IQR) & (A1_kmeans.monetary <= Q3 + 1.5*IQR)]

# Removing outliers for Recency
Q1 = A1_kmeans.recency.quantile(0.05)
Q3 = A1_kmeans.recency.quantile(0.95)
IQR = Q3 - Q1
A1_kmeans = A1_kmeans[(A1_kmeans.recency >= Q1 - 1.5*IQR) & (A1_kmeans.recency <= Q3 + 1.5*IQR)]

# Removing outliers for Frequency
Q1 = A1_kmeans.frequency.quantile(0.05)
Q3 = A1_kmeans.frequency.quantile(0.95)
IQR = Q3 - Q1
A1_kmeans = A1_kmeans[(A1_kmeans.frequency >= Q1 - 1.5*IQR) & (A1_kmeans.frequency <= Q3 + 1.5*IQR)]


# In[ ]:


# resetting the index
A1_kmeans = A1_kmeans.reset_index(drop=True)
A1_kmeans.info()


# In[ ]:


# looking at random 5 rows
A1_kmeans.sample(5)


# In[ ]:


# removing customer id as it will not used in making cluster
A1_kmeans = A1_kmeans.iloc[:,1:]

# scaling the variables and store it in different df
standard_scaler = StandardScaler()
A1_kmeans_norm = standard_scaler.fit_transform(A1_kmeans)

# converting it into dataframe
A1_kmeans_norm = pd.DataFrame(A1_kmeans_norm)
A1_kmeans_norm.columns = ['recency','frequency','monetary']
A1_kmeans_norm.head()

Initially without any knowledge we are clustering the data into 5 clusters. The only intution to do is as in RFM we categorize the data into 5 categories.
Later we look different methods to decide the optimal value for k.
# In[ ]:


# Kmeans with K=5

model_clus5 = KMeans(n_clusters = 5)
model_clus5.fit(A1_kmeans_norm)


# In[ ]:


# checking the labels
model_clus5.labels_


# In[ ]:


A1_kmeans['clusters'] = model_clus5.labels_
A1_kmeans.head()


# In[ ]:


A1_kmeans.groupby('clusters').mean().round(0)


# Optimal Number of Clusters above shown

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




