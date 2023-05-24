#!/usr/bin/env python
# coding: utf-8

# # Comcast Telecommunication Complaints
#    # Project
#    

# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


import seaborn as sns


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # loading dataset

# In[13]:


df = pd.read_csv(r"C:\Users\hp\Downloads\1568699544_comcast_telecom_complaints_data.zip")


# In[14]:


df . head()


# In[15]:


print (df.isnull().sum())


# # There are no nan values present in dataset

# In[17]:


df.describe(include='all')


# In[18]:


df.shape


# In[20]:


df= df.drop(['Ticket #','Time'], axis=1)


# In[21]:


df. head()


# # Task-1 Provide the trend chart for the number of complaint s at monthly and dailyu granularity levels

# In[26]:


# Pandas to datetime() method helps to convert string Date time into Python date
df['Date_month_year' ]=df['Date_month_year' ].apply(pd.to_datetime)
# Setting 'Date_month_year' as index
df=df.set_index('Date_month_year')


# # plotting monthly chart

# In[28]:


#dataframe.groupby() function is splitting the data into groups accoring to free
months= df.groupby(pd.Grouper(freq="m")).size().plot()
plt.xlabel("MONTHS")
plt.ylabel("FREQUENCY")
plt.title("MONTHLY TREND CHART")


# INSIGHTS:- From the above trend chart, we can clearly see that complaints for the month of June 2015 are maximum

# In[32]:


#value_count( function os getting a Series containing counts of unique values)
df['Date'].value_counts(dropna=False)[:8]


# plotting daily chart

# In[35]:


df=df.sort_values(by='Date')
plt.figure(figsize=(6,6))
df['Date'].value_counts().plot()
plt.xlabel("Date")
plt.ylabel("FREQUENCY")
plt.title("DAILY TREND CHART")


# Task-2 Provide a table with the frequency of complaint types.

# In[38]:


df['Customer Complaint'].value_counts(dropna=False)[:9]


# In[39]:


df['Customer Complaint'].value_counts(dropna=False)[:9].plot.bar()


# Task 3- Which complaint types are maximum i.e., arounds internet, network issues, or across any other domain.

# In[41]:


internet_issues1=df[df['Customer Complaint'].str.contains("network")].count()


# In[42]:


internet_issues2=df[df['Customer Complaint'].str.contains("speed")].count()


# In[43]:


internet_issues3=df[df['Customer Complaint'].str.contains("data")].count()


# In[44]:


internet_issues4=df[df['Customer Complaint'].str.contains("internet")].count()


# In[45]:


billing_issues1=df[df['Customer Complaint'].str.contains("bill")].count()


# In[46]:


billing_issues2=df[df['Customer Complaint'].str.contains("billing")].count()


# In[47]:


billing_issues3=df[df['Customer Complaint'].str.contains("charges")].count()


# In[48]:


service_issues1=df[df['Customer Complaint'].str.contains("service")].count()


# In[49]:


service_issues2=df[df['Customer Complaint'].str.contains("customer")].count()


# In[54]:


total_internet_issues=internet_issues1+internet_issues2+internet_issues3+internet_issues4
print(total_internet_issues)


# In[57]:


total_billing_issues=billing_issues1+billing_issues2+billing_issues3
print(total_billing_issues)


# In[59]:


total_service_issues=service_issues1+service_issues2
print(total_service_issues)


# In[63]:


other_issues=2224-(total_internet_issues+total_billing_issues+total_service_issues)
print(other_issues)


# INSIGHTS:- From the above analysis we can see that the other issues are maximum.

# 4.Create a new categorical variable with value as Open and
# Closed. Open & Pending is to be categorized as Open and
# Closed & Solved is to be categorized as Closed.

# In[64]:


df.Status.unique()


# In[66]:


df["newStatus"] = ["Open" if Status=="Open" or Status=="Pending" else "Closed" for Status in df['Status']]
df=df.drop(['Status'], axis=1)


# In[68]:


df


# # 5. Which state has the maximum complaints

# In[69]:


df.groupby(["State"]).size().sort_values(ascending=False)[:5]


# INSIGHT:- From the above table we can see that Georgia has maximum complaints

# # Task-6 Provide state wise status of complaints in a stacked bar chart.

# In[70]:


Status_complaints = df.groupby(["State","newStatus"]).size().unstack()
print(Status_complaints)


# In[72]:


Status_complaints.plot.bar(figsize=(10,10),stacked=True)


# INSIGHTS:- From the above chart, we can clearly see that Georgia has maximum complaints

# # Task-7 State which has highest percentage of unresolved complaints

# In[73]:


print(df['newStatus'].value_counts())


# In[74]:


unresolved_data = df.groupby(["State",'newStatus']).size().unstack().fillna(0).sort_values(by='Open',ascending=False)
unresolved_data['Unresolved_cmp_prct'] = unresolved_data['Open']/unresolved_data['Open'].sum()*100
print(unresolved_data)
unresolved_data.plot()


# INSIGHTS:- From the table generated above we can see that Georgia has maximum unresolved complaints i.e. 80.

# # Task-8 Provide the percentage of complaints resolved till date, which were received through the internet and the customer care call

# In[76]:


resolved_data = df.groupby(['Received Via','newStatus']).size().unstack().fillna(0)
resolved_data['resolved'] = resolved_data['Closed']/resolved_data['Closed'].sum()*100
resolved_data['resolved']


# In[77]:


resolved_data.plot(kind="bar",figsize=(8,8))


# INSIGHT:- From the above pie chart we can clearly see that there are total 50.61% coimplaints resolved for Customer resolved for customer care call and 49.39% for received v
