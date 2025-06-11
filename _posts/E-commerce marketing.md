---
layout: post
title: Understanding E-Commerce Consumer Buying Habits  
image: "/posts/e-commerce.png"
tags: [Python, Power BI]
---

I took the pleasure to segment and analyze customers of an e-commerce business to better understand their purchasing behavior and provided recommendations for targeted marketing and better resource allocation. The aim was to improve customer experiences, and potentially increase sales and customer retention. 

---

First I imported the required libraries

```ruby
# load libraries
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set()
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy 
import statsmodels.api as sme

```
First I loaded and combined the datasents using pandas. I started by combining 4 of the five datasets and then I converted the transaction date to datetime using  pd.to_datetime() before merging the new df to the discounts and coupons data.

```ruby
df1 = pd.read_excel("CustomersData.xlsx")
df2 = pd.read_csv("Discount_Coupon.csv")
df3 = pd.read_csv("Marketing_Spend.csv")
df4 = pd.read_csv("Online_Sales.csv")
df5 = pd.read_excel("Tax_amount.xlsx")

# merge the dataset using pandas df.merge() function
df = df1.merge(df4, on = "CustomerID", how = "inner")
df = df.merge(df5, on = "Product_Category")
df = df.merge(df3, left_on="Transaction_Date", right_on="Date")

#convert transaction date to datetime
df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], format='%m/%d/%Y')
df['Month'] = df["Transaction_Date"].apply(lambda x : x.strftime('%m'))
df['Month'] = df['Month'].astype('int')

#convert 'month' in df2 to datetime also
df2["Month"] = df2['Month'].apply(lambda x: datetime.datetime.strptime(x, '%b').month)

# now merge df2 to the rest of the data on 'Month' and 'Product_Category'
df = df.merge(df2, on = ['Month','Product_Category'], how = 'outer')
```
# Sales Behavior 

To analyze the sales behavior, I calculated total net sales (total transaction value) excluding tax and delivery charges for each transaction by multiplying unit price by quantity purchased and the discount.

```ruby
#convert discount to decimal
df["Discount_pct"] = df["Discount_pct"] / 100
#calculate net_sales
df["net_sales"] = df["Avg_UnitPrice"] * df["Quantity"] * (1 - df['Discount_pct'])
```
Next I checked the distribution of net sales and found that the data is highly skewed.

```ruby
#sales distribution
sns.boxplot(data = df, y = "net_sales")
plt.ylabel("Total Net Sales")
plt.title("Sales Distribution")
plt.savefig("sales_distribution.png")
plt.show()
```
![alt text](/img/sales_distribution.png "Box Plot")
---
I then generated descriptive statistics to better understand the central tendency, dispersion and shape of the net sale's distribution. I also looked at the quantiles and noticed the skewedness in transaction amount.
```ruby
df["net_sales"].describe()
df["net_sales"].quantile([0.5, 0.75, 0.95, 0.99])
```

![alt text](/img/quantiles.png "Quantiles")

### Business Insights

I noticed that most transactions are relatively small: The median is just 24.5.

There is a long tail: The gap between the median (24.5) and the 99th percentile (428.4) shows that while most transactions are modest, a small number are much larger.

**High-value customers:**

The top 5% and especially the top 1% of transactions are significantly higher, which may indicate VIP or bulk buyers.

The top 1% are rare, very high-value transactions.

This prompted me to perform rule-based customer segmentation and investigate the high value transactions.

## Rule-Based Customer Segmentation

First I performed discriptive statistics on the top 1% spenders to undestand the distribution of transaction values and the quantity of products they buy
```ruby
one_percent = df["net_sales"].quantile([0.99]).iloc[0]
top_1_percent = df[df["net_sales"] > one_percent]
top_1_percent[["net_sales", "Quantity"]].describe()

ax = sns.histplot(data = top_1_percent, x = "net_sales")
for location in ['right', 'top']:
        ax.spines[location].set_visible(False)
plt.title("Top 1% Spenders")
plt.xlabel("Net Sales")
plt.savefig("top1_percent.png")
plt.show()
```
![alt text](/img/top1_percent.png "Histogram")
---
I found that even in the top 1% spenders, there is still a long tail because 75 percent of the transactions are below the mean transaction value. The average transaction is $825 while the maximum transaction is about $8.5k. A result, I decided to introduce another threshold: the 99 percentile transactions to range from above 99% but below $1,000. I introduced a VIP group for the customers whose transaction value exceed $1k per transaction.

### Creating Customer Segments
Top 5% Customers by Spend
Top 1% spenders
Ultra high-value customers by spend: Transactions above 1k
The rest of the customers
```ruby
#define the criteria
cutoff = df["net_sales"].quantile(0.95)
one_percent = df["net_sales"].quantile([0.99]).iloc[0]
above_1k = df[df["net_sales"] > 1000]

# define customer segments
vip = df[df["net_sales"] > 1000]
premium =  df[(df["net_sales"] > one_percent) & (~df["CustomerID"].isin(vip["CustomerID"]))]
loyalty = df[(df["net_sales"] > cutoff) & (~df["CustomerID"].isin(premium["CustomerID"])) & (~df["CustomerID"].isin(vip["CustomerID"]))]
regular = df[(df["net_sales"] <= cutoff) & (~df["CustomerID"].isin(premium["CustomerID"])) \
& (~df["CustomerID"].isin(vip["CustomerID"])) & (~df["CustomerID"].isin(loyalty["CustomerID"]))]

# number of customers per segment
print("Vip customers:", vip["CustomerID"].nunique())
print("Premium customers:", premium["CustomerID"].nunique())
print("Loyalty customers:", loyalty["CustomerID"].nunique())
print("Regular customers:", regular["CustomerID"].nunique())
print("Total customers:", df["CustomerID"].nunique())

#number of transactions per segment
print("Vip transactions:", vip["CustomerID"].count())
print("Premium transactions:", premium["CustomerID"].count())
print("Loyalty transactions:", loyalty["CustomerID"].count())
print("Regular transactions:", regular["CustomerID"].count())
```
![image](https://github.com/user-attachments/assets/b186047f-3fd9-43f9-bd1d-1f95e5d0a92e)

### Business Insights

**High value customers:**

I discovered that we have high-value customers and decided to focus my analysis on understanding their behavior and preferences for personalized marketing.

The 41 customers with transactions above 1,000 represent a small but extremely valuable segment. Targeting them with personalized offers, loyalty programs, or exclusive services can drive significant revenue growth.

Top percentile leverage:
The 807 (95th percentile) and 299 (99th percentile) customers are critical for sustaining and growing the business. Understanding their preferences, purchase patterns, and demographics (e.g., location, gender) can help to tailor marketing and retention strategies.

### Assign customer segments

Customer Segmentation: Rule-Based

VIP/Elite tier: The 41 ultra high-value customers

Premium tier: The 258 in the 99th percentile

Loyalty tier: The 508 in the 95th percentile

Regular: The rest of the customers, 661

```ruby
# Initialize segment column with NaNs
df["segment"] = np.nan
df["segment"] = pd.Series(dtype="object")

# Assign segments in order of exclusivity
df.loc[df["CustomerID"].isin(vip["CustomerID"]), "segment"] = "VIP Tier"

df.loc[
    (df["CustomerID"].isin(premium["CustomerID"])) & (df["segment"].isna()),
    "segment"] = "Premium Tier"

df.loc[
    (df["CustomerID"].isin(loyalty["CustomerID"])) & (df["segment"].isna()),
    "segment"] = "Loyalty Tier"

df.loc[
    (df["CustomerID"].isin(regular["CustomerID"])) & (df["segment"].isna()),
    "segment"] = "Regular"
```
After performing more analysis, I did research and found that Mailchip suggests that 10 AM is the most optimal time to send out newsletters/emails to subscribers. I used that as  my recommendation to The Column.

![alt text](/img/posts/Opens_Analysis.jpg "Opens Analysis")

![alt text](/img/posts/Clicks_Analysis.jpg "Clicks Analysis")

![alt text](/img/posts/Lifetime_Column.jpg "Lifetime Performance")

