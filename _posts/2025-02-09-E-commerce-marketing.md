---
layout: post
title: Understanding E-Commerce Consumer Buying Habits  
image: "/posts/ecommerce-image.png"
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
import plotly.graph_objects as go
```
#### Write a helper function to call whenever I want to plot horizontal bars using seaborn
```ruby
#helper function to plot horizontal bars
def horizontal_bars(df, x, y, palette):
    ax = sns.barplot(data = df, x = x, y = y, orient = 'h', errorbar = None, palette = palette)
    #remove all grids
    for location in ['left', 'right', 'top', 'bottom']:
        ax.spines[location].set_visible(False)
    #move the tick labels to the top of the graph, the top ticks instead of the bottom ones & color the x-tick labels
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(top=False, left=False)
    plt.tight_layout()
    return ax
````
First I loaded and combined the datasets using pandas. I started by combining 4 of the five datasets and then I converted the transaction date to datetime using  pd.to_datetime() before merging the new df to the discounts and coupons data.

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
Top 5% Customers by Spend;
Top 1% spenders;
Ultra high-value customers by spend: Transactions above 1k;
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
![alt text](/img/vip_customers.png "Table")

### Business Insights

**High value customers:**

I discovered that we have high-value customers and decided to focus my analysis on understanding their behavior and preferences for personalized marketing.

The 41 customers with transactions above $1,000 represent a small but extremely valuable segment. Targeting them with personalized offers, loyalty programs, or exclusive services can drive significant revenue growth.

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
## Investigate High-Value Transactions

### Which Location and Gender are Contributing More to the VIP Sales?

The goal is to understand which locations the high spenders come from and their gender distribution in order to inform targeted marketing strategies.

I already have vip dataframe from above (vip = df[df["net_sales"] > 1000]). I used pandas grouby() and agg() functions to generate a summary of vip sales by location and gender.

```ruby
vip_summary = vip.groupby("CustomerID").agg({
    "net_sales": "sum",
    "Transaction_ID": "count",
    "Quantity": "sum",
    "Delivery_Charges": "sum",
    "Location": "first",
    "Gender": "first"
})
# Group by both Location and Gender, then count unique customers
gender_summary = vip.groupby(["Location", "Gender"])["CustomerID"].nunique().reset_index()
# Rename the column for clarity
gender_summary.rename(columns={"CustomerID": "Unique_Customers"}, inplace=True)

#plots
fig = px.bar(vip_summary, x="net_sales", y="Location", color="Gender", orientation='h',
    hover_data=["CustomerID", "Quantity"],
    color_discrete_sequence=["#678199", "#9E7C92"])
fig.update_layout(xaxis_title="", yaxis_title="")
fig.show()

fig = px.bar(gender_summary, x = "Location", y = "Unique_Customers", color = "Gender", barmode="group",  
    title="VIP Customers by Location and Gender", width=600, height=500,
    color_discrete_sequence=["#9E7C92", "#678199"]
)
fig.update_layout(xaxis_title="", yaxis_title="Number of Customers", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)' )
fig.show()
```
![alt text](/img/vipsales_loc_gender.png "Stacked Bars")
![alt text](/img/customers_location.png "Column Plot")

Most VIP sales come from Chicago and California and mainly from female customers. However, There are more male than female customers in Chicago.

### Which Product Categories are Associated with High-Value Transactions?

The aim is to discover which categories drive the most revenue from big spenders and inform inventory, promotions, and procurement.
```ruby
category = vip.groupby("Product_Category")["net_sales"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)

# call horizontal_bars we created above to plot sales by product category
colors = ['brown' if (x > 30000)  else "#b6b6b6" for x in category["sum"]] 
plt.figure(figsize = (9, 4))
fig = horizontal_bars(category, x = "sum", y = "Product_Category", palette = colors)
fig.text(x=-30, y=-1.5, s="Total VIP Sales by Product Category", fontsize = 12, weight = "bold")
plt.xlabel("")
plt.ylabel("")
plt.savefig("vip sales_bycategory.png")
plt.show()
```
![alt text](/img/vip_sales_bycategory.png "Horizontal Bars")

### Which products are mostly bought by vip customers?
```ruby
fig = px.bar(vip, x = "Product_Category", y = "Quantity", color = "Location", barmode="group",
        title = "Quantity Sold to VIP Customers by Category and Location", width=900, height=600,
        color_discrete_sequence=["black", "gray", "#8e501b", "#b6b6b6", "#dc7f18"])
fig.update_layout(xaxis_title="", yaxis_title="", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.show()
```
![alt text](/img/quantity_bycate.png "Bar Plot")

A lot of the vip revenue is coming from apparel and notebooks & journals categories. Bulky orders are coming mainly from Chicago and California. Interestingly, this group of customers do not buy Nest products.

## Temporal Trends of High-Value Transactions

The aim of trends analysis is to identify seasonality or anomalies in high-value sales so as to optimize timing for marketing and restocking.

### Daily Transactions and Sales Over Time
```ruby
# Group by date for transaction count & net sales sum
daily_counts = vip.groupby(vip["Transaction_Date"].dt.date).size().reset_index(name="transaction_count")

daily_sales = vip.groupby(vip["Transaction_Date"])["net_sales"].sum().reset_index()

fig1 = px.line(daily_counts, x="Transaction_Date", y="transaction_count", markers=True, title="Daily Transaction Count", width=900, height=500)
fig1.update_layout(xaxis_title="", yaxis_title="Number of Transactions", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig1.update_traces(line_color = "#8e501b")

fig2 = px.line(daily_sales, x="Transaction_Date", y="net_sales", markers=True, title="Daily Net Sales", width=900, height=500)
fig2.update_layout(xaxis_title="", yaxis_title="Net Sales", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig2.update_traces(line_color = "#8e501b")
fig1.show()
fig2.show()
```
![alt text](/img/vipsales_overtime.png "Line Graph")
![alt text](/img/vip_transactions.png "Line Graph")

## Which Days of the Week do We Expect VIP Sales?

Next I sought to understand which days of the week are we expecting high transactions to inform inventory and staffing.
```ruby
vip["DayOfWeek"] = vip["Transaction_Date"].dt.day_name()

#number of transactions per day
day_counts = vip["DayOfWeek"].value_counts().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]).reset_index()
day_counts.columns = ["DayOfWeek", "TransactionCount"]

fig = px.line(day_counts, x="DayOfWeek", y="TransactionCount", markers=True, title="Transactions by Day of the Week", width=600, height=450)
fig.update_layout(xaxis_title="", yaxis_title="Number of Transactions", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_xaxes(tickangle= -45)
fig.update_traces(line_color = "black")
fig.show()
```
![alt text](/img/transa_weekdays.png "Line Graph")

## Business Insights

After performing more analysis, I discovered that there was a spike of sales in April from high value transactions. Most VIP transactions happen on Thursdays and Fridays and there are literally no transactions on Tuesdays. 

## Comparing Key KPIs Across Customer Segments

Finally I exported the data to Power BI and created a dashboard comparing key KPIs across the four customer segments.

### **Business Insights**

In general, the drivers of revenue are the loyalty and premium groups, and the leading locations in terms of net sales are Chicago and California. Office products are the leading category in quantity sold. There are more female customers than male across all segments and only a few one-time off buyers from the regular group. There was a spike of sales in April from the VIP tier. I used that as  my recommendation to the company.


![alt text](/img/ecommerce_dashboard.png "Dashboard")



