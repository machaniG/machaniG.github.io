---
layout: post
title: Crimsonloop Customer Segmentation and Behavior Analysis
image: "/posts/customers.png"
tags: [Python, Power BI]
---

# Introduction

This project focuses on segmenting and analyzing customers of an e-commerce business, Crimsonloop, to better understand their purchasing behavior. The goal was to provide actionable recommendations for targeted marketing and optimized resource allocation, ultimately improving customer experiences, increasing sales, and enhancing customer retention.

## The Challenge

Crimsonloop needed a deeper understanding of its diverse customer base. Generic marketing campaigns and resource allocation were inefficient, as they failed to account for varying customer values and behaviors. The business needed clear insights to identify high-value customers, anticipate purchasing patterns, and tailor engagement strategies.

## My Approach

I conducted a comprehensive customer segmentation and behavior analysis using Python (Pandas, NumPy, Matplotlib, Seaborn, Plotly) for data manipulation, statistical analysis, and initial visualization, culminating in an interactive Power BI dashboard for holistic KPI comparison. My process involved:

- **Integrated and preprocessed five different datasets**, ensuring data quality and readiness for advanced analysis.
- **Conducted in-depth sales behavior analysis**, revealing significant data skewness that highlighted high-value transaction opportunities.
- **Implemented both rule-based and machine learning (K-means) customer segmentation**, defining distinct customer tiers to identify and analyze high-value segments.
- **Performed comprehensive behavioral and temporal trend analysis of VIP customers**, uncovering critical demographic, product, and purchasing patterns.
- **Developed an interactive Power BI dashboard** for holistic KPI comparison across all customer segments, facilitating easy exploration and data-driven decision-making.


# Key Discoveries & Actionable Insights:

Through rigorous analysis and visualization, I uncovered critical insights that directly informed Crimsonloop's business strategy:

## 1. **Unveiling High-Value Customers & Segmentation:**
   
Initial net sales analysis revealed a highly skewed distribution and a significant "long tail" of high-value transactions. This prompted a multi-faceted approach to segmentation.

**Rule-Based Segmentation:** Based on business understanding and key thresholds, I identified distinct customer tiers:

- **VIP Tier:** Ultra-high-value customers with transactions over $1,000.
- **Premium Tier:** Customers in the 99th percentile.
- **Loyalty Tier:** Customers in the 95th percentile.
- **Regular Tier:** The remaining customers.

This segmentation underscored the importance of focusing on high-value customers for personalized marketing and loyalty programs.

*Initial analysis, as seen in the **Sales Distribution Box Plot** and **Top 1% Spenders Histogram** below, revealed a highly skewed distribution of sales, signaling significant high-value customer opportunities.*

![alt text](/img/sales_distribution.png "Box Plot")

![alt text](/img/top1_percent.png "Histogram")
![alt text](/img/newplot(1).png "Histogram")
*The core logic for defining these rule-based customer tiers in Python is provided below:*

```ruby
# Define segmentation cutoffs based on quantiles for data-driven thresholds
cutoff_95 = df["net_sales"].quantile(0.95)
cutoff_99 = df["net_sales"].quantile(0.99)
vip_threshold > 1000 # Specific business-defined threshold for VIP

# Assign segments based on calculated thresholds, ensuring exclusivity
df.loc[df["net_sales"] > vip_threshold, "segment"] = "VIP Tier"
df.loc[(df["net_sales"] > cutoff_99) & (df["segment"].isna()), "segment"] = "Premium Tier"
df.loc[(df["net_sales"] > cutoff_95) & (df["segment"].isna()), "segment"] = "Loyalty Tier"
df.loc[df["segment"].isna(), "segment"] = "Regular"
```

**Machine Learning-Driven Segmentation (K-Means Clustering):**

To validate and potentially discover deeper, data-driven segments, I also applied K-means clustering using key RFM (Recency, Frequency, Monetary) features. This unsupervised learning approach provided an alternative perspective on natural customer groupings.

*A comparison of the K-means clusters against the rule-based segments revealed a nuanced overlap, validating existing business rules while also highlighting areas for deeper understanding. Specifically, **Cluster 3** showed a strong alignment with the **VIP Tier**, reinforcing its distinctiveness. The **Loyalty, Premium, and Regular Tiers** were primarily distributed across **Cluster 0 and Cluster 2,** suggesting a natural grouping of mid-to-low value customers that could be explored for more granular targeting. This dual approach ensured robust, actionable segmentation.*

*The clustering results below show a visualized scatter plot of RFM features colored by cluster and using segments as markers:*

![alt text](/img/Kmeans_clusters.png "scatter Plot")

*Below is the Python code for applying K-means clustering, starting with the derivation of RFM features from the transaction data:*

```ruby
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime

# Calculate RFM features from the transaction data
# Assuming 'net_sales' for Monetary, 'Transaction_ID' for Frequency, and 'Transaction_Date_x' for Recency
current_date = df['Transaction_Date_x'].max() # Use the most recent transaction date as reference

rfm_data = df.groupby('CustomerID').agg(
    Recency=('Transaction_Date_x', lambda date: (current_date - date.max()).days),
    Frequency=('Transaction_ID', 'count'),
    Monetary=('net_sales', 'sum')
).reset_index()

# Scale features for K-Means to ensure equal weighting
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

# Apply K-Means clustering (e.g., for an optimal number of clusters, determined via elbow method/silhouette score)
# After exploring a number of K, I choose the optimal K = 4 clusters

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # n_init is set to 'auto' or explicit value in newer scikit-learn
rfm_data['Cluster'] = kmeans.fit_predict(rfm_scaled)

# The actual implementation involving exploring optimal K and in-depth cluster profiling can be found here [].
```

## 2. **Geographic & Gender Insights for VIPs:**
   
Analysis of VIP sales by location and gender revealed that **Chicago and California** were the leading regions for high-value transactions, with **female customers** generally contributing more to VIP sales. Interestingly, Chicago showed a higher number of male VIP customers compared to other locations.

*The **VIP Sales by Location and Gender** charts below provide detailed demographic insights, revealing leading regions for high-value transactions and gender distribution.*

![alt text](/img/vip_sales_location_and_gender.png "Stacked Bars")
![alt text](/img/vip_customers_loc_gender.png "Column Plot")

*The summary of VIP sales by location and gender was generated using Pandas for aggregation:*

```ruby
# Group VIP transactions to summarize sales and customer counts by location and gender
vip_summary = vip.groupby("CustomerID").agg({
    "net_sales": "sum",
    "Transaction_ID": "count",
    "Quantity": "sum",
    "Location": "first",
    "Gender": "first"
}).reset_index()

gender_summary = vip.groupby(["Location", "Gender"])["CustomerID"].nunique().reset_index()
gender_summary.rename(columns={"CustomerID": "Unique_Customers"}, inplace=True)
```

## 3. **Product Preferences of High-Value Buyers:**
   
A significant portion of VIP revenue was derived from **Apparel** and **Notebooks & Journals** categories. Conversely, **Nest products** showed negligible sales among VIP customers. Bulky orders predominantly originated from Chicago and California, aligning with their high transaction volumes.

*To understand product preferences, the charts below illustrate the **product categories driving VIP sales** and the **quantity sold to VIPs by category and location.***

![alt text](/img/vipsales_bycategory.png "Horizontal Bars") 
![alt text](/img/vip_quantiy_bycategory.png "Bar Plot")


## 4. **Critical Temporal Trends:**
   
Analysis of daily and weekly transaction patterns revealed significant anomalies:

- A notable **spike in high-value sales occurred in April.**
- **Most VIP transactions clustered on Thursdays and Fridays.**
- Crucially, **no VIP transactions occurred on Tuesdays**, highlighting a clear pattern for operational adjustments.

*These critical **temporal patterns**, including monthly spikes and daily variations, are clearly visualized in the **VIP Sales and Transactions Over Time ** and **VIP Transactions by Day of the Week** charts below.*

![alt text](/img/vip_sales_overtime.png "Line Graph")
![alt text](/img/vip_transactions_overtime.png "Line Graph")
![alt text](/img/transactions_by_dayofweek.png "Line Graph")

*Temporal trends were identified by grouping VIP transactions by date and day of the week:*

```ruby
# Analyze daily transaction counts and sales sums for VIPs
daily_counts = vip.groupby(vip["Transaction_Date"].dt.date).size().reset_index(name="transaction_count")
daily_sales = vip.groupby(vip["Transaction_Date"])["net_sales"].sum().reset_index()

# Determine transaction counts by day of the week
vip["DayOfWeek"] = vip["Transaction_Date"].dt.day_name()
day_counts = vip["DayOfWeek"].value_counts().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]).reset_index()
day_counts.columns = ["DayOfWeek", "TransactionCount"]
```

5. **Comprehensive KPI Comparison Across Segments:**
   
A final Power BI dashboard compared key KPIs across all four customer segments, offering a holistic view:

- **Loyalty and Premium groups** were the primary revenue drivers.
- Chicago and California led in net sales.
- Office products were the top category in quantity sold.
- Overall, more female customers than male across all segments, with a smaller proportion of one-time buyers in the Regular group.

This dashboard provided a centralized view for ongoing strategic monitoring.

*Finally, the **Business Performance and Segment Insights Dashboard** below offers a comprehensive, interactive overview of key KPIs across all customer segments, serving as a centralized strategic monitoring tool.*


![alt text](/img/ecommerce_dashboard.png "Dashboard")


# Business Impact:

This project provided Crimsonloop with a granular, data-driven understanding of their customer base, enabling them to:

- **Optimize Marketing Strategies:** Tailor campaigns with precision based on segment-specific preferences, locations, and purchasing habits, increasing conversion and retention.
- **Enhance Resource Allocation:** Optimize staffing and inventory management by anticipating high-value sales spikes (e.g., in April, on Thursdays/Fridays) and low activity periods (e.g., Tuesdays).
- **Drive Revenue Growth:** Focus efforts on high-potential segments and product categories, leading to more efficient sales efforts.
- **Improve Customer Experience:** Develop personalized offers and loyalty programs that resonate with different customer tiers.

This analysis empowered Crimsonloop to move beyond generic strategies, fostering a truly data-driven approach to customer relationship management and business growth.

# Technical Details & Full Codebase:

This section provides more technical details and serves as a gateway to the complete codebase.

The project leveraged Python's extensive data science ecosystem. For data manipulation and analysis, **Pandas and NumPy** were used. **Matplotlib, Seaborn, and Plotly Express** facilitated detailed data visualization and exploration, leading to the insights presented above. The final holistic KPI comparison was developed in **Power BI.**

For the full codebase, including all data preparation steps, detailed analytical scripts, and additional visualizations, please visit the dedicated GitHub repository: [Your GitHub Repo Link Here]

**Tools Used in Technical Implementation:** Python (Pandas, NumPy, Matplotlib, Seaborn, Plotly), Power BI, SQL
