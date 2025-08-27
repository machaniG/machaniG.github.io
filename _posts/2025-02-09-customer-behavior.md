---
layout: post
title: Crimsonloop Customer Segmentation and Behavior Analysis
image: "/posts/customers.png"
tags: [Python, Power BI]
---

# Executive Summary

This analysis empowered Crimsonloop to move beyond a one-size-fits-all strategy by segmenting our diverse customer base into four distinct tiers. My findings reveal that two customer segments, the Premium and Loyalty tiers, drive 74% of our revenue. Our high value segment, customers who have a single-transaction value over $1,000, represent only 2.8% of our customers but contributes 13% of revenue.By focusing on these high-value segments, we can develop targeted, high-impact marketing strategies, ultimately enhancing customer retention and driving significant revenue growth.

## My Approach

After preprocessing and integrating multiple datasets, I used rule-based segmentation to define four distinct customer tiers based on their transaction values. This approach allowed for a clear, immediate understanding of our customer base, paving the way for targeted marketing and resource allocation. The analysis culminated in a set of dashboards designed to provide a real-time insights on our customer base. 


# Key Discoveries & Actionable Insights:

## 1. **Unveiling High-Value Customers & Segmentation:**
   
Initial analysis revealed a right skewed distribution of sales and a "long tail" between the 99 percentile and the maximum transaction value, signalling high-value customer opportunities. This prompted me to perform customer segmentation and analyze their unique behaviors and preferences. Click [here](https://github.com/machaniG/machaniG.github.io/blob/master/notebooks/Crimsonloop%20Customer%20segmentation%20%26%20behavior%20analysis.ipynb) for technical implementation of this project.

The Elite segment is made of 41 customers out of 1468 of our customers, representing only 2.7%. They are returning customers who buy products worth at least $1,000 per transaction and generated 13.1% of revenue in 2019. As illustrated in the *sales dashboard below*, the average transaction value for Elite customers is $98.4 while that of the Premium and Loyalty tiers are $72.3 and $63.5, respectively. While our two leading customer segments, the Premium and Loyalty tiers, drive 38.1% and 36.4% of our revenue respectively, they have 12 and 6 times more customers than the Elite tier. This implies that the Elite tier represent a small but extremely valuable segment if targeted with personalized offers and exclusive services can drive significant revenue growth. The Premium and Loyalty customers are critical for sustaining and growing the business. Understanding their preferences, purchase patterns, and demographics (e.g., location, gender) can help to tailor marketing and retention strategies.

![alt text](/img/sales_insights.png "sales dashboard")


## 2. **Uncovering Product Preferences:**

Analysis of product category revealed that a single category, **Nest-USA**, **generates 55.1% of our revenue**. However, we sold more Office products followed by Apparel and Drinkware. Office category are our most selling products, leading with 88k units sold but generates only 6% of total revenue while  Nest-USA is number 5 in terms of units sold. This revenue gap is understandable because the average unit cost for Nest-USA is $124.3 while that of Office products is $3.8. The dashboard and chart below show Office, Apparel, Drinkware, Nest, Nest-USA, Lifestyle, and Bags are our most selling products. The inventory manager can leverage this insights for procurement to balance supply and demand.

![alt text](/img/category_insights.png "category dashboard")

<img width="1535" height="842" alt="image" src="https://github.com/user-attachments/assets/76f373f7-05bb-4e8f-934b-7b3d5c5cc44b" />


## 3. **Geographic and Gender Insights:**

Most of our revenue come from California and Chicago and the two states accounted for 65% of total revenue in 2019. However, as seen in the dashboards below, the two regions have 920 out of our 1468 customers and more than half of our high value customers, that is; 28 Elite, 159 Premium, and 313 Loyalty customers. 


## 4. **Critical Temporal Trends:**

After analyzing temporal trends over time, I uncovered a notable spike of Elite sales in April. A deep dive into the Elite transactions further revealed that such high-value transactions cluster on Thursdays and Fridays, and crucially, **no Elite transaction on Tuesdays**; highlighting a clear pattern for operational adjustments.  

Apparel and Notebooks & Journals are the leading categories sold to elite customers

## 5. **Product Preferences of High-Value Buyers:**
   
A significant portion of Elite revenue derive from **Apparel** and **Notebooks & Journals** categories. Conversely, **Nest products** showed negligible sales among Elite customers. Bulky orders predominantly originated from Chicago and California, aligning with their high transaction volumes.

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

For the full codebase, including all data preparation steps, detailed analytical scripts, and additional visualizations, please visit the dedicated GitHub repository: [([https://github.com/machaniG/machaniG.github.io/blob/master/notebooks/Crimsonloop%20Customer%20segmentation%20%26%20behavior%20analysis.ipynb](https://github.com/machaniG/machaniG.github.io/blob/master/notebooks/Crimsonloop%20Customer%20segmentation%20%26%20behavior%20analysis.ipynb))]

**Tools Used in Technical Implementation:** Python (Pandas, Scikit-Learn, NumPy, Matplotlib, Seaborn, Plotly), Power BI
