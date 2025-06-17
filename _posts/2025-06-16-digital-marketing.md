---
layout: post
title: Digital Marketing KPIs and Predicting Conversion
image: "/posts/marketing-image.png"
tags: [SQL, Machine Learning]
---

In digital marketing, multiple customers see a campaign; only some convert and some are already existing customers. Digital marketers rely on specific marketing metrics and Key Performance Indicators (KPIs) to optimize their strategies and improve campaign effectiveness. Digital marketing metrics and KPIs are ways to measure how well your online marketing efforts are working and insights from these metrics help the marketing manager to allocate marketing budgets more effectively, ensuring resources are invested in the most impactful areas.

In this notebook, I will analyze a few KPIs using SQL to see you whether we are achieving our marketing goals, generate a customer funnels and predict conversion using machine learning. 

---
First I loaded the data and: performed quality checks with pandas, converted column names to lower cases, created a database connection, and wrote the data into the database.

```ruby
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
```
```ruby
df = pd.read_csv("digital_marketing.csv")
df.isnull().sum().sort_values(ascending = False)
df.columns = [c.lower() for c in df.columns]
```
```ruby
# Database connection info
username = 'postgres'
password = 'password'
host = 'localhost'
port = '5432'
database = 'marketing'
# Create engine
engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')# Write DataFrame to database
df.to_sql('digital_marketing', engine, index=False, if_exists='replace')  
```
## Customer Funnel

I started by investigating the customer journey from seeing the ad through conversion for new customers only. I visualized the customer journey using a funnel.

```ruby
query3 = """
WITH funnel_new_customers AS (
    SELECT
        COUNT(DISTINCT CustomerID) AS visited,
        COUNT(DISTINCT CASE WHEN PagesPerVisit > 3 OR TimeOnSite > 1 THEN CustomerID END) AS engaged,
        COUNT(DISTINCT CASE WHEN ClickThroughRate > 0 THEN CustomerID END) AS clicked_ad,
        COUNT(DISTINCT CASE WHEN EmailOpens > 0 THEN CustomerID END) AS opened_email,
        COUNT(DISTINCT CASE WHEN Conversion = 1 THEN CustomerID END) AS converted
    FROM digital_marketing
    WHERE previouspurchases = 0
)
SELECT * FROM funnel_new_customers;
"""
df4 = pd.read_sql(query3, engine)

```ruby
# Sometimes the SQL returns columns in lowercase or underscores, so I mapped them manually:
stages = ['Visited', 'Engaged', 'Clicked Ad', 'Opened Email', 'Converted']
counts = [
    df4['visited'][0],
    df4['engaged'][0],
    df4['clicked_ad'][0],
    df4['opened_email'][0],
    df4['converted'][0]
]
#plot the funnel
fig = go.Figure(go.Funnel(y=stages, x=counts, textinfo="value+percent initial", marker=dict(color="#a9ba9d"), textfont={"size": 16}))
fig.update_layout(title="New Customers Funnel", width=600, height=400, font=dict(size=16, family="Arial"), margin=dict(t=80, l=50, r=50, b=50))
fig.show()
```
![alt text](/img/new_customerfunnel.png "funnel")

I found we have a very good conversion of new first time customers with a conversion rate of 77%, meaning that a good percentage of customers we showed our ads completed the desired action of making a purchase. This prompted me to visualize the journey for customers who did not convert to help to identify drop-off points in acquisition and later we can investigate why they are not converting.

**Where in the journey are we losing potential customers?**

```ruby
query2 = """
WITH funnel_not_converted AS (
    SELECT
        COUNT(DISTINCT CustomerID) AS visited,
        COUNT(DISTINCT CASE WHEN PagesPerVisit > 3 OR TimeOnSite > 2 THEN CustomerID END) AS engaged,
        COUNT(DISTINCT CASE WHEN ClickThroughRate > 0 THEN CustomerID END) AS clicked_ad,
        COUNT(DISTINCT CASE WHEN EmailOpens > 0 THEN CustomerID END) AS opened_email,
        COUNT(DISTINCT CASE WHEN Conversion = 0 THEN CustomerID END) AS not_converted
    FROM digital_marketing  
)
SELECT * FROM funnel_not_converted;
"""
df3 = pd.read_sql(query2, engine)

stages = ['Visited', 'Engaged', 'Clicked Ad', 'Opened Email', 'not_Converted']
counts = [df3['visited'][0], df3['engaged'][0], df3['clicked_ad'][0], df3['opened_email'][0], df3['not_converted'][0]]

fig = go.Figure(go.Funnel(y=stages, x=counts, textinfo="value+percent initial", textfont={"size": 16}  # Font size of text inside bars))
fig.update_layout(title="Not Converted Customer Funnel", width=600, height=400, font=dict(size=16, family="Arial"), margin=dict(t=80, l=50, r=50, b=50))
fig.show()
```
![alt text](/img/notconverted_funnel.png "not converted funnel")

I found that the drop-off point is at the conversion stage. I followed this up by comparing the click through rate (CTR) with conversion rate. 

I visualized CTR side by side with the conversion rate and noticed that the CTR is consistently higher than the conversion rate across all advertising channels; meaning the ads are attracting clicks, but a small percentage of customers are not completing the desired action. This implies that the ad copy and targeting might be effective in grabbing attention, but the offer might not be compelling enough to drive conversions for this group of customers. 

```ruby
#define a reusable query function
def run_query(query):
    return pd.read_sql(query, engine)

df_rates = run_query("""
    SELECT 
    CampaignChannel,
    CAST(AVG(ClickThroughRate) AS DECIMAL(5,4)) AS ctr,
    CAST(AVG(ConversionRate) AS DECIMAL(5,4)) AS conversion_rate
    FROM digital_marketing
    GROUP BY CampaignChannel
    ORDER BY CampaignChannel;
""")

# Melt for side-by-side bar plot
df_melted = df_rates.melt(id_vars='campaignchannel', value_vars=['ctr', 'conversion_rate'], var_name='Metric', value_name='Rate')

ax = sns.barplot(data=df_melted, x='campaignchannel', y='Rate', hue='Metric', palette='viridis', width = 0.7, gap = 0.05, dodge = True)
for location in ["top", "right", "bottom"]:
    ax.spines[location].set_visible(False)
ax.tick_params(bottom=False)
plt.title("Average CTR and Conversion Rate by Campaign Channel")
plt.ylabel("Rate (%)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("ctr_conversionrate")
plt.show()
```
![alt text](/img/ctr_conversionrate.png "ctr-cr bar plot")

**Marketing Cost-Effectiveness**

Next I wanted to understand how efficient was the advertising so I calculated and visualized the cost per click (CPC) by advertising channel. 

```ruby
#  Cost Per Click by CampaignChannel
df_cpc = run_query("""
    SELECT 
     CampaignChannel,
     SUM(AdSpend) AS total_ad_spend,
     SUM(ClickThroughRate * WebsiteVisits) AS total_clicks,
     CASE 
         WHEN SUM(ClickThroughRate * WebsiteVisits) > 0 THEN
             CAST(SUM(AdSpend) AS DECIMAL(10,2)) / SUM(ClickThroughRate * WebsiteVisits)
         ELSE NULL
     END AS CPC
   FROM digital_marketing
   GROUP BY CampaignChannel
   ORDER BY CPC;
""")

fig = px.bar(df_cpc, x='campaignchannel', y='cpc', color='campaignchannel', text='cpc', title='Cost Per Click (CPC) by Campaign Channel',
            color_discrete_sequence=["#717171", "#482878", "#d2a990", "#31688e", "#cb6817"])
fig.update_traces(texttemplate='€%{text:.2f}', textposition='outside')
fig.update_layout(yaxis_title='CPC in Dollars', xaxis_title='Campaign Channel', height=500, width = 700)
fig.show()
```
![alt text](/img/cpc_channel2.png "cpc bar plot")
![alt text](/img/cr_channel.png "cr bar plot")

I noticed that we paid more each time an ad delivered through referrals was clicked compared to ads placed on social media. Running ads through social media is not only cheap but the most efficient form of marketing since the conversion rate is also slightly higher for social media ads.

The cost per click is generally high, but **how much does it cost to actually acquire a new customer?**

###  Customer Acquisition Cost (CAC) 
CAC is a vital digital advertising performance metrics it provides insights into the efficiency and sustainability of marketing and sales strategies. I calculated CAC vs customer volume by Campaign Type.
```ruby
cac_df = run_query("""
    SELECT
    CampaignType,
    SUM(AdSpend) AS total_ad_spend_on_new_converts,
    COUNT(DISTINCT CustomerID) AS new_customers_acquired,
    CAST(SUM(AdSpend) * 1.0 / COUNT(DISTINCT CustomerID) AS NUMERIC(10,2)) AS CAC
    FROM digital_marketing
    WHERE PreviousPurchases = 0 AND Conversion = 1
    GROUP BY 1
    ORDER BY 4 DESC;
""")

fig = px.scatter(cac_df, x='campaigntype', y='cac', size='new_customers_acquired', color='campaigntype',
                 color_discrete_sequence=["#909bc7", "#6ece58", "#c59dc1", "#8dbeca"],
                 title='CAC vs Customer Volume by Campaign', size_max=60)
fig.update_layout(height=500, width = 700, xaxis_title="", yaxis_title="CAC in Dollars")
fig.show()
```
![alt text](/img/cac.png "bubble chart")

After performing more analysis, I did research and found that Mailchip suggests that 10 AM is the most optimal time to send out newsletters/emails to subscribers. I used that as  my recommendation to The Column.

![alt text](/img/posts/Opens_Analysis.jpg "Opens Analysis")

![alt text](/img/posts/Clicks_Analysis.jpg "Clicks Analysis")

![alt text](/img/posts/Lifetime_Column.jpg "Lifetime Performance")

Since we have a good percentage of users who completed the desired action of making a purchase. We cannot calculate ROI because we do not have access to the revenue generated from this marketing
