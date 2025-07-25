---
layout: post
title: Optimizing Marketing KPIs and Predicting Conversion
image: "/posts/marketing.png"
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
# KPI

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
fig = go.Figure(go.Funnel(y=stages, x=counts, textinfo="value+percent initial",
                            marker=dict(color="#a9ba9d"), textfont={"size": 16}))
fig.update_layout(title="New Customers Funnel", width=600, height=400, font=dict(size=16, family="Arial"),
                     margin=dict(t=80, l=50, r=50, b=50))
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

## Click Through Rate and Conversion Rate

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

## Marketing Cost-Effectiveness

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

##  Customer Acquisition Cost (CAC) 
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

I found that campaigns aimed at customer conversion are cheap and effective because of low CAC and more customers are acquired, making them the best type. Awarenes campaigns on the other hand have high CAC and even though we are acquiring many customers with this type of campaigns, they are expensive! 

## Total Ad Spend

Understanding total ad spend helps a business to allocate resources effectively across different campaigns and channels. After analyzing total ad spend by campaign type and channel, I found that, although conversion campaigns were the cheapest and effective in terms of customer acquisition costs, we spent more money on conversion campaigns in total and less dollars on retention ones. Social media marketing again is the leading in terms of cost-efficiency. Retention campaigns run through social media might be more efficient.
```ruby
df = run_query("""
    SELECT DISTINCT campaigntype AS "Campaign name",
    campaignchannel AS "Campaign channel",
    SUM(adspend) AS "Ad Spend"
    FROM digital_marketing
    GROUP BY 1, 2
    ORDER BY 3 DESC;
""")

fig = px.bar(df, x = "Campaign channel", y = "Ad Spend", color='Campaign channel', text = 'Ad Spend', title='Ad Spend in Dollars by Campaign Channel',
            color_discrete_sequence=["#cb6817", "#d2a990", "#482878", "#31688e", "#717171"])
fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
fig.update_layout(yaxis_title='Ad Spend in Dollars', xaxis_title='Campaign Channel', height=500, width = 700)
fig.show()
```
![alt text](/img/ad_spend.png "ad spend")

## Social Media Engagement

I finally wanted to understand how our target audience interact with our ad content. This metric is crucial for shaping content strategies and understanding what drives interaction. I visualized social shares by campaign type using a pie chart. I found that conversion campaigns have high engagement, meaning that these posts resonate well with the audience, encouraging more visibility and reach through platform algorithms. 
```ruby
df_social = run_query("""
    SELECT 
        CampaignType,
        SUM(SocialShares) AS total_social_shares
    FROM digital_marketing
    GROUP BY CampaignType
    ORDER BY total_social_shares DESC;
""")

explode_index = df_social['total_social_shares'].idxmax()
pull = [0.1 if i == explode_index else 0 for i in range(len(df_social))]
fig = px.pie(df_social, names='campaigntype', values='total_social_shares', title='Social Media Engagement by Campaign Type', hole=0.3,
            color_discrete_sequence=["#909bc7", "#6ece58", "#c59dc1", "#8dbeca"])  
fig.update_traces(pull=pull, textinfo='label+percent+value')
fig.update_layout(height=500, width = 500)
fig.show()
```
![alt text](/img/social_shares.png "pie chart")

# Machine Learning: Predicting Conversion

After analysing the key KPIs, I decided to use machine learning approach to predict customer conversion. I used Random Forest Regression from scikit-learn and XGBRegressor from XGBoost libraries. I trained the two models, compared metrics and found that:
- Random Forest catches almost all converters (recall = 1.0%) but makes more false positive predictions.
- XGBoost makes fewer false positive predictions, so its precision is higher (92%), but misses more actual converters (recall = 99%).

So Which Is Better?

- As a marketing team, if we are trying to target converters precisely, and the campaign is costly, then we should go with **XGBoost** becuase **fewer wrong people get targeted**. The Precision and F1 scores are better.

- However, when we care more about not missing any real converters and we want to do a broad retargeting, then we should go with **Random Forest** becuase it has the best recall, but it is riskier in terms of wasted ad spend. There are **more False Positives**.

```ruby
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
```
## Preprocessing pipeline
```ruby
X = df.drop(columns=["Conversion", "CustomerID"], axis = 1)
y = df["Conversion"]

# Categorical columns
cat_cols = ["Gender", "CampaignChannel", "CampaignType"]
num_cols = [col for col in X.columns if col not in cat_cols]

# Preprocessor
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), num_cols),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])
```
## Train Random Forest Model
```ruby
pipeline_rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
```
## Train Model: XGBoost
```ruby
pipeline_xgb = Pipeline([
    ("preprocess", preprocessor),
    ("model", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

pipeline_xgb.fit(X_train, y_train)
y_pred_xgb = pipeline_xgb.predict(X_test)

print("XGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
```

## Create a Comparison Plot of Precision, Recall, and F1 Score between models
```ruby
# metrics data
metrics_data = {
    "Model": ["Random Forest", "XGBoost"],
    "Precision": [0.89,  0.92],
    "Recall": [1.0,  0.99],
    "F1 Score": [0.94, 0.95]
}

df_metrics = pd.DataFrame(metrics_data)

#Melt the DataFrame into long format for seaborn
df_melted = df_metrics.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1 Score"], var_name="Metric", value_name="Score")

# Plot using seaborn
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
plt.title("Model Comparison: Precision, Recall, and F1 Score", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.xlabel("Evaluation Metric", fontsize=12)
plt.ylim(0.85, 1.02)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("model_comparison_metrics.png", dpi=300)
plt.show()
```
![alt text](/img/model_comparison_metrics.png "model metrics")

## Confusion Matrix
```ruby
#plot Random Forest confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Converted", "Converted"], yticklabels=["Not Converted", "Converted"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusionmatrix.png", dpi=300)
plt.show()

#plot XGBoost confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("xgb_confusionmatrix.png", dpi=300, bbox_inches="tight")
plt.show()
```
![alt text](/img/confusion_matrix.png "confusion matrix")

## Top Drivers of Conversion 

I extracted the top predictors the model used to determine whether a customer converts using **feature importances_**. I visualized the top 10 drivers of conversion to inform where marketing team should focus their efforts. 
```ruby
# Fit preprocessing separately for feature names
X_preprocessed = preprocessor.fit_transform(X_train)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_preprocessed, y_train)

# Get feature names
encoded_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
all_feature_names = list(encoded_feature_names) + num_cols

importances = xgb_model.feature_importances_

# Map to names
feat_imp = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature", palette="viridis")
for location in ['top', 'right', 'left', 'bottom']:
    ax.spines[location].set_visible(False)
ax.tick_params(bottom=False)
ax.tick_params(left=False)
plt.ylabel("")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("important_features.png")
plt.show()
```
![alt text](/img/important_features.png "horizontal bars")

I deduced that;

| Rank | Feature                           | Business Insight                                                                                          |
| ---- | --------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 1    | **PreviousPurchases**             | Returning customers are more likely to convert. Loyalty matters, therefore, consider retention campaigns. |
| 2    | **Age**                           | Certain age groups are more likely to convert. Segment targeting could improve results.                   |
| 3    | **CampaignType\_Retention**       | Retention campaigns are effective. We may get strong ROI from targeting existing users.                   |
| 4    | **CampaignChannel\_Email**        | Email campaigns are highly predictive so email strategy is a powerful lever.                              |
| 5    | **CampaignType\_Conversion**      | Conversion campaigns are likely driving actual purchases compared to awareness only.                      |
| 6    | **CampaignChannel\_PPC**          | Paid search has significant impact.                                                                       |
| 7    | **CampaignType\_Awareness**       | Even awareness campaigns play a role in conversion possibly as one of the top-funnel influencers.         |
| 8    | **CampaignChannel\_Social Media** | Social media plays a meaningful role, likely influencing early engagement.                                |
| 9    | **Income**                        | Higher income may correlate with conversion; consider tailoring offers.                                   |
| 10   | **CampaignChannel\_Referral**     | Referrals can bring in high-intent users. Strengthen referral programs.                                   |


## **Recommendations**

1. Double Down on What Works e.g.,
   - Prioritize email, retention, and conversion campaigns
   - Ensure PPC and referral traffic are well-optimized
     
2. Segment Strategically
   - Age and income influence behavior. Therefore, tailor content and offers accordingly

3. Test Combinations
   - Combine top channels or campaign types (e.g., Email + Retention) to test synergy

4. Investigate Weak Channels
   - Examine features with low importance to understand whether they are wasteful or poorly used.
     
5. Use domain knowledge and AB testing to validate strategies.





