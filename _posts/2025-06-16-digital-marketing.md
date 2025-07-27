---
layout: post
title: Optimizing Marketing Spend and Predicting Customer Conversion for Velteva Naturals
image: "/posts/marketing.png"
tags: [SQL, Machine Learning]
---

This project showcases a data-driven approach to enhancing digital marketing effectiveness. I partnered with Velteva Naturals, a growing consumer brand, to address a critical challenge: optimizing their marketing spend to maximize customer acquisition and conversion. The goal was to move beyond generic metrics and provide actionable insights that would inform future strategy and prove a tangible return on investment.

**Methodology:**

I began by performing a diagnostic analysis of key performance indicators (KPIs) using SQL to understand the current state of marketing efforts. This diagnostic phase was followed by building and comparing two machine learning models (Random Forest and XGBoost) to predict customer conversion. The final step was to translate these technical findings into strategic business recommendations that Velteva Naturals could immediately implement.

## Part 1: Deconstructing the Customer Journey with SQL

To understand where Velteva Naturals was succeeding and where potential customers were dropping off, I began by mapping the customer journey with a funnel analysis.

**Funnel Analysis: Identifying Key Drop-off Points**

The funnel for new customers showed strong performance, with a significant **77% conversion rate** from initial ad view to purchase. This indicates that our initial campaigns are highly effective at driving conversions.

**Insight:** While initial conversion is high, understanding drop-offs is crucial for improving overall acquisition.

*The code below shows how this was implemented using SQL and Python (Pandas, Plotly:*

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
fig.update_layout(title="New Customers Funnel", width=600, height=400, font=dict(size=16, family="Arial"),
                     margin=dict(t=80, l=50, r=50, b=50))
fig.show()
```

![alt text](/img/new_customerfunnel.png "funnel")


To identify where we were losing potential customers, I analyzed the journey of those who did not convert. The most significant **drop-off point occurred at the final conversion stage.**

**Hypothesis:** This finding is critical as it suggests that while our ads and initial emails are effective at getting attention, there may be friction on the landing page or a disconnect between the ad's promise and the final offer.

*I implemented this as shown in the query below:*
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

This observation led me to investigate the effectiveness of our ad copy by comparing the click-through rate (CTR) with the conversion rate (CR).

**Click Through Rate (CTR) and Conversion Rate (CR)**

I noticed that CTR is consistently higher than the conversion rate across all advertising channels. This means our ads successfully attract clicks, but a lower percentage of customers complete the desired action of making a purchase.

**Insight:** This implies that our ad copy and targeting are effective, but the offer or the landing page experience may not be compelling enough to drive conversions for some customers. 

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

#plot with seaborn and matplotlib
```
![alt text](/img/ctr_conversionrate.png "ctr-cr bar plot")


**Marketing Cost-Effectiveness**

To understand the financial implications of our campaigns, I calculated the Cost Per Click (CPC) and Customer Acquisition Cost (CAC) by channel and campaign type.

The CPC analysis revealed that social media marketing is the most cost-effective channel per click. Referrals, while a valuable channel, have a higher CPC, and require careful monitoring to ensure they remain profitable.


**Insight:** Social media campaigns are not only cheap on a per-click basis but also highly efficient, as they have a higher conversion rate compared to other channels. This presents a powerful opportunity for growth. 

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
#visualize using plotly express
```
![alt text](/img/cpc_channel2.png "cpc bar plot")

![alt text](/img/cr_channel.png "cr bar plot")

Next, I calculated the Customer Acquisition Cost (CAC) to see how much it truly costs to acquire a new customer.

**CAC Analysis:**

- **Conversion campaigns** are the most efficient, acquiring customers at the lowest cost.
- **Awareness campaigns** have a high CAC, meaning that while they bring in a large volume of customers, they do so at a significantly higher cost.

**Insight:** This finding is a key strategic takeaway. It highlights the importance of balancing top-of-funnel awareness campaigns with more targeted conversion campaigns to maintain a healthy budget and acquisition pipeline.

*I used the query below to calculate CAC.*

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


## Budget Allocation Strategy: A Call for Optimization

The previous analysis showed that conversion campaigns have the lowest CAC, making them the most efficient way to acquire new customers. However, an analysis of the total budget allocation reveals a strategic mismatch: a disproportionately high amount of the ad budget was allocated to awareness campaigns, which have a significantly higher CAC.

**Insight:** This proves that Velteva is spending the most money on its least efficient campaigns. An optimized budget should shift funds from expensive awareness campaigns to more profitable conversion and retention campaigns, which are demonstrably more effective.

Since we lack a revenue column, I compared the percentage of total conversions each campaign type generates against the percentage of the total budget it consumes to understand resource allocation efficiency. This clearly showed which campaign types are "pulling their weight" and which ones are not.

```ruby
#percentage of total conversions each campaign type generates against the percentage of the total budget it consumes
total_metrics_df = run_query("""
             SELECT
             CampaignType,
             SUM(adSpend) AS "AdSpend",
             COUNT(DISTINCT CASE WHEN Conversion = 1 THEN CustomerID END) AS "TotalConversions"
             FROM digital_marketing
             GROUP BY CampaignType
""")

#total spent ratio per campaign type
total_metrics_df['Spend_Ratio'] = total_metrics_df['AdSpend'] / total_metrics_df['AdSpend'].sum()

#conversion ratio per campgain type as a ratio of total conversion
total_metrics_df['Conversion_Ratio'] = total_metrics_df['TotalConversions'] / total_metrics_df['TotalConversions'].sum()

# melt the df to have the ratios in one column
plot_df = total_metrics_df.melt(id_vars='campaigntype', value_vars=['Spend_Ratio', 'Conversion_Ratio'], var_name='Metric',
    value_name='Ratio')
# Visualize the ratios in a bar chart using plotly express
```
![alt text](/img/conversion_spend_ratios.png "conversion-spend ratios")



## Part 2: Predicting Conversion with Machine Learning

To move beyond historical analysis and gain a predictive edge, I built a machine learning model to identify which specific customer attributes drive conversion. This approach allows the marketing team to target high-potential customers with precision, preventing wasted ad spend.

**Model Selection: Random Forest vs. XGBoost**

I trained two popular classification models, Random Forest and XGBoost, to predict customer conversion. I compared their performance to understand which model was best suited for different business objectives.
- **Random Forest:** This model has a near-perfect recall (1.0), meaning it successfully identifies almost all actual converters. However, it also makes more false positive predictions, which could lead to wasted ad spend on customers who don't convert.
- **XGBoost:** This model has a higher precision (92%), meaning that when it predicts a conversion, it is more likely to be correct. It is a more conservative model that minimizes costly false-positive errors but misses a few more actual converters (recall of 99%).

*Models were trained as shown below:*
```ruby
#import libraries
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
### Preprocessing pipeline
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

### Train Random Forest Model
```ruby
pipeline_rf = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
```

### Train Model: XGBoost
```ruby
pipeline_xgb = Pipeline([
    ("preprocess", preprocessor),
    ("model", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])

pipeline_xgb.fit(X_train, y_train)
y_pred_xgb = pipeline_xgb.predict(X_test)

print("XGBoost Results:")
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
```

**Strategic Conclusion:**

The choice between these models depends on the specific campaign goal.
- For **high-cost, highly targeted campaigns**, XGBoost is the superior choice. Its high precision ensures that marketing budget is spent on the most likely converters, minimizing waste.
- For **broad retargeting campaigns** where not missing a single potential customer is the priority, Random Forest is the better option due to its higher recall.

![alt text](/img/model_comparison_metrics.png "model metrics")

![alt text](/img/confusion_matrix.png "confusion matrix")

## Top Drivers of Conversion 

Using the XGBoost model, I extracted the top predictors of conversion *as shown below*. This analysis provides a clear roadmap for where the marketing team should focus its efforts. 

```ruby
# Fit preprocessor separately for feature names
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

# Plot with seaborn and matplotlib
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




## Part 3: Actionable Recommendations for Velteva Naturals

Based on the KPI analysis and predictive modeling, I provided the following strategic recommendations to Velteva Naturals to optimize their marketing efforts:

- **Prioritize and Double Down on What Works:** Increase investment in the most impactful channels and campaign types. This includes email, retention, and conversion campaigns. The analysis shows these have the highest ROI and are the strongest drivers of conversion.
- **Strategic Segmentation:** Leverage the insights from the predictive model to tailor content and offers based on age and income demographics, which were identified as top predictors of conversion.
- **Optimize the Final Funnel:** Address the conversion funnel's drop-off point by running A/B tests on landing pages and product offers to ensure they are compelling enough to convert engaged visitors into customers.
- **Implement Predictive Modeling for Budget Allocation:** Use the XGBoost model to inform budget allocation for high-cost campaigns, ensuring resources are spent with high precision on the most likely converters.




