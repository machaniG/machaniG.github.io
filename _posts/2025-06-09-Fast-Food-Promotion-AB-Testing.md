---
layout: post
title: Which Promotion Works? Fast Food Campaign Effectiveness
image: "/posts/fastfood-image.jpeg"
tags: [Python, A/B Testing]
---

In this post I conducted an A/B testing for a fast-food chain where I compared three versions of a marketing campaign of a new product to see which one has the greatest effect on sales. The aim was to help the fast-food chain decide which one of the three possible marketing campaigns they can use to promote their new product.

---

First I imported the required libraries.

```ruby
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
```

I started with visualizing the distribution of sales by promotion type and across the different markets.

## Sales Distribution
```ruby
ax = sns.boxplot(data = df, x = 'Promotion', y = 'SalesInThousands', palette = 'Set2', width = 0.6)
for location in ["top", "bottom", "right"]:
    ax.spines[location].set_visible(False)
plt.title('Sales Distribution by Promotion', weight = "bold", fontsize = 12)
plt.ylabel("Sales in Thousands")
plt.xlabel("")
ax.tick_params(bottom = False)

# Add stripplot for actual data points (optional)
sns.stripplot(data = df, y = "SalesInThousands", x = "Promotion", color = "black", alpha = 0.2, jitter = 0.2, size = 4)
plt.xlabel("")
plt.savefig("sales_dist_bypromotion.png")
plt.show()

#sales distribution across markets
ax = sns.barplot(df, x = "MarketSize", y = "SalesInThousands", hue = "Promotion", 
            errorbar = None, palette = "Set2", width = 0.6, gap = 0.1, dodge = True)
for location in ['right', 'top']:
    ax.spines[location].set_visible(False) 
plt.title("Distribution of Sales Across Market Sizes", weight = "bold", fontsize = 12)
plt.ylabel("Sales in Thousands")
plt.xlabel("")
plt.savefig("sales_bymarketsize.png")
plt.show()
```
![alt text](/img/sales_dist_bypromotion.png "Box Plot")
![alt text](/img/sales_bymarketsize.png "Grouped bar")

### Distribution of Promotions Across Market Sizes:

I then sought to understand how often was each promotion applied to the different markets
```ruby
pd.crosstab(df['Promotion'], df['MarketSize'], margins=True, margins_name="Total")
```
![alt text](/img/prom-frequency.png "Table")

In general, we see that we have fewer sales from promotion 2 across all the three markets. However, promotion 2 was applied less frequent only in smaller markets. 

For us to be sure which promotion is working, we have to carry out statistical analyses. We can't really tell by the look whether the differences we see in sales are statistically significant.

Therefor, I will do an anlysis of variance (ANOVA) to test which of the three promotions does actually drive sales. 

## ANOVA Test

```ruby
# One-way ANOVA
group1 = df[df['Promotion'] == 1]['SalesInThousands']
group2 = df[df['Promotion'] == 2]['SalesInThousands']
group3 = df[df['Promotion'] == 3]['SalesInThousands']

f_stat, p_val = f_oneway(group1, group2, group3)
print(f"ANOVA F-statistic: {f_stat:.3f}, p-value: {p_val:.3f}")
```
#### **ANOVA F-statistic: 21.953, p-value: 0.000**

The p-value (0.000) is below the typical threshold (0.05), meaning that at least one promotion group is statistically different from the others in terms of sales. But ANOVA only tells us that a difference exists, not where it is. I then did a post-hoc test which can tell us exactly which promotion is different from the others.

## Tukey's HSD  Post-Hoc Test
```ruby
tukey = pairwise_tukeyhsd(endog=df['SalesInThousands'],
                          groups=df['Promotion'],
                          alpha=0.05)
print(tukey)
```
![alt text](/img/post-hoc.png "Table")

From the post-hoc test, we can see that group 2 is significantly different from both group 1 and 3, p value < 0.001.

### Regression Modeling

I finally build a simple linear regression model to estimate the impact of each promotion on sales adjusting for week of campaign, market size, and age of the store. I plotted regression coefficients to show the direction and magnitude of the estimated impact.

```ruby
# Convert categorical variables
df['Promotion'] = df['Promotion'].astype('category')
df['MarketSize'] = df['MarketSize'].astype('category')

formula = 'SalesInThousands ~ C(Promotion) + week + AgeOfStore + C(MarketSize)'
model = smf.ols(formula=formula, data = df).fit()
```
```ruby
# Extract coefficients related to Promotion from the model
coef_df = model.params.filter(like='C(Promotion)').reset_index()
coef_df.columns = ['Promotion', 'Coefficient']

# Add the baseline intercept to show relative to Promotion 1
coef_df['Promotion'] = coef_df['Promotion'].str.extract(r'\[T\.(\d)\]')
coef_df['Promotion'] = 'Promotion ' + coef_df['Promotion']

# Plot
sns.barplot(x='Promotion', y='Coefficient', data=coef_df, palette='Set2')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Estimated Effect of Each Promotion on Sales')
plt.ylabel('Effect on Sales (in Thousands)')
plt.xlabel("")
plt.savefig("Regression_Coeffs.png")
plt.show()
```
![alt text](/img/Regression_Coeffs.png "bar graph")


## Business Interpretation

After performing post-hoc test and regression, we can see that Promotion 2 was associated with a statistically significant decrease of ~10.7K units in sales compared to the baseline (Promotion 1). Promotion 3 had no significant impact. We can conclude that:
- Promotion 2 has significantly lower sales than both Promotion 1 and Promotion 3.
- Promotion 1 and 3 are closer to each other, with no statistically significant difference.

#### **My Recommendation:**

The marketing manager should confidently choose Promotions 3 for large markets and promotion 1 for small and medium markets as the most effective campaigns.
