---
layout: post
title: Digital Marketing Analytics & Predictive Modeling
image: "/posts/marketing.png"
tags: [SQL, Machine Learning]
---

## Executive Summary

My analysis provides a data-driven roadmap to optimize marketing spend and ensure we acquire new customers without overspending. The key finding is that while our overall funnel is healthy, there is a significant opportunity to optimize our budget. By shifting resources to our most efficient campaigns, we can maximize our return on investment and drive more profitable, sustainable growth.

**Goal:** To enhance digital marketing effectiveness and acquire new customers without overspending.

**Problem:** The client, Velteva, was running multiple digital marketing campaigns but lacked a clear, data-driven understanding of their effectiveness and profitability. The goal was to acquire new customers without overspending.

**Solution:** I developed a comprehensive analytics solution. First, using SQL and Python, I diagnosed the marketing funnel, calculating key performance indicators (KPIs) like Customer Acquisition Cost (CAC), Cost Per Click (CPC), conversion rates, and Social Shares. I then built and compared two machine learning models (XGBoost and Random Forest) to predict customer conversion. The best-performing model achieved a 99% recall and identified the top drivers of conversion.

**Impact:** My analysis provided a clear, actionable roadmap. I identified highly efficient campaigns and diagnosed the tactical reasons for low performance in others. The project's findings now serve as a foundation for strategic budget reallocation and more profitable customer acquisition for the Velteva marketing team.

## Our Funnel Health

Funnel analysis revealed that our campaigns are effective at turning new users into customers, with a strong 77.3% conversion rate. Our existing customers are also highly engaged and loyal, with a stellar 88.9% conversion rate. The challenge, however, lies with our non-converting customers, for whom we must understand the reasons for drop-off. 

As illustrated below, the most significant drop-off occurs at the final Converted stage. This is a critical insight as it pinpoints the exact stage where the marketing strategy needs to be re-evaluated.

<img width="700" height="600" alt="conversion_rates" src="https://github.com/user-attachments/assets/d8c72e4d-db1b-4420-ad9e-116f02aaeecf" />


<img width="976" height="440" alt="image" src="https://github.com/user-attachments/assets/40415d4d-7900-4d90-9e33-e12f21794a9b" />



## Spend vs. Conversion Efficiency

To understand where our budget is most effective, I compared our ad spend allocation to our conversion results and found a disconnect between what we spend and what we get.
- Our Conversion and Retention campaigns are our most efficient. They capture a significant portion of our conversions with a proportionally smaller ad spend, highlighting their high ROI.
- Conversely, our Awareness and Consideration campaigns, while crucial for brand building, are currently inefficient from a conversion standpoint. They consume a large portion of the budget but contribute a smaller percentage of our conversions.

The grouped bar chart below shows a clear disconnect between what we spend and what we get.

![alt text](/img/spend_ratio_conversion_ratio.png "grouped bar")


## Cost & Engagement Diagnostics

**Customer Acquisition Cost (CAC):**

My analysis revealed a significant difference in CAC.
	- Awareness: $5,791
	- Consideration: $5,459
    - Retention: $5,098
	- Conversion: $4,825

This confirmed my finding that Conversion campaigns have the lowest CAC and therefore are the most cost-effective way to acquire new customers while Awareness campaigns are the most expensive.

This chart below clearly shows that conversion and retention campaigns are the most efficient with low CAC and high volume of new customers acquared (191 and 160 customers, respectively).

![alt text](/img/cac_spend_campaign.png "bubble chart")


**Cost Per Click (CPC):**

By looking at CPC, I was able to diagnose whether the problem is at the top of the funnel (expensive clicks) or mid-funnel (low conversion rate). My key findings were: 

- Awareness campaigns have high CAC and CPC ($1,322.7), implying that the problem is at the top of the funnel. The cost of attracting a single click is too expensive.
- Consideration campaigns have a high CAC, but its CPC ($1,305.3) is the lowest. This suggests that the problem is mid-funnel. The ads are getting cheap clicks, but the landing page or offer is not converting the visitors.

- Conversion campaign has a low CAC but high CPC ($1,316.8), indicating that the ads are attracting highly qualified clicks that are very likely to convert. The high cost per click is justified because each click is incredibly valuable, leading to a cheap customer acquisition. This is a classic example of high-quality traffic outweighing high cost. It suggests that:

     - Our ads are reaching the right audience: The targeting is precise, and the people who are clicking are the exact customers we want to acquire.

     - The high CPC is likely due to a competitive bidding environment. We are paying a premium to acquire these clicks because other advertisers also want them.


**Social Shares:**
This metric is crucial for our brand-building efforts. The data reveals that Conversion and awareness campaigns resonate well with the audience due to high social shares, over 100k shares, indicating a successful brand message and strong audience connection. While our Awareness campaigns are financially inefficient, they have been successful at generating social buzz, a key top-of-funnel metric that may contribute to future growth. 


## Predictive Modeling: Understanding Our Most Valuable Customers

Moving beyond what has already happened, I built two machine learning models, Random Forest and XGBoost, to predict the likelihood of conversion.

**Model Performance:**

Both models performed exceptionally well, with XGBoost showing a slight edge. Its precision of 92% and recall of 99% indicate it is highly effective at identifying the users most likely to convert.

**Top Drivers of Conversion:**

To make this model actionable, I analyzed the features that most influence conversion. The bar chart below shows the top 10 drivers, with Previous Purchase and Age leading the way. This gives us a clear roadmap for where to focus our efforts to increase our conversion rate.

![alt text](/img/important_features.png "bar chart")


## My Recommendations

1. **Reallocate Budget:** Shift a portion of our ad spend from less efficient campaigns (Awareness and Consideration) to our top performers (Conversion and Retention).
2. **Optimize Funnel Drivers:** Focus on improving the top drivers of conversion identified by our model, such as increasing website visits and optimizing ad spend for higher-value campaigns.
3. **Refine Targeting:** Use the insights from our model to create lookalike audiences that share the traits of our most valuable customers, leading to even more profitable campaigns."



*For technical implementation of this analysis, please check this notebook here:* 
[Marketing Analytics Notebook](https://github.com/machaniG/machaniG.github.io/blob/master/notebooks/Marketing_Analytics.ipynb)




