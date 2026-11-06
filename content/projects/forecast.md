---
title: Demand forecasting for a grocery chain
order: 3
who: Retail · 120 stores
metric: −1.8% forecast bias → 11.3% MAPE weekly 
cat: Employer
svc: ml
time: 6 months
stack: Python, LightGBM, scikit-learn, FastAPI, Docker
repo: https://github.com/machaniG/retail-demand-forecasting
kind: bars
seed: 3
---

## The problem

Spreadsheet forecasts caused stock-outs on fresh products and heavy waste.

## What I did

Hierarchical forecasting with promotions, weather, holiday, calendar, momentum, and lag features, backtested across 5 months and iteratively optimized customized loss functions for the client's dataset.


## The result

Model moved from overpredicting sales by nearly 14% to a -1.8% near-negligible under-prediction which is below ±5% industry benchmark, eliminating tied-up capital from overstocking. The model became operational supporting weekly inventory planning with an operational buffers of applying only a small upward correction multiplier.


## Model Exlainability & Trust

To ensure the model is fair, unbiased, and understandable, I evaluated how the model made its decisions and identified ranked top 10 demand drivers, directly informing inventory manager which factors dictacte genuinely dictate demand. 

**Building Executive Trust**: Instead of presenting a black-box forecast to stakeholder, I used feature importance to transform the model into an interpretable narrative: "Our AI predicts a 20% spike next month primarily due to the upcoming regional holiday combined with unseasonably warm weather projections." This not only revealed why demand shifts but also provided a strategic roadmap for business decision-making. 