---
title: Data quality monitoring for a subscription app
description: Automated tests and alerts so broken data never reaches the churn report.
metric: "−85% broken reports"
client: "SaaS · 250k subscribers"
role: Analytics engineer
duration: 5 weeks
decision: "The growth team trusted the churn numbers enough to launch a win-back campaign without a manual check."
deliverables:
  - Freshness, volume and accuracy tests
  - Alerts to the right Slack channels
  - Runbook for the five most common failures
tags: [dbt, Great Expectations, Slack]
image: ""
github: https://github.com/machaniG/your-quality-repo
demo: ""
---
## The mess
Reports broke silently. A late data load or a changed column would quietly skew the churn numbers, and analysts only found out when a stakeholder asked why a chart looked wrong.

## What I built
I added tests for freshness, row counts and key business rules, and routed failures to Slack with the name of the owner. A short runbook explains how to fix the most common problems.

## The decision it enabled
Broken reports dropped by 85%. Because the team trusted the numbers, they launched a win-back campaign straight from the dashboard, without a manual check first.
