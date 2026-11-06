---
title: Visual defect detection on a production line
order: 4
who: Manufacturing · 3 plants
metric: 94% recall
cat: Personal project
svc: ml
time: 5 months
stack: PyTorch, ONNX, OpenCV, MLflow
repo: https://github.com/yourusername/defect-detection
kind: scatter
seed: 44
---

## The problem

Manual inspection missed small surface defects and slowed the line.

## What I did

Fine-tuned a vision model on 12k labelled images, added active learning to label only uncertain cases, and deployed it on edge devices.

## The result

94% defect recall at line speed, with human review only for uncertain cases.
