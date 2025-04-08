# Credit Risk Prediction with LightGBM, CatBoost & SHAP

This project predicts loan default risk using optimized feature engineering and interpretable ML models. It compares LightGBM and CatBoost using SHAP to visualize and explain feature influence. Built using the Kaggle Home Credit dataset.

---

## Highlights

- Efficient feature engineering using Polars
- Dual-model training with LightGBM & CatBoost
- SHAP summary plots and model explainability
- Venn diagram comparison of top features
- Interpretable ML for credit scoring decisions

---

## SHAP Explainability

### LightGBM vs CatBoost SHAP Summary

<p align="center">
  <img src="shap_outputs/shap_lgb_catboost_comparison.png" width="100%">
</p>

LightGBM emphasizes income type, gender, and response dates.  
CatBoost uncovers insights from birth dates, request types, and contract values.

---

### Top Feature Overlap (Venn Diagram)

<p align="center">
  <img src="shap_outputs/shap_feature_overlap_venn.png" width="60%">
</p>

Of the top 20 SHAP-ranked features:
- 8 are shared across both models
- 12 are unique to LightGBM
- 12 are unique to CatBoost  

This highlights the value of model diversity and complementary learning.

## Results (CV AUC)

| Model     | AUC Score     |
|-----------|---------------|
| LightGBM  | ~0.81–0.85    |
| CatBoost  | ~0.81–0.82    |

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
