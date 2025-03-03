# Russian Car Plates Prices Prediction [Kaggle Competition]

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/russian-car-plates-prices-prediction)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

A machine learning solution to predict prices of Russian vehicle license plates using semantic feature engineering and optimized regression models. Built for Kaggle's competition with **SMAPE evaluation**.

## Overview
Predict prices of Russian license plates using historical sales data containing plate characteristics and price evolution. Plates with government affiliation or symbolic numbers (e.g., 777) command premium prices.

**Competition Metric**: Symmetric Mean Absolute Percentage Error (SMAPE)

## Approach
### Key Steps
1. **Plate Decomposition**  
   - Extracted region codes (82 regions + special codes)
   - Split into letters (12 allowed chars) and numbers
   ```python
   'X059CP797' → {'region': '797', 'letters': 'XCP', 'number': 59}
   ```

2. **Feature Engineering**  
   - Government plate flags (`AMP`, `EKX` series)
   - Lucky numbers (777/888), palindromes, round numbers
   - Region popularity encoding

3. **Class Imbalance Handling**  
   - Custom sample weights (1:86 ratio for rare government plates)
   - Stratified cross-validation by price bins

4. **Modeling**  
   - LightGBM with hyperparameter tuning
   - Regularization (λ=10, α=5) to prevent overfitting
   - 5-fold cross-validation (SMAPE: 23.5%)

##  Challenges
1. **High Cardinality**  
   - 1,728 unique letter combinations → Target encoding
   - 88 regions → Frequency encoding

2. **Skewed Distribution**  
   - 98.85% regular plates vs 1.15% government plates
   - Addressed with custom weighting:
   ```python
   weights = (100/1.158 * has_advantage) * (100/5.134 * is_special_number)
   ```

3. **SMAPE Sensitivity**  
   - Zero prices prohibited → Clip predictions >0
   - Logarithmic transformation experiments

## Competition Results
Current Best Submission:  
- Public LB SMAPE: 24.1%  
- Private LB SMAPE: TBD  

*Note: Please go through the notebook, which is well explained using comments.*

