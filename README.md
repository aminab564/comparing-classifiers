# Predicting Term Deposit Subscriptions Comparative Classifier Analysis

A machine learning project comparing classification algorithms to optimize bank marketing campaign efficiency through predictive customer targeting.

## 📋 Project Overview

This project analyzes data from a Portuguese banking institution's direct marketing campaigns (17 campaigns, 41,188 contacts, May 2008 - November 2010) to predict whether clients will subscribe to term deposit products. Using data mining and machine learning techniques, we developed models to improve campaign efficiency by identifying high-potential customers.

**Business Objective:** Reduce marketing contacts while maintaining or improving subscription rates by predicting which clients are most likely to subscribe to term deposits.

## 🎯 Key Results

### Best Model: Tuned Decision Tree
- **F1-Score:** 0.2512
- **Recall:** 63.90% (finds 2 in 3 subscribers)
- **ROC-AUC:** 0.6529
- **Business Impact:** 51% reduction in contacts while capturing 64% of revenue

### Campaign Efficiency Improvement
| Metric | Random Targeting | ML Model (Decision Tree) | Improvement |
|--------|------------------|--------------------------|-------------|
| Contacts needed | 8,238 (100%) | ~4,000 (49%) | **51% reduction** |
| Subscribers found | 928 (100%) | 593 (64%) | **64% capture rate** |
| Success rate | 11.3% | 15.6% | **+38%** |
| ROI | 10.3x | 13.8x | **+34%** |

## 📊 Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

**Key Statistics:**
- **Total records:** 41,188 contacts
- **Features:** 20 input variables (demographic, campaign, economic indicators)
- **Target variable:** Binary (yes/no subscription)
- **Class imbalance:** 11.27% positive class (subscriptions)
- **Time period:** May 2008 - November 2010

**Features include:**
- **Client data:** Age, job, marital status, education, financial status
- **Campaign data:** Contact type, month, day of week, duration, number of contacts
- **Historical data:** Previous campaign outcomes, days since last contact
- **Economic indicators:** Employment rate, consumer confidence, Euribor rates

## 🔍 Methodology

### 1. Exploratory Data Analysis
- Identified severe class imbalance (88.73% negative class)
- Discovered key predictors: call duration, previous campaign success, temporal patterns (quarterly peaks)
- Found economic indicators strongly correlated with success rates

### 2. Data Preprocessing
- Handled 'unknown' categorical values
- Label encoded categorical features (6 variables)
- Standardized numeric features for distance-based models
- Train-test split: 80/20 with stratification

### 3. Model Development & Comparison

**Models Tested:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)

**Key Challenge:** Class imbalance required `class_weight='balanced'` parameter for effective minority class prediction.

### 4. Hyperparameter Tuning
- Grid Search with 5-fold cross-validation
- Optimized for F1-Score (better than accuracy for imbalanced data)
- Decision Tree: 420 parameter combinations tested
- SVM: 12 combinations (limited due to computational cost)

## 📈 Model Performance

### Final Model Comparison (After Tuning)

| Model | F1-Score | Recall | Precision | ROC-AUC | Training Time | Status |
|-------|----------|--------|-----------|---------|---------------|--------|
| **Decision Tree** | **0.2512** | 63.90% | 15.63% | **0.6529** | 25s | ✅ Recommended |
| **SVM** | 0.2567 | 63.79% | 16.07% | 0.6500 | 1h 56m | ⚠️ Too slow |
| **Logistic Regression** | 0.2440 | **69.50%** | 14.80% | 0.6172 | 6s | ✅ Alternative |
| **KNN** | 0.1397 | 10.02% | 23.08% | 0.5705 | 18s | ❌ Poor recall |

### Decision Tree (Optimal Parameters)
```python
DecisionTreeClassifier(
    criterion='entropy',
    max_depth=7,
    min_samples_split=100,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)
```

## 💡 Key Insights

### 1. Class Imbalance is Critical
- Without class balancing: Models predicted "no" for everyone (88% accuracy, 0% business value)
- With class balancing: Accuracy dropped to 50-68%, but models found 64-70% of subscribers
- **Learning:** Accuracy is misleading for imbalanced data—focus on F1-Score and ROC-AUC

### 2. Temporal Patterns Matter
- Best months: March, September, October, December (end of quarters)
- Success rates vary 2-3x across months
- **Recommendation:** Schedule campaigns at quarter-end

### 3. Previous Campaign History is Highly Predictive
- Previous success → 65% current success rate
- No previous contact → 10.4% success rate
- **Recommendation:** Prioritize clients with past success

### 4. Precision-Recall Trade-off
- High recall (64-70%) = find most subscribers but many false positives
- Low precision (15-16%) = 5-6 wasted contacts per successful subscription
- **Business decision:** Acceptable trade-off given contact cost vs. subscription value

### 5. Feature Limitations
- Current model uses only 7 basic bank client features
- Adding all 20+ features expected to improve F1-Score by 40-60%
- Economic indicators and campaign history likely to boost performance significantly

## 🚀 Future Improvements

### High Priority
1. **Add all features** (contact info, campaign history, economic indicators)
   - Expected impact: F1-Score 0.25 → 0.35-0.45
2. **Threshold optimization** to balance precision/recall for business goals
3. **Feature engineering** (quarter indicators, age groups, economic sentiment)

### Medium Priority
4. **Ensemble methods** (Random Forest, XGBoost) for better performance
5. **SMOTE resampling** to improve KNN performance
6. **Cost-sensitive learning** with custom business costs ($5/contact, $500/subscription)

### Long-term
7. **Neural networks** for complex pattern detection
8. **Real-time model updates** with new campaign data
9. **Explainable AI (SHAP, LIME)** for stakeholder interpretation

### Performance Targets
| Stage | F1-Score | Recall | Precision | ROC-AUC |
|-------|----------|--------|-----------|---------|
| Current (Tuned DT) | 0.25 | 64% | 16% | 0.65 |
| + All Features | 0.35-0.40 | 70-75% | 25-30% | 0.75-0.80 |
| + Ensemble/Engineering | 0.45-0.50 | 75-80% | 30-35% | 0.80-0.85 |

## 🛠️ Technologies Used

- **Python 3.x**
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Notebook:** Jupyter

## 📓 View the Analysis

**[📊 Open Jupyter Notebook](./notebooks/bank_marketing_analysis.ipynb)**

The notebook contains:
- Complete exploratory data analysis with visualizations
- Data preprocessing and feature engineering
- Model training and comparison
- Hyperparameter tuning with GridSearchCV
- Performance evaluation and business impact analysis
- Detailed findings and recommendations

## 🔑 Key Takeaways

1. **Machine learning can significantly improve marketing efficiency** - 51% fewer contacts with 64% revenue capture
2. **Class imbalance must be addressed** - default models failed completely until class weights were applied
3. **Decision Tree offers best balance** - strong performance, fast training, interpretable results
4. **Simple models can be effective** - with proper tuning and class balancing, traditional ML algorithms provide business value
5. **Feature expansion is crucial** - current model uses only 7 of 20+ available features

## 📧 Contact

Feel free to reach out with questions or suggestions!

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Data source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- Original research: Moro et al., 2014 - "A Data-Driven Approach to Predict the Success of Bank Telemarketing"
