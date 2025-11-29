# Heart Disease Prediction using Machine Learning

A comprehensive data science project implementing machine learning algorithms to predict the presence of heart disease based on medical attributes. This project demonstrates the complete data science workflow from data exploration to model deployment.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
- [Project Workflow](#project-workflow)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Results & Insights](#results--insights)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

---

## 🎯 Project Overview

### Course Information
- **Course**: CS 591 - Tools & Techniques for Data Science
- **Institution**: [Your University Name]
- **Semester**: Fall 2025
- **Submission Date**: December 7, 2025

### Objectives
This project aims to:
1. Analyze the UCI Heart Disease dataset to identify key risk factors
2. Build and compare multiple machine learning classification models
3. Predict the presence of heart disease with high accuracy
4. Deploy a practical interface for real-world clinical use
5. Demonstrate proficiency in the complete data science pipeline

### Problem Statement
Heart disease is one of the leading causes of death globally. Early detection and prediction can save lives through timely intervention. This project develops a machine learning system to predict heart disease presence based on 13 clinical and demographic features.

---

## 📊 Dataset Information

### Source
**UCI Machine Learning Repository** - Cleveland Heart Disease Dataset  
**URL**: https://archive.ics.uci.edu/dataset/45/heart+disease

### Dataset Characteristics
- **Total Records**: 303 patients
- **Features**: 14 attributes (13 predictors + 1 target)
- **Type**: Multivariate, Mixed (Numeric & Categorical)
- **Missing Values**: 6 (ca: 4, thal: 2)
- **Class Distribution**: 
  - No Disease: 164 patients (54.1%)
  - Disease Present: 139 patients (45.9%)

### Features Description

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| **age** | Age in years | Numeric | 29-77 |
| **sex** | Gender | Categorical | 0 = Female, 1 = Male |
| **cp** | Chest pain type | Categorical | 1-4 |
| **trestbps** | Resting blood pressure (mm Hg) | Numeric | 94-200 |
| **chol** | Serum cholesterol (mg/dl) | Numeric | 126-564 |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0 = No, 1 = Yes |
| **restecg** | Resting ECG results | Categorical | 0-2 |
| **thalach** | Maximum heart rate achieved | Numeric | 71-202 |
| **exang** | Exercise induced angina | Binary | 0 = No, 1 = Yes |
| **oldpeak** | ST depression induced by exercise | Numeric | 0-6.2 |
| **slope** | Slope of peak exercise ST segment | Categorical | 1-3 |
| **ca** | Number of major vessels (0-3) | Numeric | 0-3 |
| **thal** | Thalassemia | Categorical | 3, 6, 7 |
| **target** | Heart disease diagnosis | Binary | 0 = No, 1 = Yes |

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── data/
│   ├── raw/
│   │   └── processed.cleveland.data          # Original dataset
│   └── processed/
│       └── heart_disease_cleaned.csv         # Cleaned dataset
│
├── notebooks/
│   ├── 01_data_loading.ipynb                 # Data loading and overview
│   ├── 02_eda.ipynb                          # Exploratory data analysis
│   ├── 03_data_cleaning.ipynb                # Data preprocessing
│   ├── 04_visualizations.ipynb               # Data visualizations
│   └── 05_ml_models.ipynb                    # Machine learning models
│
├── src/
│   ├── predict.py                            # Prediction module
│   └── app.py                                # Interactive interface
│
├── models/
│   └── final_best_model.pkl                  # Trained model
│
├── reports/
│   ├── figures/                              # All visualization images
│   │   ├── 01_target_distribution.png
│   │   ├── 02_age_distribution.png
│   │   ├── 03_gender_analysis.png
│   │   ├── 04_chest_pain_analysis.png
│   │   ├── 05_correlation_heatmap.png
│   │   ├── 06_cholesterol_analysis.png
│   │   ├── 07_bp_analysis.png
│   │   ├── 08_heart_rate_analysis.png
│   │   ├── 09_feature_correlations.png
│   │   ├── 10_multiple_features.png
│   │   ├── 11_model_comparison.png
│   │   ├── 12_confusion_matrix.png
│   │   ├── 13_feature_importance.png
│   │   └── 14_learning_curves.png
│   └── [RollNumber]_CS591_Project.pdf        # Final report
│
├── .gitignore                                # Git ignore file
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
└── setup.py                                  # Project setup script
```

---

## 🛠️ Technologies Used

### Programming Language
- **Python 3.11+**

### Core Libraries
- **Data Manipulation**: pandas 2.0.3, numpy 1.24.3
- **Visualization**: matplotlib 3.7.2, seaborn 0.12.2
- **Machine Learning**: scikit-learn 1.3.0
- **Model Persistence**: joblib 1.3.2
- **Development**: jupyter 1.0.0, ipykernel 6.25.0

### Development Tools
- **IDE**: Jupyter Notebook, VS Code
- **Version Control**: Git, GitHub
- **Environment**: Virtual Environment (venv)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Git
- pip (Python package manager)

### Step 1: Clone Repository
```bash
git clone https://github.com/HaseebUlHassan437/data-science-mscs-semester-project.git
cd data-science-mscs-semester-project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run Jupyter Notebooks
```bash
jupyter notebook
```

Navigate to `notebooks/` folder and run notebooks in order (01 to 05).

### Step 6: Run Prediction Interface
```bash
cd src
python app.py
```

---

## 🔄 Project Workflow

### 1. Data Loading & Understanding 
**Notebook**: `01_data_loading.ipynb`

- Loaded UCI Heart Disease dataset (303 records, 14 features)
- Added column names to raw data file
- Handled missing value indicators (? → NaN)
- Converted multi-class target (0-4) to binary (0/1)
- Generated dataset overview and statistics

**Key Outputs**:
- Dataset shape: (303, 14)
- Missing values identified: 6 total
- Disease prevalence: 45.9%

---

### 2. Exploratory Data Analysis
**Notebook**: `02_eda.ipynb`

#### Statistical Analysis
- Descriptive statistics for all features
- Data type verification (all numeric)
- Missing value analysis (ca: 4, thal: 2)

#### Univariate Analysis
- Age: Mean 54.4 years, range 29-77
- Gender: 68% male, 32% female
- Chest pain: Type 4 (asymptomatic) most common (47.5%)

#### Bivariate Analysis
- Disease by gender: Males 55.3% vs Females 25.8%
- Disease by age: Increases with age (60+ group: 55.7%)
- Disease by chest pain: Type 4 highest risk (72.9%)

#### Correlation Analysis
Top correlations with disease:
1. Thalassemia (thal): 0.53
2. Major vessels (ca): 0.46
3. Exercise angina (exang): 0.43
4. ST depression (oldpeak): 0.42
5. Chest pain (cp): 0.41

#### Outlier Detection
- Age outliers: 0
- Cholesterol outliers: 5 (max 564 mg/dl)

---

### 3. Data Cleaning & Preprocessing 
**Notebook**: `03_data_cleaning.ipynb`

#### Missing Value Handling
- **ca column**: Filled 4 missing values with median
- **thal column**: Filled 2 missing values with mode
- Result: 0 missing values

#### Duplicate Check
- No duplicate rows found

#### Outlier Treatment
- Identified 5 cholesterol outliers using IQR method
- Applied capping (clipping) instead of removal
- Range normalized: 115.0 to 371.0 mg/dl

#### Feature Engineering
Created 3 new binary features:
1. **age_above_50**: Age threshold indicator
2. **high_chol**: Cholesterol > 200 mg/dl
3. **low_heart_rate**: Max heart rate < 140 bpm

#### Final Dataset
- Records: 303 (no data loss)
- Features: 17 (14 original + 3 engineered)
- Missing values: 0
- Saved as: `heart_disease_cleaned.csv`

---

### 4. Data Visualization [15 marks]
**Notebook**: `04_visualizations.ipynb`

Created 10 comprehensive visualizations:

1. **Target Distribution**: Bar chart showing class balance
2. **Age Distribution**: Histogram and box plot by disease status
3. **Gender Analysis**: Disease prevalence by gender
4. **Chest Pain Analysis**: Disease rates across pain types
5. **Correlation Heatmap**: Feature relationships (14x14 matrix)
6. **Cholesterol Analysis**: Distribution by disease status
7. **Blood Pressure Scatter**: Age vs BP colored by disease
8. **Heart Rate Analysis**: Box plot comparison
9. **Feature Correlations**: Top 10 predictors bar chart
10. **Multiple Features**: 2x2 grid comparing key features

**Key Observations**:
- Clear visual patterns in gender, chest pain type, and age groups
- Strong correlation visualized between thal, ca, cp and target
- Asymptomatic chest pain shows highest disease association
- Males show significantly higher disease rates than females

All figures saved in `reports/figures/` as high-resolution PNG files (300 DPI).

---

### 5. Machine Learning Models 
**Notebook**: `05_ml_models.ipynb`

#### Data Preparation
- Train-test split: 80-20 (242 train, 61 test)
- Stratified sampling to maintain class distribution
- Feature scaling: StandardScaler for distance-based models

#### Models Trained
Five classification algorithms compared:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **88.52%** | 81.82% | 96.43% | 88.52% |
| Logistic Regression | 86.89% | 81.25% | 92.86% | 86.67% |
| Gradient Boosting | 85.25% | 78.79% | 92.86% | 85.25% |
| SVM | 81.97% | 75.76% | 89.29% | 81.97% |
| Decision Tree | 70.49% | 64.71% | 78.57% | 70.97% |

#### Best Model: Random Forest
- **Test Accuracy**: 88.52%
- **Cross-validation Score**: 80.98%
- **Confusion Matrix**:
  - True Negatives: 27
  - False Positives: 6
  - False Negatives: 1
  - True Positives: 27

#### Feature Importance (Top 5)
1. Thalassemia (thal): 14.42%
2. Maximum heart rate (thalach): 11.82%
3. Chest pain type (cp): 11.72%
4. Major vessels (ca): 10.57%
5. ST depression (oldpeak): 9.74%

#### Model Optimization

**Overfitting Analysis**:
- Training accuracy: 100%
- Test accuracy: 88.52%
- Gap: 11.48% (mild overfitting detected)
- Cross-validation: 80.98% (good generalization)

**Feature Selection**:
- Tested with top 10 features only
- Result: Maintained 88.52% accuracy
- Conclusion: All features contribute effectively

**Hyperparameter Tuning**:
- Method: GridSearchCV with 5-fold CV
- Parameters tested: n_estimators, max_depth, min_samples_split
- Best parameters: 
  - n_estimators: 200
  - max_depth: None
  - min_samples_split: 10
- **Final Accuracy: 90.16%** (improved by 1.64%)

#### Final Model
- **Algorithm**: Tuned Random Forest Classifier
- **Final Test Accuracy**: 90.16%
- **Model saved**: `models/final_best_model.pkl`

---

### 6. Git Version Control [15 marks]

#### Repository Management
- **Platform**: GitHub
- **Repository URL**: https://github.com/HaseebUlHassan437/data-science-mscs-semester-project
- **Visibility**: Public

#### Commit History
Total commits: 25+

**Sample commit messages**:
- "Initial commit: Project structure setup"
- "Add data loading notebook with dataset overview"
- "Complete EDA: correlation analysis shows thal, ca, cp as top predictors"
- "Complete data cleaning: handled 6 missing values and 5 outliers"
- "Create 10 visualizations: distribution, correlation, and feature analysis"
- "Train 5 ML models: best model achieves 88.52% accuracy"
- "Optimize model to 90.16% accuracy: handle overfitting via hyperparameter tuning"
- "Add interactive prediction interface for deployment"

#### Repository Structure
- Well-organized folder hierarchy
- Comprehensive .gitignore (excludes venv, cache, temp files)
- Regular commits after each major milestone
- Meaningful commit messages describing changes
- Professional README documentation

---

## 📈 Key Findings

### Risk Factors Identified
1. **Thalassemia status** is the strongest predictor (14.4% importance)
2. **Males** have 2.15x higher disease rate than females
3. **Asymptomatic chest pain** paradoxically shows highest risk (72.9%)
4. **Age over 60** significantly increases disease probability
5. **Lower maximum heart rate** correlates with disease presence

### Medical Insights
- Exercise-induced angina is a strong indicator
- Number of major vessels colored by fluoroscopy highly predictive
- ST depression during exercise shows strong correlation
- Cholesterol alone is a weak predictor (correlation: 0.085)

### Data Quality Observations
- Minimal missing data (1.98%)
- No duplicate records
- Some outliers in cholesterol measurements
- Well-balanced target classes (54.1% vs 45.9%)

---

## 🎯 Model Performance

### Final Model Specifications
- **Algorithm**: Random Forest Classifier (Tuned)
- **Parameters**: 
  - n_estimators: 200
  - max_depth: None
  - min_samples_split: 10
  - random_state: 42

### Performance Metrics
- **Accuracy**: 90.16%
- **Precision**: 81.82%
- **Recall**: 96.43%
- **F1-Score**: 88.52%
- **Cross-validation Score**: 82.22%

### Model Strengths
- High recall (96.43%): Catches most disease cases
- Balanced performance across metrics
- Good generalization (CV score close to test)
- Interpretable feature importance
- Fast prediction (<0.1 seconds)

### Model Limitations
- Perfect training accuracy indicates mild overfitting
- May misclassify some edge cases (7 errors in 61 predictions)
- Limited to features available in training data
- Requires medical validation before clinical use

---

## 🖥️ Deployment

### Interactive Prediction System

A command-line interface was developed for practical deployment.

#### Features
- User-friendly input prompts for all 13 parameters
- Automatic calculation of engineered features
- Real-time prediction with confidence scores
- Clinical recommendations based on results

#### Usage
```bash
cd src
python app.py
```

#### Example Interaction
```
Enter patient information:
Age (years): 55
Sex (1=Male, 0=Female): 1
Chest Pain Type (1-4): 4
...

PREDICTION RESULT
================
Result: HEART DISEASE DETECTED
Confidence: 87.45%
Recommendation: Consult a cardiologist immediately
```

#### Deployment Files
- `predict.py`: Core prediction logic
- `app.py`: User interface
- `final_best_model.pkl`: Trained model (131 KB)

---

## 💡 Results & Insights

### Project Achievements
✅ Successfully predicted heart disease with 90.16% accuracy  
✅ Identified top 5 risk factors with clinical relevance  
✅ Created 10 publication-quality visualizations  
✅ Developed deployment-ready prediction interface  
✅ Maintained comprehensive version control (25+ commits)  
✅ Zero data loss during cleaning (all 303 records retained)  

### Business Value
- **Clinical Decision Support**: Aids doctors in preliminary screening
- **Cost Reduction**: Reduces unnecessary advanced testing
- **Time Efficiency**: Instant predictions vs hours of analysis
- **Accessibility**: Command-line tool deployable on any system
- **Transparency**: Feature importance shows model reasoning

### Academic Contributions
- Comprehensive data science workflow demonstration
- Proper handling of medical dataset challenges
- Comparison of multiple ML algorithms
- Overfitting detection and mitigation strategies
- Professional-level documentation and reporting

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] Implement ensemble methods (voting, stacking)
- [ ] Try deep learning approaches (neural networks)
- [ ] Add SHAP values for better interpretability
- [ ] Implement k-fold stratified cross-validation
- [ ] Test with additional datasets for validation

### Feature Engineering
- [ ] Create interaction features (e.g., age × cholesterol)
- [ ] Add polynomial features for non-linear relationships
- [ ] Implement automated feature selection (RFE, SelectKBest)
- [ ] Include temporal features if longitudinal data available

### Deployment
- [ ] Build web interface using Streamlit or Flask
- [ ] Create REST API for integration with hospital systems
- [ ] Add patient data management database
- [ ] Implement user authentication and authorization
- [ ] Deploy on cloud platform (AWS, Azure, GCP)

### Data Collection
- [ ] Expand dataset with more recent patient records
- [ ] Include additional clinical parameters
- [ ] Collect follow-up data for outcome validation
- [ ] Add demographic diversity (multiple hospitals)

### Validation
- [ ] Clinical trial with healthcare professionals
- [ ] External dataset validation
- [ ] A/B testing in real clinical settings
- [ ] Regulatory compliance assessment (FDA, HIPAA)

---

## 👥 Contributors

**Student Information**
- **Name**: [Haseeb Ul Hassan]
- **Roll Number**: [MSCS25003]
- **Email**: [mscs25003@itu.edu.pk]
- **Program**: MS Computer Science
- **Semester**: 1st Semester, Fall 2025

**Course Instructor**
- **Name**: Kamil Majeed
- **Course Code**: CS 591
- **Course Title**: Tools & Techniques for Data Science

---

## 📄 License

This project is submitted as an academic assignment for educational purposes only.

### Usage Terms
- ⚠️ **Not for clinical use**: This model is for educational demonstration only
- ⚠️ **Medical disclaimer**: Should not replace professional medical diagnosis
- ✅ Academic reference and learning purposes permitted
- ✅ Code reuse allowed with proper attribution

### Dataset Attribution
- **Original Source**: UCI Machine Learning Repository
- **Citation**: Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

## 📞 Contact & Support

For questions, suggestions, or collaboration:

- **GitHub Issues**: [Create an issue](https://github.com/HaseebUlHassan437/data-science-mscs-semester-project/issues)
- **Email**: [mscs25003@itu.edu.pk]

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the dataset
- **Scikit-learn documentation** for ML implementation guidance
- **Stack Overflow community** for troubleshooting support
- **Course instructor** for project guidance and feedback

---

## 📚 References

1. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease Dataset. UCI Machine Learning Repository.

2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

3. McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 56-61.

4. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95.

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

**Built with ❤️ for CS 591 - Tools & Techniques for Data Science**

*Last Updated: November 29, 2025*

</div>
