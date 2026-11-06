# 📱 Mobile Price Classification & Model Evaluation Dashboard

This repository contains an interactive machine learning pipeline and a production-grade Streamlit web application. The dashboard evaluates and compares 5 distinct pre-trained classifiers on the Mobile Price Classification dataset without running online model re-training phases.

---
### a. Problem Statement
In the highly competitive mobile phone market, telecommunication companies and retail distributors cannot simply set device prices based on arbitrary estimations or basic brand markups. Instead, pricing must be derived systematically from the device's technical specifications and internal hardware components (e.g., RAM capacity, battery power, camera megapixels, and network capabilities). 

The objective of this project is to build a high-performance **Multi-Class Machine Learning Classification Pipeline** using the **Mobile Price Classification Dataset**. The system evaluates 21 distinct technical specifications to accurately categorize an unknown mobile device into one of four distinct price ranges: `0` (Low Cost), `1` (Medium Cost), `2` (High Cost), or `3` (Very High Cost). 

By establishing optimized mathematical boundaries across five baseline classification models (Logistic Regression, Decision Trees, KNN, Naive Bayes, and Support Vector Machines) and deploying the final pipeline as a zero-retraining web application via **Streamlit Community Cloud**, this project provides businesses with an automated, data-driven framework to estimate device valuation, optimize product sourcing costs, and streamline inventory pricing strategies.

---

### b. Dataset Description
The dataset contains **2,000 instances** with **21 raw technical attributes** representing the internal hardware configurations of various mobile devices. It is perfectly balanced, with exactly 500 records allocated to each of the four target price range classes. 

| Feature Name | Feature Type | Unit / Representation | Description |
| :--- | :--- | :--- | :--- |
| `battery_power` | Continuous Numerical | milliampere-hour (mAh) | Total energy capacity of the battery |
| `blue` | Binary Boolean | 0 = No, 1 = Yes | Presence of built-in Bluetooth connectivity |
| `clock_speed` | Continuous Numerical | Gigahertz (GHz) | Speed at which the microprocessor executes instructions |
| `dual_sim` | Binary Boolean | 0 = No, 1 = Yes | Has dual SIM card support capability |
| `fc` | Discrete Numerical | Megapixels (MP) | Front-facing camera resolution |
| `four_g` | Binary Boolean | 0 = No, 1 = Yes | Has 4G LTE network communication support |
| `int_memory` | Discrete Numerical | Gigabytes (GB) | Internal storage space capacity |
| `m_dep` | Continuous Numerical | Centimeters (cm) | Mobile device depth (thickness metric) |
| `mobile_wt` | Discrete Numerical | Grams (g) | Total weight of the mobile handset |
| `n_cores` | Discrete Numerical | Integer Count | Number of active CPU processing cores |
| `pc` | Discrete Numerical | Megapixels (MP) | Primary (rear-facing) camera resolution |
| `px_height` | Discrete Numerical | Pixels | Pixel resolution height of the display screen |
| `px_width` | Discrete Numerical | Pixels | Pixel resolution width of the display screen |
| `ram` | Continuous Numerical | Megabytes (MB) | Random Access Memory space allocation |
| `sc_h` | Discrete Numerical | Centimeters (cm) | Physical screen height dimensions |
| `sc_w` | Discrete Numerical | Centimeters (cm) | Physical screen width dimensions |
| `talk_time` | Discrete Numerical | Hours (h) | Maximum battery duration during continuous call usage |
| `three_g` | Binary Boolean | 0 = No, 1 = Yes | Has 3G network communication support |
| `touch_screen` | Binary Boolean | 0 = No, 1 = Yes | Has an interactive touch-responsive display |
| `wifi` | Binary Boolean | 0 = No, 1 = Yes | Has built-in Wi-Fi connectivity support |
| **`price_range`** | **Categorical Target** | **`0`, `1`, `2`, `3`** | **Target Class Mapping: Low, Medium, High, or Very High Cost** |

---

### c. GitHub Repository Link

👉 **GitHub Repository Link:** [https://github.com/mgroovyank/ml-assignment-2](https://github.com/mgroovyank/ml-assignment-2)


### d. Models Used
The metrics below demonstrate validation results computed across the exact same 400 holdout samples within the uniform processed matrix space:

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **96.25%** | **0.9982** | **0.9623** | **0.9625** | **0.9623** | **0.9500** |
| **Support Vector Machine** | 90.75% | 0.9629 | 0.9069 | 0.9075 | 0.9070 | 0.8768 |
| **Decision Tree** | 88.25% | 0.9557 | 0.8820 | 0.8825 | 0.8818 | 0.8436 |
| **Gaussian Naive Bayes** | 83.75% | 0.9714 | 0.8407 | 0.8375 | 0.8381 | 0.7840 |
| **K-Nearest Neighbors** | 77.25% | 0.9294 | 0.7754 | 0.7725 | 0.7722 | 0.6977 |

### Observations on the performance of each model on the chosen dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Top Performer (96.25% Acc, 0.9982 AUC).** The linear boundary assumption maps perfectly onto this dataset. Adding the engineered RAM × Battery synergy feature allowed the solver to construct near-flawless separating hyperplanes between pricing categories. |
| **Support Vector Machine** | **Strong Contender (90.75% Acc).** Strong Margin Classifier (90.75% Acc). Cross-validated tuning of the soft-margin C parameter balanced structural variance effectively. The RBF kernel handled multi-feature interactions cleanly, but it slightly trails Logistic Regression due to the continuous nature of the input features. |
| **Decision Tree** | **Optimised Tree Splitting (88.25% Acc).**  Using GridSearchCV to tune max_depth and min_samples_leaf successfully prevented over-branching. Performance improved significantly because the tree could no longer overfit the noise features dropped during selection. |
| **Gaussian Naive Bayes** | **Lightweight Probabilistic Baseline (83.75% Acc).** Consolidating the redundant camera attributes into a single engineered feature satisfied the class conditional independence assumption. This directly led to an exceptionally strong 0.9714 AUC score. |
| **K-Nearest Neighbors** | **Most Improved Classifier (77.25% Acc).** Tuning via grid search completely rescued this model from its baseline limitations. Finding the optimal n_neighbors combined with Manhattan distance coordinate geometries dramatically suppressed dimensional noise. |

### Model-Driven Feature Engineering Rationale
Instead of feeding raw dataset distributions directly into the classifiers, the pipeline builds model-driven transformations to satisfy specific mathematical assumptions:
1. **Camera Consolidation:** Merges the front camera (`fc`) and primary camera (`pc`) features into `total_megapixels` to resolve strong feature multicollinearity ($0.64$ correlation score). This adjustment prevents probabilistic double-counting in Naive Bayes and Logistic Regression.
2. **Synergy Modeling:** Generates a `ram_battery_interaction` feature ($RAM \times Battery\ Power$) to provide the linear classifiers with an explicit numeric column capturing the diagonal class boundaries identified during data visualization.
3. **Irrelevant Feature Selection:** Drops `clock_speed` ($-0.01$), `mobile_wt` ($-0.03$), and `wifi` ($0.02$) due to near-zero correlations with the target price range. Removing this statistical noise prevents distance metric collapse in KNN and limits unpruned branch variance in the Decision Tree.

### Parameter Selection Rationale
* **Logistic Regression (`max_iter=1000`):** Increased optimization iterations from default bounds to guarantee gradient convergence across multi-class vectors.
* **Decision Tree Classifier (`criterion=gini`, `max_depth=8`, `min_samples_leaf=5`):** Used Grid Search CV to restrict depth and setting a minimum sample boundary per leaf which prevents the algorithm from creating brittle, overfitted leaf splits on data boundary points.
* **K-Nearest Neighbors (`metric=manhattan`, `n_neighbors=15`, `weights=distance`):** • Switched to distance-based weighting so that highly similar hardware configurations hold greater voting power than distant outliers. Implemented the Manhattan distance metric to provide geometric stability across the 14-dimensional feature vector layout, mitigating the Curse of Dimensionality.
* **Gaussian Naive Bayes (`var_smoothing=1e-09`):** Adds a small fraction of the largest feature variance to stabilize probability arrays, preventing runtime crashes on strict out-of-bounds metrics.
* **Support Vector Machine (`kernel='rbf'` wrapped in `CalibratedClassifierCV`):** Employs a Radial Basis Function kernel to project complex non-linear coordinate spaces into higher dimensions. The calibration wrapper generates accurate multi-class class probability arrays without calling deprecated internal parameters.
