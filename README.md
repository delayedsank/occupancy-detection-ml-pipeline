# occupancy-detection-ml-pipeline
Full ML pipeline on corrupted sensor data — data cleaning, EDA, from-scratch  Logistic Regression/kNN/PCA/k-means, Random Forest, MLP — occupancy detection




**Dataset:** Room occupancy sensor data (Temperature, Humidity, Light, CO2, HumidityRatio)  
**Task:** Binary classification — predict whether a room is occupied (1) or not (0)

---

## What This Project Does

End-to-end machine learning pipeline on a **deliberately corrupted** real-world 
sensor dataset, covering every stage from raw data to final test evaluation.

---

## Pipeline Stages

### 1. Data Cleaning
The raw CSV had multiple corruption issues handled explicitly:
- Column names with leading/trailing whitespace — stripped
- Timestamps out of ascending order — sorted
- 157 duplicate rows — removed
- Numeric columns stored as strings with comma decimals (`"21,3"`) — converted
- `"?"` placeholders masking missing values — replaced with `NaN`, then median-imputed
- Identifier column (`RecordID`) dropped to prevent leakage

Before/after comparison plots and dtype tables included.

### 2. Exploratory Data Analysis (EDA)
- Class imbalance identified: ~79% unoccupied vs ~21% occupied → F1 chosen over accuracy
- CO2 and Light identified as strongest predictors via correlation analysis
- Boxplots, histograms, scatter plots, and correlation heatmap

### 3. Preprocessing
- Stratified 60/20/20 train/validation/test split
- `StandardScaler` on numeric features, `OneHotEncoder` on categoricals
- Fit only on training data to prevent data leakage

### 4. From-Scratch Implementations (NumPy only)

| Component | Implementation Details |
|---|---|
| **Logistic Regression** | Sigmoid (clipped for stability), binary cross-entropy loss, L2 regularization, gradient descent |
| **kNN** | Vectorized Euclidean distance using `‖x-x'‖² = ‖x‖² + ‖x'‖² - 2xᵀx'`, majority + weighted vote |
| **PCA** | SVD decomposition via `np.linalg.svd`, explained variance ratio, 2D projection |
| **k-means** | Random centroid init, iterative assignment/update, convergence check |
| **Silhouette Score** | Manual O(n²) implementation: intra-cluster distance a(i), nearest-cluster distance b(i) |

### 5. Supervised Learning — Model Selection & Complexity Curves

Each model was tuned via its complexity parameter on the **validation set only**:

| Model | Tuned Parameter | Best Val F1 |
|---|---|---|
| Logistic Regression (scratch) | L2 regularization | ~0.66 |
| kNN (scratch) | k neighbors | ~0.94 |
| Decision Tree (sklearn) | max_depth | ~0.97 |
| Random Forest (sklearn) | n_estimators | ~0.97 |
| MLP (sklearn) | architecture | ~0.97 |

### 6. Unsupervised Learning
- PCA: 2D scatter confirms two separable clusters aligning with occupancy labels
- k-means sweep (k=2..6) with silhouette scoring — best k=2 aligns with binary labels
- Visual comparison: k-means clusters vs true occupancy in PCA space

### 7. Final Evaluation
Best model (Random Forest, `class_weight="balanced"`) retrained on train+val, 
evaluated **once** on held-out test set:
- Confusion matrix
- ROC curve + AUC
- Precision-Recall curve

---

## Tech Stack
