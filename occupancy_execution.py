import pandas as pd
import numpy as np

df_raw = pd.read_csv("MT1575_occupancy_corrupted.csv") #reads file and converts it into a DataFrame named df_raw

print("Initial shape:", df_raw.shape) #tells dataset's whole structure rows n stuff 
df_raw.head() #first 5 rows for sanity check 

#1 - Dtype Table BEFORE Cleaning
print("Dtypes BEFORE cleaning:")
df_raw.dtypes

#2 - Missing Values Table BEFORE Cleaning
missing_before = df_raw.isna().sum().sort_values(ascending=False)
missing_before.head(10)
#print(missing_before.head(10))

#3 - Remove Column Name Corruption - Removes leading/trailing spaces in column names
df = df_raw.copy()
df.columns = df.columns.str.strip()
print("Columns after stripping spaces:")
print(df.columns.tolist())

#4 - Remove Duplicates
print("Rows before duplicate removal:", df.shape[0])
df = df.drop_duplicates()
print("Rows after duplicate removal:", df.shape[0])


#5 - Fixing Numeric Corruption
numeric_cols = ["Temperature", "Humidity", "Light", "CO2", "humidityratio"]

for col in numeric_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(",", ".", regex=False)
    df[col] = df[col].replace("?", np.nan)
    df[col] = pd.to_numeric(df[col], errors="coerce")

#6 - Dtype Table AFTER Cleaning
print("Dtypes AFTER numeric conversion:")
df.dtypes


#7 - Missing Values AFTER Type Fix , that ?? bulls turns to NaN standards 
missing_after_typefix = df.isna().sum().sort_values(ascending=False)
missing_after_typefix.head(10)

#8 - Impute Missing Values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print("Missing AFTER imputation:")
df[numeric_cols].isna().sum()

#9 - Drop Identifier Column 
df = df.drop(columns=["RecordID"])

#10 - Ensure Target Correct -converted to logic 1 / Logic 0 
TARGET_COL = "Occupancy"
df[TARGET_COL] = df[TARGET_COL].astype(int)
print("Unique target values:", df[TARGET_COL].unique())

#11 - Final Missing Checks - Important stuff 
#print("Final missing values:")
df.isna().sum().sum()
#print(df.isna().sum().sum())

###### Plots #######

#Plot 1 — Example Problem BEFORE Cleaning
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(df_raw["Light"].astype(str))
plt.title("Light values BEFORE cleaning")
#plt.show()

#Plot 2 - Same AFTER Cleaning 
sns.histplot(df["Light"])
plt.title("Light values AFTER cleaning")
#plt.show()

#Plot 3 - Histogram of Numeric Feature
sns.histplot(df["CO2"], bins=30)
plt.title("CO2 Distribution After Cleaning")
#plt.show()

#Plot 4 - Target Distribution ##### THIS IS HEAVILY IMBALANCED !!!!!
df[TARGET_COL].value_counts().plot(kind="bar")
plt.title("Occupancy Class Distribution")
#plt.show()

################### Exploratory Data Analysis (EDA) #####################

# Class count
class_counts = df[TARGET_COL].value_counts()
class_percent = df[TARGET_COL].value_counts(normalize=True) * 100

class_table = pd.DataFrame({
    "Count": class_counts,
    "Percentage": class_percent.round(2)
})

class_table

#Plotting Class Balance 
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x=TARGET_COL, data=df)
plt.title("Occupancy Class Distribution")
#plt.show()


#Histograms
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df["CO2"], bins=30)
plt.title("CO2 Distribution")
plt.subplot(1,2,2)
sns.histplot(df["Light"], bins=30)
plt.title("Light Distribution")
plt.tight_layout()
#plt.show()

#Is CO2 higher when occupied?
#Is Light mostly zero at night?
#Are distributions skewed?

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.boxplot(x=TARGET_COL, y="CO2", data=df)
plt.title("CO2 vs Occupancy")
plt.subplot(1,2,2)
sns.boxplot(x=TARGET_COL, y="Light", data=df)
plt.title("Light vs Occupancy")
plt.tight_layout()
#plt.show()

#You will likely see:
#CO2 higher when occupied
#Light higher when occupied
#Clear separation potential
#This suggests supervised models will perform well.



#SCATTER PLOT 

plt.figure(figsize=(6,5))
sns.scatterplot(
    x="CO2",
    y="Light",
    hue=TARGET_COL,
    data=df,
    alpha=0.6
)
plt.title("CO2 vs Light Colored by Occupancy")
#plt.show()

#Look For
#Do occupied points cluster in a region?
#Is separation roughly linear?
#Overlap?
#This helps anticipate model complexity.


#CORRELATION ANALYSIS 
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
#plt.show()

#SHOW TOP CORRELATION WITH TARGETS FOR BETTER GRADE 
corr_with_target = corr[TARGET_COL].sort_values(ascending=False)
corr_with_target


################################ TRAIN / VALIDATION / TEST SPLIT + PREPROCESSING #####################

#Separate Features and Target

TARGET_COL = "Occupancy"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# Drop Leakage 
DROP_COLS = ["Timestamp"]
X = X.drop(columns=DROP_COLS)
print("Remaining columns:", X.columns.tolist())


# TRAIN / TEST / VALIDATION / TEST SPLIT 
# Using stratification because dataset is imbalanced.
from sklearn.model_selection import train_test_split
# First split: train vs temp (val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Second split: val vs test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)


#Check Class Balance in Splits (REQUIRED SANITY CHECK)
def print_balance(name, y_split):
    print(f"{name} distribution:")
    print(y_split.value_counts(normalize=True))
    print()

print_balance("Train", y_train)
print_balance("Validation", y_val)
print_balance("Test", y_test)


#Handling Categorical Variables - ONE HOT-ENCODING  
#as Daysofweek + daytype + room is catogorased by str 

numeric_features = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

#Building Processing Pipelines 
#using standardscale numeric features 
#used /using onehotencode categorical features 

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Fit on training only

Xtr_np = preprocessor.fit_transform(X_train)
Xva_np = preprocessor.transform(X_val)
Xte_np = preprocessor.transform(X_test)
print("Processed train shape:", Xtr_np.shape)
print("Processed validation shape:", Xva_np.shape)
print("Processed test shape:", Xte_np.shape)

#Convert Targets to NumPy Arrays for some scratch implementation 
import numpy as np
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()
y_test = y_test.to_numpy()
print("Target shapes:")
print(y_train.shape, y_val.shape, y_test.shape)




########################## LOGISTIC REGRESSION ###################################

#A - SIGMOID FUNCTION 
import numpy as np

#THIS SIGMOID CAUSED HEAVY OVERFLOW 
#def sigmoid(z):
    # Numerically stable sigmoid
 #   return 1 / (1 + np.exp(-z))

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

#B - Loss + Gradient Function
def logreg_loss_and_grad(X, y, w, l2=0.0):
    
    #X: (n, d)
    #y: (n,)
    #w: (d,)
    #l2: regularization strength
    
    n = X.shape[0]
    z = X @ w
    p = sigmoid(z)
    
    # Avoid log(0)
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    
    # Binary cross-entropy loss
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    # Add L2 penalty
    loss += (l2 / 2) * np.sum(w**2)
    
    # Gradient
    grad = (1/n) * (X.T @ (p - y)) + l2 * w
    
    return loss, grad

#C - GRADIENT DESCENT TRAINING 
def fit_logreg_gd(X, y, lr=0.1, n_steps=500, l2=0.0):
    
    #Returns:
    #w: learned weights
    #loss_history: list of losses
    
    n, d = X.shape
    w = np.zeros(d)
    loss_history = []
    
    for step in range(n_steps):
        loss, grad = logreg_loss_and_grad(X, y, w, l2)
        w -= lr * grad
        loss_history.append(loss)
    
    return w, loss_history

#Prediction Functions
def predict_proba_logreg(X, w):
    return sigmoid(X @ w)

def predict_logreg(X, w, threshold=0.5):
    probs = predict_proba_logreg(X, w)
    return (probs >= threshold).astype(int)

#Train Model (Baseline)
w, loss_history = fit_logreg_gd(
    Xtr_np,
    y_train,
    lr=0.1,
    n_steps=500,
    l2=0.0
)

print("Training complete.")

#PLOT LOSS CURVE 
#If curve goes down smoothly → correct implementation.
#If exploding → learning rate too high.

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.title("Logistic Regression Loss vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")
#plt.show()

# Evaluate on Train & Validation
from sklearn.metrics import accuracy_score, f1_score

# Train performance
y_train_pred = predict_logreg(Xtr_np, w)
train_acc = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Validation performance
y_val_pred = predict_logreg(Xva_np, w)
val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

print("Train Accuracy:", train_acc)
print("Train F1:", train_f1)
print("Validation Accuracy:", val_acc)
print("Validation F1:", val_f1)



#COMPLEXITY CURVE 
#You must vary L2 and plot train vs val F1.
#l2_values = np.logspace(-4, 2, 10) #this stops at 100 caused overflows , this is HIGH search range for L2 


#reducing L2 Search Range 
l2_values = np.logspace(-4, 1, 10) #Now this stops at 10 

train_f1_list = []
val_f1_list = []

for l2 in l2_values:
    w, _ = fit_logreg_gd(Xtr_np, y_train, lr=0.1, n_steps=500, l2=l2) #Learning rate decresed from 0.1 to 0.05  
    
    train_pred = predict_logreg(Xtr_np, w)
    val_pred = predict_logreg(Xva_np, w)
    
    train_f1_list.append(f1_score(y_train, train_pred))
    val_f1_list.append(f1_score(y_val, val_pred))

plt.plot(l2_values, train_f1_list, label="Train F1")
plt.plot(l2_values, val_f1_list, label="Validation F1")
plt.xscale("log")
plt.xlabel("L2 Regularization")
plt.ylabel("F1 Score")
plt.legend()
plt.title("Logistic Regression Complexity Curve")
#plt.show()

#LEGENDS TO CHECK IN ABOVE CURVE ********* IMPORTANT 
#	•	Low L2 → overfitting (high train, lower val)
#	•	High L2 → underfitting (both low)
#	•	Best L2 → good bias-variance tradeoff




#################### K _ nearest Neighbor #################################


#For each query sample:
#	1.	Compute distance to all training samples
#	2.	Select k closest
#	3.	Majority vote
#	4.	Optional: weighted vote

#Distance metric: Euclidean

#For two matrices:

#||x - x'||^2 = ||x||^2 + ||x'||^2 - 2x x'

#This avoids loops.

import numpy as np

def knn_predict(X_train, y_train, X_query, k=5, weighted=False):
    
   # X_train: (n_train, d)
   # y_train: (n_train,)
   # X_query: (n_query, d)
    
    
    # Compute squared distances efficiently
    X_train_sq = np.sum(X_train**2, axis=1).reshape(-1, 1)
    X_query_sq = np.sum(X_query**2, axis=1).reshape(1, -1)
    
    distances = X_train_sq + X_query_sq - 2 * X_train @ X_query.T
    distances = np.sqrt(np.maximum(distances, 0))  # avoid negative due to precision
    
    # Get indices of k nearest neighbors
    knn_idx = np.argsort(distances, axis=0)[:k]
    
    # Gather neighbor labels
    knn_labels = y_train[knn_idx]
    
    if not weighted:
        # Majority vote
        preds = (np.mean(knn_labels, axis=0) >= 0.5).astype(int)
    else:
        # Weighted vote (inverse distance)
        knn_dist = np.take_along_axis(distances, knn_idx, axis=0)
        weights = 1 / (knn_dist + 1e-9)
        weighted_sum = np.sum(weights * knn_labels, axis=0)
        preds = (weighted_sum >= np.sum(weights, axis=0)/2).astype(int)
    
    return preds


#TRYING K = 5 first 
from sklearn.metrics import accuracy_score, f1_score

k = 1

y_train_pred = knn_predict(Xtr_np, y_train, Xtr_np, k=k)
y_val_pred = knn_predict(Xtr_np, y_train, Xva_np, k=k)

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Train F1:", f1_score(y_train, y_train_pred))
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation F1:", f1_score(y_val, y_val_pred))


#Complexity Curve 
k_values = range(1, 21)

train_f1_list = []
val_f1_list = []

for k in k_values:
    y_train_pred = knn_predict(Xtr_np, y_train, Xtr_np, k=k)
    y_val_pred = knn_predict(Xtr_np, y_train, Xva_np, k=k)
    
    train_f1_list.append(f1_score(y_train, y_train_pred))
    val_f1_list.append(f1_score(y_val, y_val_pred))


    # Convert lists to numpy arrays for convenience
train_f1_array = np.array(train_f1_list)
val_f1_array = np.array(val_f1_list)

# Find best k based on validation F1
best_index = np.argmax(val_f1_array)
best_k = list(k_values)[best_index]
best_val_f1 = val_f1_array[best_index]
best_train_f1 = train_f1_array[best_index]

print("Best k:", best_k)
print("Best Validation F1:", best_val_f1)
print("Train F1 at Best k:", best_train_f1)

import matplotlib.pyplot as plt

plt.plot(k_values, train_f1_list, label="Train F1")
plt.plot(k_values, val_f1_list, label="Validation F1")
plt.xlabel("k")
plt.ylabel("F1 Score")
plt.title("kNN Complexity Curve")
plt.legend()
#plt.show()

#best_k = k_values[np.argmax(val_f1_list)]
#best_val_f1 = max(val_f1_list)
#print(best_val_f1)


#######################  Principal Component Analysis ###############################
"""
PCA finds directions of maximum variance.

Mathematically:
	1.	Center data
	2.	Compute SVD:

X = U \Sigma V^T
	3.	Principal components = columns of V
	4.	Explained variance:

\frac{\sigma_i^2}{\sum \sigma_j^2}

We will apply PCA to:Xtr_np
"""

# Center training data
#Even though scaled, we center again explicitly
X_mean = np.mean(Xtr_np, axis=0)
X_centered = Xtr_np - X_mean


#Compute SVD 
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
# Principal components
components = Vt

#Compute Explained Variance Ratio
explained_variance = S**2
explained_variance_ratio = explained_variance / np.sum(explained_variance)
cumulative_variance = np.cumsum(explained_variance_ratio)

#Plot Cumulative Explained Variance
import matplotlib.pyplot as plt

plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
#plt.show()
	#How many components explain 90%?
#	•	How many explain 95%?
#	•	Is variance concentrated in first few?

#Choose Number of Components
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print("Number of components for 95% variance:", n_components_95)

#Project Data to 2D for Visualization
Xtr_pca_2d = X_centered @ components[:2].T


#2D Scatter Plot Colored by Occupancy
plt.figure(figsize=(6,5))
plt.scatter(
    Xtr_pca_2d[:, 0],
    Xtr_pca_2d[:, 1],
    c=y_train,
    cmap="coolwarm",
    alpha=0.6
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection (2D) Colored by Occupancy")
plt.colorbar(label="Occupancy")
#plt.show()

""""
Interpretation Guide

When you look at the scatter:
	•	Are classes separable?
	•	Do clusters form?
	•	Is separation roughly linear?
	•	Is there overlap?

This connects to:
	•	Why kNN performed so well
	•	Why logistic regression struggled
    """

####################### k-means Clustering #############################

"""""
k-means algorithm:
	1.	Initialize k centroids randomly
	2.	Assign each point to nearest centroid
	3.	Recompute centroids as mean of assigned points
	4.	Repeat until convergence

Objective:

\sum ||x_i - \mu_{cluster(i)}||^2

"""

#k-means Implementation

import numpy as np

def kmeans_fit(X, k, max_iter=100, tol=1e-4, random_state=42):
    np.random.seed(random_state)
    
    n, d = X.shape
    
    # Randomly initialize centroids
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]
    
    for _ in range(max_iter):
        # Compute distances
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        
        # Assign clusters
        labels = np.argmin(distances, axis=1)
        
        # Compute new centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i)
            else centroids[i]
            for i in range(k)
        ])
        
        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    return centroids, labels

"""""
Silhouette Score (Manual Implementation)

We must compute silhouette score manually (no sklearn).

Silhouette for sample i:

s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}

Where:
	•	a(i) = mean intra-cluster distance
	•	b(i) = mean nearest-cluster distance

    """

def silhouette_score_manual(X, labels):
    n = len(X)
    unique_clusters = np.unique(labels)
    
    silhouette_vals = []
    
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = unique_clusters[unique_clusters != labels[i]]
        
        # a(i)
        if len(same_cluster) > 1:
            a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        else:
            a = 0
        
        # b(i)
        b = np.inf
        for cluster in other_clusters:
            cluster_points = X[labels == cluster]
            dist = np.mean(np.linalg.norm(cluster_points - X[i], axis=1))
            b = min(b, dist)
        
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_vals.append(s)
    
    return np.mean(silhouette_vals)


#TIME ODER COMPLEXITY FOR ABOVE CODE IS O(n^2)



#trying for multiple K values 
k_values = [2, 3, 4, 5, 6]

sil_scores = []

for k in k_values:
    centroids, labels = kmeans_fit(Xtr_np, k)
    score = silhouette_score_manual(Xtr_np, labels)
    sil_scores.append(score)
    print(f"k={k}, Silhouette={score:.4f}")



    #Plot of Silhouette vs k

    import matplotlib.pyplot as plt

plt.plot(k_values, sil_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs k")
plt.grid(True)
#plt.show()

#Choosing best K 
best_k_index = np.argmax(sil_scores)
best_k = k_values[best_k_index]

print("Best k based on silhouette:", best_k)


#Visualize Clusters in PCA 2D Space
#Using PCA 2D projection from earlier.
centroids, labels = kmeans_fit(Xtr_np, best_k)

plt.figure(figsize=(6,5))
plt.scatter(
    Xtr_pca_2d[:, 0],
    Xtr_pca_2d[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.6
)
plt.title("k-means Clusters in PCA 2D Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
#plt.show()


#compare to true occupancy 
plt.figure(figsize=(6,5))
plt.scatter(
    Xtr_pca_2d[:, 0],
    Xtr_pca_2d[:, 1],
    c=y_train,
    cmap="coolwarm",
    alpha=0.6
)
plt.title("True Occupancy in PCA 2D Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Occupancy")
#plt.show()

#Compare visually:
#	•	Do clusters align with occupancy?
#	•	Or does clustering find different structure?





############################ Decision Tree (using SKLearn) ######################## 
"""" Decision Trees?
	•	Nonlinear model
	•	Handles interactions naturally
	•	No scaling sensitivity
	•	Can overfit badly if depth too large

Perfect for overfitting demonstration.

"""


#Train Baseline Decision Tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(Xtr_np, y_train)

# Predictions
y_train_pred = dt.predict(Xtr_np)
y_val_pred = dt.predict(Xva_np)

print("Train F1:", f1_score(y_train, y_train_pred))
print("Validation F1:", f1_score(y_val, y_val_pred))


#Complexity Curve — Vary max_depth 
depth_values = range(1, 21)

train_f1_list = []
val_f1_list = []

for depth in depth_values:
    dt = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    
    dt.fit(Xtr_np, y_train)
    
    train_pred = dt.predict(Xtr_np)
    val_pred = dt.predict(Xva_np)
    
    train_f1_list.append(f1_score(y_train, train_pred))
    val_f1_list.append(f1_score(y_val, val_pred))

import matplotlib.pyplot as plt

plt.plot(depth_values, train_f1_list, label="Train F1")
plt.plot(depth_values, val_f1_list, label="Validation F1")
plt.xlabel("Max Depth")
plt.ylabel("F1 Score")
plt.title("Decision Tree Complexity Curve")
plt.legend()
plt.grid(True)
#plt.show()

import numpy as np

val_f1_array = np.array(val_f1_list)
best_index = np.argmax(val_f1_array)
best_depth = list(depth_values)[best_index]
best_val_f1 = val_f1_array[best_index]
best_train_f1 = train_f1_list[best_index]

print("Best Depth:", best_depth)
print("Best Validation F1:", best_val_f1)
print("Train F1 at Best Depth:", best_train_f1)

""""

What You Should See
	•	Small depth → underfitting (low train & val F1)
	•	Medium depth → best validation
	•	Large depth → train → 1.0, val decreases → overfitting

    """


##################### RANDOM FOREST ##############################
"""
Why Random Forest?
	•	Ensemble of decision trees
	•	Reduces overfitting
	•	Handles nonlinear interactions
	•	Robust to noise
	•	Often strong baseline for tabular data
"""

# Baseline Random Forest 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(Xtr_np, y_train)

y_train_pred = rf.predict(Xtr_np)
y_val_pred = rf.predict(Xva_np)

print("Train F1:", f1_score(y_train, y_train_pred))
print("Validation F1:", f1_score(y_val, y_val_pred))

######## Complexity Curve -- Vary n_estimators ##################
n_values = [5, 10, 25, 50, 100, 200]

train_f1_list = []
val_f1_list = []

for n in n_values:
    rf = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(Xtr_np, y_train)
    
    train_pred = rf.predict(Xtr_np)
    val_pred = rf.predict(Xva_np)
    
    train_f1_list.append(f1_score(y_train, train_pred))
    val_f1_list.append(f1_score(y_val, val_pred))

import matplotlib.pyplot as plt

plt.plot(n_values, train_f1_list, label="Train F1")
plt.plot(n_values, val_f1_list, label="Validation F1")
plt.xlabel("Number of Trees")
plt.ylabel("F1 Score")
plt.title("Random Forest Complexity Curve")
plt.legend()
plt.grid(True)
#plt.show()


#Select Best n_estimators 

import numpy as np

val_f1_array = np.array(val_f1_list)
best_index = np.argmax(val_f1_array)

best_n = n_values[best_index]
best_val_f1 = val_f1_array[best_index]
best_train_f1 = train_f1_list[best_index]

print("Best n_estimators:", best_n)
print("Best Validation F1:", best_val_f1)
print("Train F1 at Best n:", best_train_f1)


"""
Typically:
	•	Small n → unstable
	•	Increasing n → validation improves
	•	Eventually plateaus
	•	Train F1 may stay near 1.0
	•	Validation becomes stable

Random Forest usually:
	•	Matches or slightly improves over single tree
	•	Reduces variance
	•	More stable
"""

######################## Model Comparison and Final Evaluation ##############################

#Building Comparison Table 
import pandas as pd

comparison_table = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "kNN (k=4)",
        "Decision Tree (depth=5)",
        "Random Forest (200 trees)"
    ],
    "Validation F1": [
        0.6623,   # replace with your exact value
        0.9356,
        0.9700,
        0.9727
    ]
})

comparison_table


#FINAL BEST -> RANDOM FOREST 
#COMBINING TRAIN +VALIDATION 

import numpy as np

X_trainval = np.vstack([Xtr_np, Xva_np])
y_trainval = np.concatenate([y_train, y_val])

print("Train+Val shape:", X_trainval.shape)

#Retrain Final Model
from sklearn.ensemble import RandomForestClassifier

final_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_trainval, y_trainval)


#Evaluating on Test Set 
from sklearn.metrics import accuracy_score, f1_score

y_test_pred = final_model.predict(Xte_np)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Test Accuracy:", test_accuracy)
print("Test F1:", test_f1)

#Confusion Matrix 
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.show()


#ROC curve 

from sklearn.metrics import roc_curve, auc

y_test_probs = final_model.predict_proba(Xte_np)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_test_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.show()

#Precision recall curve 
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_test_probs)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()



