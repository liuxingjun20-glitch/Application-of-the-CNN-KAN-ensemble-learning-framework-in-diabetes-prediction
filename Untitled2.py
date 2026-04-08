#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from BorutaShap import BorutaShap
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)
import random


# In[2]:


df = pd.read_csv('C:/Users/65163/Desktop/python/diabetes_data.csv')
x = df.drop(columns=['PatientID','Diagnosis','DoctorInCharge'])
y = df['Diagnosis']


# In[3]:


continuous_features = [
    'Age',
    'BMI',
    'AlcoholConsumption',
    'PhysicalActivity',
    'DietQuality',
    'SleepQuality',
    'SystolicBP',
    'DiastolicBP',
    'FastingBloodSugar',
    'HbA1c',
    'SerumCreatinine',
    'BUNLevels',
    'CholesterolTotal',
    'CholesterolLDL',
    'CholesterolHDL',
    'CholesterolTriglycerides',
    'FatigueLevels',
    'QualityOfLifeScore',
    'MedicalCheckupsFrequency',
    'MedicationAdherence',
    'HealthLiteracy']

categorical_features = [
    'Gender',
    'Ethnicity',
    'SocioeconomicStatus',
    'EducationLevel',
    'Smoking',
    'FamilyHistoryDiabetes',
    'GestationalDiabetes',
    'PolycysticOvarySyndrome',
    'PreviousPreDiabetes',
    'Hypertension',
    'AntihypertensiveMedications',
    'Statins',
    'AntidiabeticMedications',
    'FrequentUrination',
    'ExcessiveThirst',
    'UnexplainedWeightLoss',
    'BlurredVision',
    'SlowHealingSores',
    'TinglingHandsFeet',
    'HeavyMetalsExposure',
    'OccupationalExposureChemicals',
    'WaterQuality']


# In[4]:


#Descriptive statistical analysis of continuous variables
continuous_stats = df[continuous_features].describe().T
continuous_stats = continuous_stats[['mean','std','min','max']]
print(continuous_stats)


# In[5]:


#Descriptive statistical analysis of categorical variables
rows = []

for col in categorical_features:
    counts = df[col].value_counts()
    percents = df[col].value_counts(normalize=True) * 100

    first = True
    for category in counts.index:
        rows.append({
            "Variable": col if first else "",
            "Category": category,
            "N": counts[category],
            "%": round(percents[category], 1)
        })
        first = False

categorical_table = pd.DataFrame(rows)

print(categorical_table)


# In[6]:


distribution = pd.DataFrame({
    'Count': y.value_counts(),
    'Percentage (%)': y.value_counts(normalize=True) * 100
})
print(distribution)


# In[7]:


#Data splitting
seeds = list(range(1, 51))  

data_splits = {}   

for seed in seeds:
    X_train, X_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        stratify=y,
        random_state=seed
    )

    data_splits[seed] = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


# In[8]:


Boruta_seed = 50
X_train_set = data_splits[Boruta_seed]["X_train"]
y_train_set = data_splits[Boruta_seed]["y_train"]


# In[9]:


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
# Initialize XGBoost classifier
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=Boruta_seed,
    n_jobs=1
)

# Define hyperparameter search space
param_dist = {
    'n_estimators': randint(100, 500),        
    'max_depth': randint(3, 10),              
    'learning_rate': uniform(0.01, 0.19),     
    'subsample': uniform(0.6, 0.4),           
    'colsample_bytree': uniform(0.6, 0.4),    
    'gamma': uniform(0, 0.5),                 
    'min_child_weight': randint(1, 10)        
}

# Stratified 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=Boruta_seed
)

# Randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=100,                
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=Boruta_seed
)

random_search.fit(X_train_set, y_train_set)

print("Best parameters:")
print(random_search.best_params_)

print("Best CV AUC:")
print(random_search.best_score_)


# In[12]:


# Use the best hyperparameters from RandomizedSearchCV
best_params = random_search.best_params_
xgb_model = XGBClassifier(
    **best_params,
    random_state=Boruta_seed,
    use_label_encoder=False,
    eval_metric="logloss",
    n_jobs=1,          
)
# Stratified 5-fold cross-validation
kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=Boruta_seed

)

fold_selected_features = {}
feature_importance_dict = {} 

for fold, (train_idx, val_idx) in enumerate(
    kf.split(X_train_set, y_train_set)
    ):
    print(f"\nFold {fold + 1} ----------------")

    X_train = X_train_set.iloc[train_idx]
    y_train = y_train_set.iloc[train_idx]

    feature_selector = BorutaShap(
        model=xgb_model,
        importance_measure="shap",
        classification=True
    )

    feature_selector.fit(
        X=X_train,
        y=y_train,
        n_trials=200,
        random_state=Boruta_seed
    )

    selected_features = feature_selector.accepted
    fold_selected_features[fold] = selected_features

    print(f"Selected features ({len(selected_features)}):")
    print(selected_features)
    xgb_model.fit(X_train, y_train)
    importances = xgb_model.feature_importances_

    for f, v in zip(X_train.columns, importances):
        feature_importance_dict.setdefault(f, []).append(v)


# In[83]:


#The final selected features
from collections import Counter
all_features = []
for feats in fold_selected_features.values():
    all_features.extend(feats)

feature_counter = Counter(all_features)

feature_freq = {
    k: v / kf.n_splits
    for k, v in feature_counter.items()
}

final_features = [
    k for k, v in feature_freq.items() if v > 0.6
]


mean_importance = {
    f: np.mean(v)
    for f, v in feature_importance_dict.items()
}



final_features_sorted = sorted(
    final_features,
    key=lambda x: (feature_freq[x], mean_importance.get(x, 0)),
    reverse=True
)

print("\nFinal selected features (sorted):")
print(final_features_sorted)


# In[78]:


import matplotlib.pyplot as plt
import seaborn as sns


freq_df = pd.DataFrame.from_dict(
    feature_freq, orient="index", columns=["Frequency"]
)


freq_df["Importance"] = freq_df.index.map(mean_importance)

freq_df["Selected"] = freq_df.index.isin(final_features)

freq_df = freq_df.sort_values(
    by=["Selected", "Importance"],
    ascending=[False, False]
)

colors = freq_df["Selected"].map({True: "red", False: "lightgray"})


plt.figure(figsize=(6, 4))

sns.barplot(
    data=freq_df,
    x="Frequency",
    y=freq_df.index,
    palette=colors.tolist()
)

plt.axvline(0.6, linestyle="--", color="black", label="60% threshold")

plt.xlabel("Selection Frequency")
plt.ylabel("Features")
plt.title("Feature Selection Stability (BorutaShap)")
plt.legend()
plt.tight_layout()

plt.savefig("BorutaSHAP_feature_selection.png", dpi=300)
plt.show()


# In[28]:


#The final selected features are applied to the training set and the test set.
scaled_data = {}   

for seed in seeds:

    X_train_set = data_splits[seed]["X_train"]
    X_test_set  = data_splits[seed]["X_test"]
    y_train_set = data_splits[seed]["y_train"]
    y_test_set  = data_splits[seed]["y_test"]


    X_train_selected = X_train_set[final_features_sorted]
    X_test_selected  = X_test_set[final_features_sorted]


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled  = scaler.transform(X_test_selected)


    scaled_data[seed] = {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_set": y_train_set,
        "y_test_set": y_test_set,
        "scaler": scaler
    }


# In[30]:


#Define the parameter settings for MLP
def build_mlp(input_dim, seed):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    MLP_model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation="relu", kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    MLP_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return MLP_model


# In[31]:


Accuracy_MLP = []
Recall_MLP = []
Precision_MLP = []
F1_MLP = []
AUC_MLP = []
MLP_probs_list = []
MLP_train_probs_list = [] 

for seed in seeds:
    print(f"\nSeed {seed} ----------------")

    X_train_selected = scaled_data[seed]["X_train_scaled"]
    X_test_selected  = scaled_data[seed]["X_test_scaled"]
    y_train_set = scaled_data[seed]["y_train_set"]
    y_test_set  = scaled_data[seed]["y_test_set"]

    MLP_model = build_mlp(
        input_dim=X_train_selected.shape[1],
        seed=seed
    )

    #Train MLP
    MLP_model.fit(
        X_train_selected,
        y_train_set,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1,
    )
    y_prob_train_MLP = MLP_model.predict(X_train_selected).ravel()  
    y_pred_train_MLP = (y_prob_train_MLP > 0.5).astype(int)

    #Test MLP
    y_prob_MLP = MLP_model.predict(X_test_selected).ravel()
    y_pred_MLP = (y_prob_MLP > 0.5).astype(int)  

    #Computer performance metrics
    acc_MLP = accuracy_score(y_test_set, y_pred_MLP)
    recall_MLP = recall_score(y_test_set, y_pred_MLP)
    precision_MLP = precision_score(y_test_set, y_pred_MLP, zero_division=0)
    f1_MLP = f1_score(y_test_set, y_pred_MLP)
    auc_MLP = roc_auc_score(y_test_set, y_prob_MLP)

    Accuracy_MLP.append(acc_MLP)
    Recall_MLP.append(recall_MLP)
    Precision_MLP.append(precision_MLP)
    F1_MLP.append(f1_MLP)
    AUC_MLP.append(auc_MLP)
    MLP_probs_list.append(y_prob_MLP)
    MLP_train_probs_list.append(y_prob_train_MLP)
    print(
        "[MLP] "
        f"Acc: {acc_MLP:.3f} | "
        f"Recall: {recall_MLP:.3f} | "
        f"Precision: {precision_MLP:.3f} | "
        f"F1: {f1_MLP:.3f} | "
        f"AUC: {auc_MLP:.3f}"
    )


# In[32]:


print("\n===== MLP + BorutaShap (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_MLP):.3f} ± {np.std(Accuracy_MLP):.3f}")
print(f"Recall   : {np.mean(Recall_MLP):.3f} ± {np.std(Recall_MLP):.3f}")
print(f"Precision: {np.mean(Precision_MLP):.3f} ± {np.std(Precision_MLP):.3f}")
print(f"F1-score : {np.mean(F1_MLP):.3f} ± {np.std(F1_MLP):.3f}")
print(f"AUC      : {np.mean(AUC_MLP):.3f} ± {np.std(AUC_MLP):.3f}")


# In[33]:


#Define the parameter settings for CNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

def build_cnn(input_len, seed):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    CNN_model = Sequential([
        Input(shape=(input_len, 1)),

        Conv1D(32, 3, activation="relu", padding="same"),
        MaxPooling1D(2),

        Conv1D(32, 3, activation="relu", padding="same"),
        GlobalAveragePooling1D(),

        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    CNN_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return CNN_model


# In[34]:


Accuracy_CNN = []
Recall_CNN = []
Precision_CNN = []
F1_CNN = []
AUC_CNN = []
CNN_probs_list = []
CNN_train_probs_list = []

for seed in seeds:
    print(f"\nCNN | Seed {seed} ----------------")

    X_train_CNN = scaled_data[seed]["X_train_scaled"].reshape(
        scaled_data[seed]["X_train_scaled"].shape[0],
        scaled_data[seed]["X_train_scaled"].shape[1],
        1)
    X_test_CNN = scaled_data[seed]["X_test_scaled"].reshape(
        scaled_data[seed]["X_test_scaled"].shape[0],
        scaled_data[seed]["X_test_scaled"].shape[1],
        1)
    y_train_set = scaled_data[seed]["y_train_set"]
    y_test_set  = scaled_data[seed]["y_test_set"]

    CNN_model = build_cnn(
        input_len=X_train_CNN.shape[1],
        seed=seed
    )

    #Train CNN
    CNN_model.fit(
        X_train_CNN,
        y_train_set,
        validation_split=0.2,  
        epochs=50,
        batch_size=32,
        verbose=1,
    )

    y_prob_train_CNN = CNN_model.predict(X_train_CNN).ravel()
    y_pred_train_CNN = (y_prob_train_CNN > 0.5).astype(int)

    #Test CNN
    y_prob_CNN = CNN_model.predict(X_test_CNN).ravel()
    y_pred_CNN = (y_prob_CNN > 0.5).astype(int)

    #Computer performance metrics
    acc_CNN = accuracy_score(y_test_set, y_pred_CNN)
    recall_CNN = recall_score(y_test_set, y_pred_CNN)
    precision_CNN = precision_score(y_test_set, y_pred_CNN, zero_division=0)
    f1_CNN = f1_score(y_test_set, y_pred_CNN)
    auc_CNN = roc_auc_score(y_test_set, y_prob_CNN)


    Accuracy_CNN.append(acc_CNN)
    Recall_CNN.append(recall_CNN)
    Precision_CNN.append(precision_CNN)
    F1_CNN.append(f1_CNN)
    AUC_CNN.append(auc_CNN)
    CNN_probs_list.append(y_prob_CNN)
    CNN_train_probs_list.append(y_prob_train_CNN)

    print(
        "[CNN] "
        f"Acc: {acc_CNN:.3f} | "
        f"Recall: {recall_CNN:.3f} | "
        f"Precision: {precision_CNN:.3f} | "
        f"F1: {f1_CNN:.3f} | "
        f"AUC: {auc_CNN:.3f}"
    )


# In[35]:


print("\n===== CNN + BorutaShap (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_CNN):.3f} ± {np.std(Accuracy_CNN):.3f}")
print(f"Recall   : {np.mean(Recall_CNN):.3f} ± {np.std(Recall_CNN):.3f}")
print(f"Precision: {np.mean(Precision_CNN):.3f} ± {np.std(Precision_CNN):.3f}")
print(f"F1-score : {np.mean(F1_CNN):.3f} ± {np.std(F1_CNN):.3f}")
print(f"AUC      : {np.mean(AUC_CNN):.3f} ± {np.std(AUC_CNN):.3f}")


# In[36]:


#Define the parameter settings for GRU
from tensorflow.keras.layers import GRU

def build_gru(input_len, seed):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    GRU_model = Sequential([
        Input(shape=(input_len, 1)),

        GRU(32, return_sequences=False),

        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    GRU_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return GRU_model


# In[37]:


Accuracy_GRU = []
Recall_GRU = []
Precision_GRU = []
F1_GRU = []
AUC_GRU = []
GRU_probs_list = []
GRU_train_probs_list = []

for seed in seeds:
    print(f"\nGRU | Seed {seed} ----------------")
    X_train_GRU = scaled_data[seed]["X_train_scaled"].reshape(
        scaled_data[seed]["X_train_scaled"].shape[0],
        scaled_data[seed]["X_train_scaled"].shape[1],
        1)
    X_test_GRU = scaled_data[seed]["X_test_scaled"].reshape(
        scaled_data[seed]["X_test_scaled"].shape[0],
        scaled_data[seed]["X_test_scaled"].shape[1],
        1)
    y_train_set = scaled_data[seed]["y_train_set"]
    y_test_set  = scaled_data[seed]["y_test_set"]

    GRU_model = build_gru(
        input_len=X_train_GRU.shape[1],
        seed=seed
    )

    #Train GRU
    GRU_model.fit(
        X_train_GRU,
        y_train_set,
        validation_split=0.2,   
        epochs=50,
        batch_size=32,
        verbose=1
    )

    y_prob_train_GRU = GRU_model.predict(X_train_GRU).ravel()
    y_pred_train_GRU = (y_prob_train_GRU > 0.5).astype(int)

    #Test GRU
    y_prob_GRU = GRU_model.predict(X_test_GRU).ravel()
    y_pred_GRU = (y_prob_GRU > 0.5).astype(int)

    #Computer performance metrics
    acc_GRU = accuracy_score(y_test_set, y_pred_GRU)
    recall_GRU = recall_score(y_test_set, y_pred_GRU)
    precision_GRU = precision_score(y_test_set, y_pred_GRU, zero_division=0)
    f1_GRU = f1_score(y_test_set, y_pred_GRU)
    auc_GRU = roc_auc_score(y_test_set, y_prob_GRU)

    Accuracy_GRU.append(acc_GRU)
    Recall_GRU.append(recall_GRU)
    Precision_GRU.append(precision_GRU)
    F1_GRU.append(f1_GRU)
    AUC_GRU.append(auc_GRU)
    GRU_probs_list.append(y_prob_GRU)
    GRU_train_probs_list.append(y_prob_train_GRU)
    print(
        "[GRU] "
        f"Acc: {acc_GRU:.3f} | "
        f"Recall: {recall_GRU:.3f} | "
        f"Precision: {precision_GRU:.3f} | "
        f"F1: {f1_GRU:.3f} | "
        f"AUC: {auc_GRU:.3f}"
    )


# In[38]:


print("\n===== GRU + BorutaShap (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_GRU):.3f} ± {np.std(Accuracy_GRU):.3f}")
print(f"Recall   : {np.mean(Recall_GRU):.3f} ± {np.std(Recall_GRU):.3f}")
print(f"Precision: {np.mean(Precision_GRU):.3f} ± {np.std(Precision_GRU):.3f}")
print(f"F1-score : {np.mean(F1_GRU):.3f} ± {np.std(F1_GRU):.3f}")
print(f"AUC      : {np.mean(AUC_GRU):.3f} ± {np.std(AUC_GRU):.3f}")


# In[39]:


#Define the parameter settings for KAN
import torch
import torch.nn as nn
from kan.MultKAN import KAN
import torch.nn.functional as F
def build_kan(input_dim, seed):
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    KAN_model = KAN(
        width=[input_dim, 8, 1],  
        grid=3,
        k=3,
        base_fun=F.relu,
        seed=seed
    )
    return KAN_model


# In[40]:


def kan_predict_proba(KAN_model, X):
    KAN_model.eval()
    with torch.no_grad():
        logits = KAN_model(X)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy().ravel()


# In[41]:


Accuracy_KAN = []
Recall_KAN = []
Precision_KAN = []
F1_KAN = []
AUC_KAN = []
KAN_probs_list = []
KAN_train_probs_list = []

for seed in seeds:
    print(f"\nKAN | Seed {seed} ----------------")

    X_train_KAN = torch.tensor(scaled_data[seed]["X_train_scaled"], dtype=torch.float32)
    y_train_KAN = torch.tensor(scaled_data[seed]["y_train_set"].values, dtype=torch.float32).view(-1, 1)
    X_test_KAN  = torch.tensor(scaled_data[seed]["X_test_scaled"], dtype=torch.float32)
    y_test_KAN  = torch.tensor(scaled_data[seed]["y_test_set"].values, dtype=torch.float32).view(-1, 1)


    KAN_model = build_kan(
        input_dim=X_train_KAN.shape[1],
        seed=seed
    )


    dataset = {
        "train_input": X_train_KAN,
        "train_label": y_train_KAN,
        "test_input":  X_test_KAN,
        "test_label":  y_test_KAN
    }

    #Train KAN
    KAN_model.fit(
        dataset,
        steps=50,
        lr=0.005,
        lamb=0.005,
        update_grid=False,
        loss_fn=nn.BCEWithLogitsLoss()
    )


    y_prob_train_KAN = kan_predict_proba(KAN_model, X_train_KAN)
    y_pred_train_KAN = (y_prob_train_KAN > 0.5).astype(int)
    y_true_train_KAN = y_train_KAN.cpu().numpy().ravel()

    #Test KAN
    y_prob_KAN = kan_predict_proba(KAN_model, X_test_KAN)
    y_pred_KAN = (y_prob_KAN > 0.5).astype(int)
    y_true_KAN = y_test_KAN.cpu().numpy().ravel()

    #Computer performance metrics
    acc_KAN = accuracy_score(y_true_KAN, y_pred_KAN)
    recall_KAN = recall_score(y_true_KAN, y_pred_KAN)
    precision_KAN = precision_score(y_true_KAN, y_pred_KAN, zero_division=0)
    f1_KAN = f1_score(y_true_KAN, y_pred_KAN)
    auc_KAN = roc_auc_score(y_true_KAN, y_prob_KAN)

    Accuracy_KAN.append(acc_KAN)
    Recall_KAN.append(recall_KAN)
    Precision_KAN.append(precision_KAN)
    F1_KAN.append(f1_KAN)
    AUC_KAN.append(auc_KAN)
    KAN_probs_list.append(y_prob_KAN)
    KAN_train_probs_list.append(y_prob_train_KAN)
    print(
        f"Acc: {acc_KAN:.3f} | "
        f"Recall: {recall_KAN:.3f} | "
        f"Precision: {precision_KAN:.3f} | "
        f"F1: {f1_KAN:.3f} | "
        f"AUC: {auc_KAN:.3f}"
    )


# In[42]:


print("\n===== KAN + BorutaShap (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_KAN):.3f} ± {np.std(Accuracy_KAN):.3f}")
print(f"Recall   : {np.mean(Recall_KAN):.3f} ± {np.std(Recall_KAN):.3f}")
print(f"Precision: {np.mean(Precision_KAN):.3f} ± {np.std(Precision_KAN):.3f}")
print(f"F1-score : {np.mean(F1_KAN):.3f} ± {np.std(F1_KAN):.3f}")
print(f"AUC      : {np.mean(AUC_KAN):.3f} ± {np.std(AUC_KAN):.3f}")


# In[43]:


results_df = pd.DataFrame({
    "Model": ["MLP", "CNN", "GRU", "KAN"],

    "Accuracy": [
        f"{np.mean(Accuracy_MLP):.3f} ± {np.std(Accuracy_MLP):.3f}",
        f"{np.mean(Accuracy_CNN):.3f} ± {np.std(Accuracy_CNN):.3f}",
        f"{np.mean(Accuracy_GRU):.3f} ± {np.std(Accuracy_GRU):.3f}",
        f"{np.mean(Accuracy_KAN):.3f} ± {np.std(Accuracy_KAN):.3f}",
    ],

    "Recall": [
        f"{np.mean(Recall_MLP):.3f} ± {np.std(Recall_MLP):.3f}",
        f"{np.mean(Recall_CNN):.3f} ± {np.std(Recall_CNN):.3f}",
        f"{np.mean(Recall_GRU):.3f} ± {np.std(Recall_GRU):.3f}",
        f"{np.mean(Recall_KAN):.3f} ± {np.std(Recall_KAN):.3f}",
    ],

    "Precision": [
        f"{np.mean(Precision_MLP):.3f} ± {np.std(Precision_MLP):.3f}",
        f"{np.mean(Precision_CNN):.3f} ± {np.std(Precision_CNN):.3f}",
        f"{np.mean(Precision_GRU):.3f} ± {np.std(Precision_GRU):.3f}",
        f"{np.mean(Precision_KAN):.3f} ± {np.std(Precision_KAN):.3f}",
    ],

    "F1-score": [
        f"{np.mean(F1_MLP):.3f} ± {np.std(F1_MLP):.3f}",
        f"{np.mean(F1_CNN):.3f} ± {np.std(F1_CNN):.3f}",
        f"{np.mean(F1_GRU):.3f} ± {np.std(F1_GRU):.3f}",
        f"{np.mean(F1_KAN):.3f} ± {np.std(F1_KAN):.3f}",
    ],

    "AUC": [
        f"{np.mean(AUC_MLP):.3f} ± {np.std(AUC_MLP):.3f}",
        f"{np.mean(AUC_CNN):.3f} ± {np.std(AUC_CNN):.3f}",
        f"{np.mean(AUC_GRU):.3f} ± {np.std(AUC_GRU):.3f}",
        f"{np.mean(AUC_KAN):.3f} ± {np.std(AUC_KAN):.3f}",
    ]
})

results_df


# In[44]:


from itertools import combinations
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


base_models = ["MLP", "CNN", "GRU", "KAN"]


stacking_results = []
Accuracy_stacking = []
Precision_stacking = []
Recall_stacking = []
F1_stacking = []
AUC_stacking = []
combo_stacking = []
seed_stacking = []   

for seed_idx, seed in enumerate(seeds):

    #Prepare meta features for stacking
    X_train_meta = np.column_stack([
        MLP_train_probs_list[seed_idx],
        CNN_train_probs_list[seed_idx],
        GRU_train_probs_list[seed_idx],
        KAN_train_probs_list[seed_idx]
    ])


    X_test_meta = np.column_stack([
        MLP_probs_list[seed_idx],
        CNN_probs_list[seed_idx],
        GRU_probs_list[seed_idx],
        KAN_probs_list[seed_idx]
    ])
    y_train_set = scaled_data[seeds[seed_idx]]["y_train_set"].values
    y_test_set  = scaled_data[seeds[seed_idx]]["y_test_set"].values

    model_col_idx = {name: i for i, name in enumerate(base_models)}

    #Loop over combinations of base models (2 to 4 models)

    for r in range(2, len(base_models)+1):  
        for combo in combinations(base_models, r):
            cols = [model_col_idx[m] for m in combo]


            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=seed
            )
            #Train the meta-model
            xgb.fit(X_train_meta[:, cols], y_train_set)

            #Test model
            y_prob_stack = xgb.predict_proba(X_test_meta[:, cols])[:, 1]
            y_pred_stack = (y_prob_stack > 0.5).astype(int)

            #Compute performance metrics 
            acc_stacking = accuracy_score(y_test_set, y_pred_stack)
            recall_stacking = recall_score(y_test_set, y_pred_stack)
            precision_stacking = precision_score(y_test_set, y_pred_stack, zero_division=0)
            f1_stacking = f1_score(y_test_set, y_pred_stack)
            auc_stacking = roc_auc_score(y_test_set, y_prob_stack)

            stacking_results.append({
                "Seed": seed,
                "Combination": combo,
                "Accuracy": acc_stacking,

                "Recall": recall_stacking,
                "Precision": precision_stacking,
                "F1": f1_stacking,
                "AUC": auc_stacking
            })
            Accuracy_stacking.append(acc_stacking)
            Recall_stacking.append(recall_stacking)
            Precision_stacking.append(precision_stacking)
            F1_stacking.append(f1_stacking)
            AUC_stacking.append(auc_stacking)
            combo_stacking.append(combo)
            seed_stacking.append(seed)

#Summarize results
results_df = pd.DataFrame(stacking_results)

summary_list = []
for comb in results_df['Combination'].unique():
    subset = results_df[results_df['Combination'] == comb]

    summary_list.append({
        "Combination": comb,
        "Accuracy": f"{subset['Accuracy'].mean():.3f} ± {subset['Accuracy'].std():.3f}",
        "Recall": f"{subset['Recall'].mean():.3f} ± {subset['Recall'].std():.3f}",
        "Precision": f"{subset['Precision'].mean():.3f} ± {subset['Precision'].std():.3f}",
        "F1": f"{subset['F1'].mean():.3f} ± {subset['F1'].std():.3f}",
        "AUC": f"{subset['AUC'].mean():.3f} ± {subset['AUC'].std():.3f}"
    })

summary_df = pd.DataFrame(summary_list).sort_values('AUC', ascending=False)
print(summary_df)


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']


data_numeric = summary_df.copy()
for m in metrics:

    data_numeric[m] = data_numeric[m].apply(lambda x: float(x.split(' ± ')[0]))


scaler = MinMaxScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data_numeric[metrics]),
    index=data_numeric['Combination'],
    columns=metrics
)

plt.figure(figsize=(10, max(4, len(data_scaled) * 0.4)))
sns.heatmap(
    data_scaled,
    annot=summary_df[metrics],   
    fmt="",
    cmap="RdYlGn_r",
    linewidths=0.5,
    cbar_kws={'label': 'Normalized score'}
)

plt.title("Stacking Model Performance Heatmap", fontsize=14)
plt.xlabel("Metrics")
plt.ylabel("Model Combination")
plt.tight_layout()
plt.show()


# In[47]:


#Use all features
scaled_data_all = {}
for seed in seeds:

    X_train_set = data_splits[seed]["X_train"]
    X_test_set  = data_splits[seed]["X_test"]
    y_train_set = data_splits[seed]["y_train"]
    y_test_set  = data_splits[seed]["y_test"]

    scaler = StandardScaler()
    X_train_all_scaled = scaler.fit_transform(X_train_set)
    X_test_all_scaled  = scaler.transform(X_test_set)

    scaled_data_all[seed] = {
        "X_train_all_scaled": X_train_all_scaled,
        "X_test_all_scaled": X_test_all_scaled,
        "y_train_set": y_train_set,
        "y_test_set": y_test_set,
        "scaler": scaler
    }


# In[48]:


#Define the parameter settings for MLP
def build_mlp_all(input_dim, seed):

    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    MLP_model_all = Sequential([
        Input(shape=(input_dim,)),

        Dense(
            4,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-3)
        ),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    MLP_model_all.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.005   
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return MLP_model_all


# In[49]:


Accuracy_MLP_all = []
Recall_MLP_all = []
Precision_MLP_all = []
F1_MLP_all = []
AUC_MLP_all = []

MLP_all_probs_list = []
MLP_all_train_probs_list = []

for seed in seeds:
    print(f"\nSeed {seed} ----------------")

    X_train_all = scaled_data_all[seed]["X_train_all_scaled"]
    X_test_all  = scaled_data_all[seed]["X_test_all_scaled"]
    y_train_set = scaled_data_all[seed]["y_train_set"]
    y_test_set  = scaled_data_all[seed]["y_test_set"]

    MLP_model_all = build_mlp_all(
        input_dim=X_train_all.shape[1],
        seed=seed
    )

    #Train MLP
    MLP_model_all.fit(
        X_train_all,
        y_train_set,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )


    y_prob_train_MLP_all = MLP_model_all.predict(X_train_all).ravel()
    y_pred_train_MLP_all = (y_prob_train_MLP_all > 0.5).astype(int)

    #Test MLP
    y_prob_MLP_all = MLP_model_all.predict(X_test_all).ravel()
    y_pred_MLP_all = (y_prob_MLP_all > 0.5).astype(int)


    MLP_all_acc = accuracy_score(y_test_set, y_pred_MLP_all)
    MLP_all_recall = recall_score(y_test_set, y_pred_MLP_all)
    MLP_all_precision = precision_score(y_test_set, y_pred_MLP_all, zero_division=0)
    MLP_all_f1 = f1_score(y_test_set, y_pred_MLP_all)
    MLP_all_auc = roc_auc_score(y_test_set, y_prob_MLP_all)


    Accuracy_MLP_all.append(MLP_all_acc)
    Recall_MLP_all.append(MLP_all_recall)
    Precision_MLP_all.append(MLP_all_precision)
    F1_MLP_all.append(MLP_all_f1)
    AUC_MLP_all.append(MLP_all_auc)

    MLP_all_probs_list.append(y_prob_MLP_all)
    MLP_all_train_probs_list.append(y_prob_train_MLP_all)

    print(
        "[MLP-All] "
        f"Acc: {MLP_all_acc:.3f} | "
        f"Recall: {MLP_all_recall:.3f} | "
        f"Precision: {MLP_all_precision:.3f} | "
        f"F1: {MLP_all_f1:.3f} | "
        f"AUC: {MLP_all_auc:.3f}"
    )


# In[50]:


print("\n===== MLP (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_MLP_all):.3f} ± {np.std(Accuracy_MLP_all):.3f}")
print(f"Recall   : {np.mean(Recall_MLP_all):.3f} ± {np.std(Recall_MLP_all):.3f}")
print(f"Precision: {np.mean(Precision_MLP_all):.3f} ± {np.std(Precision_MLP_all):.3f}")
print(f"F1-score : {np.mean(F1_MLP_all):.3f} ± {np.std(F1_MLP_all):.3f}")
print(f"AUC      : {np.mean(AUC_MLP_all):.3f} ± {np.std(AUC_MLP_all):.3f}")


# In[51]:


#Define the parameter settings for CNN
def build_cnn_all(input_len, seed):

    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    CNN_model_all = Sequential([
        Input(shape=(input_len, 1)),

        Conv1D(16, 3, activation="relu", padding="same",kernel_regularizer=regularizers.l2(1e-3)),
        MaxPooling1D(2),

        Conv1D(8, 3, activation="relu", padding="same",kernel_regularizer=regularizers.l2(1e-3)),
        GlobalAveragePooling1D(),

        Dropout(0.3),
        Dense(8, activation="relu",kernel_regularizer=regularizers.l2(1e-3)),
        Dense(1, activation="sigmoid")
    ])

    CNN_model_all.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001   
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return CNN_model_all


# In[52]:


Accuracy_CNN_all = []
Recall_CNN_all = []
Precision_CNN_all = []
F1_CNN_all = []
AUC_CNN_all = []

CNN_all_probs_list = []
CNN_all_train_probs_list = []

for seed in seeds:
    print(f"\nSeed {seed} ----------------")
    X_train_CNN_all = scaled_data_all[seed]["X_train_all_scaled"].reshape(
        scaled_data_all[seed]["X_train_all_scaled"].shape[0],
        scaled_data_all[seed]["X_train_all_scaled"].shape[1],
        1)
    X_test_CNN_all = scaled_data_all[seed]["X_test_all_scaled"].reshape(
        scaled_data_all[seed]["X_test_all_scaled"].shape[0],
        scaled_data_all[seed]["X_test_all_scaled"].shape[1],
        1)
    y_train_set = scaled_data_all[seed]["y_train_set"]
    y_test_set  = scaled_data_all[seed]["y_test_set"]

    CNN_model_all = build_cnn_all(
        input_len=X_train_CNN_all.shape[1],
        seed=seed
    )

    #Train CNN
    CNN_model_all.fit(
        X_train_CNN_all,
        y_train_set,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )


    y_prob_train_CNN_all = CNN_model_all.predict(X_train_CNN_all).ravel()
    y_pred_train_CNN_all = (y_prob_train_CNN_all > 0.5).astype(int)

    #Test CNN
    y_prob_CNN_all = CNN_model_all.predict(X_test_CNN_all).ravel()
    y_pred_CNN_all = (y_prob_CNN_all > 0.5).astype(int)


    acc_CNN_all = accuracy_score(y_test_set, y_pred_CNN_all)
    recall_CNN_all = recall_score(y_test_set, y_pred_CNN_all)
    precision_CNN_all = precision_score(y_test_set, y_pred_CNN_all, zero_division=0)
    f1_CNN_all = f1_score(y_test_set, y_pred_CNN_all)
    auc_CNN_all = roc_auc_score(y_test_set, y_prob_CNN_all)


    Accuracy_CNN_all.append(acc_CNN_all)
    Recall_CNN_all.append(recall_CNN_all)
    Precision_CNN_all.append(precision_CNN_all)
    F1_CNN_all.append(f1_CNN_all)
    AUC_CNN_all.append(auc_CNN_all)

    CNN_all_probs_list.append(y_prob_CNN_all)
    CNN_all_train_probs_list.append(y_prob_train_CNN_all)

    print(
        "[CNN-All] "
        f"Acc: {acc_CNN_all:.3f} | "
        f"Recall: {recall_CNN_all:.3f} | "
        f"Precision: {precision_CNN_all:.3f} | "
        f"F1: {f1_CNN_all:.3f} | "
        f"AUC: {auc_CNN_all:.3f}"
    )


# In[53]:


print("\n===== MLP (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_CNN_all):.3f} ± {np.std(Accuracy_CNN_all):.3f}")
print(f"Recall   : {np.mean(Recall_CNN_all):.3f} ± {np.std(Recall_CNN_all):.3f}")
print(f"Precision: {np.mean(Precision_CNN_all):.3f} ± {np.std(Precision_CNN_all):.3f}")
print(f"F1-score : {np.mean(F1_CNN_all):.3f} ± {np.std(F1_CNN_all):.3f}")
print(f"AUC      : {np.mean(AUC_CNN_all):.3f} ± {np.std(AUC_CNN_all):.3f}")


# In[54]:


#Define the parameter settings for GRU
def build_gru_all(input_len, seed):
    tf.keras.utils.set_random_seed(seed)   
    np.random.seed(seed)
    random.seed(seed)

    GRU_model_all = Sequential([
        Input(shape=(input_len, 1)),

        GRU(
            units=16,   
            return_sequences=False,
            kernel_regularizer=regularizers.l2(0.01)
        ),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    GRU_model_all.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return GRU_model_all


# In[55]:


Accuracy_GRU_all = []
Precision_GRU_all = []
Recall_GRU_all = []
F1_GRU_all = []
AUC_GRU_all = []

GRU_all_probs_list = []
GRU_all_train_probs_list = []

for seed in seeds:
    print(f"\nSeed {seed} ----------------")

    X_train_GRU_all = scaled_data_all[seed]["X_train_all_scaled"].reshape(
        scaled_data_all[seed]["X_train_all_scaled"].shape[0],
        scaled_data_all[seed]["X_train_all_scaled"].shape[1],
        1)
    X_test_GRU_all = scaled_data_all[seed]["X_test_all_scaled"].reshape(
        scaled_data_all[seed]["X_test_all_scaled"].shape[0],
        scaled_data_all[seed]["X_test_all_scaled"].shape[1],
        1)
    y_train_set = scaled_data_all[seed]["y_train_set"]
    y_test_set  = scaled_data_all[seed]["y_test_set"]

    GRU_model_all = build_gru_all(
        input_len=X_train_GRU_all.shape[1],
        seed=seed
    )

    #Train GRU
    GRU_model_all.fit(
        X_train_GRU_all, y_train_set,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )


    y_prob_train_GRU_all = GRU_model_all.predict(X_train_GRU_all).ravel()
    y_pred_train_GRU_all = (y_prob_train_GRU_all > 0.5).astype(int)

    #Test GRU
    y_prob_GRU_all = GRU_model_all.predict(X_test_GRU_all).ravel()
    y_pred_GRU_all = (y_prob_GRU_all > 0.5).astype(int)

    acc_GRU_all = accuracy_score(y_test_set, y_pred_GRU_all)
    recall_GRU_all = recall_score(y_test_set, y_pred_GRU_all)
    precision_GRU_all = precision_score(y_test_set, y_pred_GRU_all, zero_division=0)
    f1_GRU_all = f1_score(y_test_set, y_pred_GRU_all)
    auc_GRU_all = roc_auc_score(y_test_set, y_prob_GRU_all)


    Accuracy_GRU_all.append(acc_GRU_all)
    Precision_GRU_all.append(precision_GRU_all)
    Recall_GRU_all.append(recall_GRU_all)
    F1_GRU_all.append(f1_GRU_all)
    AUC_GRU_all.append(auc_GRU_all)

    GRU_all_probs_list.append(y_prob_GRU_all)
    GRU_all_train_probs_list.append(y_prob_train_GRU_all)

    print(
        "[GRU-All] "
        f"Acc: {acc_GRU_all:.3f} | "
        f"Recall: {recall_GRU_all:.3f} | "
        f"Precision: {precision_GRU_all:.3f} | "
        f"F1: {f1_GRU_all:.3f} | "
        f"AUC: {auc_GRU_all:.3f}"
    )


# In[56]:


print("\n===== GRU (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_GRU_all):.3f} ± {np.std(Accuracy_GRU_all):.3f}")
print(f"Recall   : {np.mean(Recall_GRU_all):.3f} ± {np.std(Recall_GRU_all):.3f}")
print(f"Precision: {np.mean(Precision_GRU_all):.3f} ± {np.std(Precision_GRU_all):.3f}")
print(f"F1-score : {np.mean(F1_GRU_all):.3f} ± {np.std(F1_GRU_all):.3f}")
print(f"AUC      : {np.mean(AUC_GRU_all):.3f} ± {np.std(AUC_GRU_all):.3f}")


# In[57]:


Define the parameter settings for KAN
def build_kan(input_dim, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    KAN_model_all = KAN(
        width=[input_dim, 4, 1],
        grid=3,
        k=3,
        seed=seed
    )
    return KAN_model_all


# In[58]:


def kan_predict_proba(KAN_model_all, X):
    KAN_model_all.eval()
    with torch.no_grad():
        logits = KAN_model_all(X)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy().ravel()


# In[59]:


Accuracy_KAN_all = []
Precision_KAN_all = []
Recall_KAN_all = []
F1_KAN_all = []
AUC_KAN_all = []

KAN_probs_list_all = []
KAN_train_probs_list_all = []

for seed in seeds:
    print(f"\nSeed {seed} ----------------")
    X_train_KAN_all = torch.tensor(scaled_data_all[seed]["X_train_all_scaled"], dtype=torch.float32)
    y_train_KAN_all = torch.tensor(scaled_data_all[seed]["y_train_set"].values, dtype=torch.float32).view(-1, 1)
    X_test_KAN_all  = torch.tensor(scaled_data_all[seed]["X_test_all_scaled"], dtype=torch.float32)
    y_test_KAN_all  = torch.tensor(scaled_data_all[seed]["y_test_set"].values, dtype=torch.float32).view(-1, 1)


    dataset_all = {
        "train_input": X_train_KAN_all,
        "train_label": y_train_KAN_all,
        "test_input":  X_test_KAN_all,
        "test_label":  y_test_KAN_all
    }

    KAN_model_all = build_kan(
        input_dim=X_train_KAN_all.shape[1],
        seed=seed
    )

    #Train KAN
    KAN_model_all.fit(
        dataset_all,
        steps=50,     
        lr=0.005,
        lamb=0.005,
        loss_fn=nn.BCEWithLogitsLoss()
    )


    y_prob_train_KAN_all = kan_predict_proba(KAN_model_all, X_train_KAN_all)
    y_pred_train_KAN_all = (y_prob_train_KAN_all > 0.5).astype(int)
    y_true_train_KAN_all = y_train_KAN_all.cpu().numpy().ravel()

    #Test KAN
    y_prob_KAN_all = kan_predict_proba(KAN_model_all, X_test_KAN_all)
    y_pred_KAN_all = (y_prob_KAN_all > 0.5).astype(int)
    y_true_KAN_all = y_test_KAN_all.cpu().numpy().ravel()

    acc_KAN_all = accuracy_score(y_true_KAN_all, y_pred_KAN_all)
    recall_KAN_all = recall_score(y_true_KAN_all, y_pred_KAN_all)
    precision_KAN_all = precision_score(y_true_KAN_all, y_pred_KAN_all, zero_division=0)
    f1_KAN_all = f1_score(y_true_KAN_all, y_pred_KAN_all)
    auc_KAN_all = roc_auc_score(y_true_KAN_all, y_prob_KAN_all)

    Accuracy_KAN_all.append(acc_KAN_all)
    Recall_KAN_all.append(recall_KAN_all)
    Precision_KAN_all.append(precision_KAN_all)
    F1_KAN_all.append(f1_KAN_all)
    AUC_KAN_all.append(auc_KAN_all)
    KAN_probs_list_all.append(y_prob_KAN_all)
    KAN_train_probs_list_all.append(y_prob_train_KAN_all)
    print(
        f"Acc: {acc_KAN_all:.3f} | "
        f"Recall: {recall_KAN_all:.3f} | "
        f"Precision: {precision_KAN_all:.3f} | "
        f"F1: {f1_KAN_all:.3f} | "
        f"AUC: {auc_KAN_all:.3f}"
    )


# In[60]:


print("\n===== KAN (mean ± std over seeds) =====")
print(f"Accuracy : {np.mean(Accuracy_KAN_all):.3f} ± {np.std(Accuracy_KAN_all):.3f}")
print(f"Recall   : {np.mean(Recall_KAN_all):.3f} ± {np.std(Recall_KAN_all):.3f}")
print(f"Precision: {np.mean(Precision_KAN_all):.3f} ± {np.std(Precision_KAN_all):.3f}")
print(f"F1-score : {np.mean(F1_KAN_all):.3f} ± {np.std(F1_KAN_all):.3f}")
print(f"AUC      : {np.mean(AUC_KAN_all):.3f} ± {np.std(AUC_KAN_all):.3f}")


# In[61]:


results_df_all = pd.DataFrame({
    "Model": ["MLP", "CNN", "GRU", "KAN"],

    "Accuracy": [
        f"{np.mean(Accuracy_MLP_all):.3f} ± {np.std(Accuracy_MLP_all):.3f}",
        f"{np.mean(Accuracy_CNN_all):.3f} ± {np.std(Accuracy_CNN_all):.3f}",
        f"{np.mean(Accuracy_GRU_all):.3f} ± {np.std(Accuracy_GRU_all):.3f}",
        f"{np.mean(Accuracy_KAN_all):.3f} ± {np.std(Accuracy_KAN_all):.3f}",
    ],

    "Recall": [
        f"{np.mean(Recall_MLP_all):.3f} ± {np.std(Recall_MLP_all):.3f}",
        f"{np.mean(Recall_CNN_all):.3f} ± {np.std(Recall_CNN_all):.3f}",
        f"{np.mean(Recall_GRU_all):.3f} ± {np.std(Recall_GRU_all):.3f}",
        f"{np.mean(Recall_KAN_all):.3f} ± {np.std(Recall_KAN_all):.3f}",
    ],

    "Precision": [
        f"{np.mean(Precision_MLP_all):.3f} ± {np.std(Precision_MLP_all):.3f}",
        f"{np.mean(Precision_CNN_all):.3f} ± {np.std(Precision_CNN_all):.3f}",
        f"{np.mean(Precision_GRU_all):.3f} ± {np.std(Precision_GRU_all):.3f}",
        f"{np.mean(Precision_KAN_all):.3f} ± {np.std(Precision_KAN_all):.3f}",
    ],

    "F1-score": [
        f"{np.mean(F1_MLP_all):.3f} ± {np.std(F1_MLP_all):.3f}",
        f"{np.mean(F1_CNN_all):.3f} ± {np.std(F1_CNN_all):.3f}",
        f"{np.mean(F1_GRU_all):.3f} ± {np.std(F1_GRU_all):.3f}",
        f"{np.mean(F1_KAN_all):.3f} ± {np.std(F1_KAN_all):.3f}",
    ],

    "AUC": [
        f"{np.mean(AUC_MLP_all):.3f} ± {np.std(AUC_MLP_all):.3f}",
        f"{np.mean(AUC_CNN_all):.3f} ± {np.std(AUC_CNN_all):.3f}",
        f"{np.mean(AUC_GRU_all):.3f} ± {np.std(AUC_GRU_all):.3f}",
        f"{np.mean(AUC_KAN_all):.3f} ± {np.std(AUC_KAN_all):.3f}",
    ]
})

results_df_all


# In[62]:


from itertools import combinations
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


base_models = ["MLP_all", "CNN_all", "GRU_all", "KAN_all"]

stacking_results_all = []
Accuracy_stacking_all = []
Precision_stacking_all = []
Recall_stacking_all = []
F1_stacking_all = []
AUC_stacking_all = []
combo_stacking_all = []  
seed_stacking_all = []   

#Prepare meta features for stacking
for seed_idx, seed in enumerate(seeds):

    X_train_meta_all = np.column_stack([
        MLP_all_train_probs_list[seed_idx],
        CNN_all_train_probs_list[seed_idx],
        GRU_all_train_probs_list[seed_idx],
        KAN_train_probs_list_all[seed_idx]
    ])

    X_test_meta_all = np.column_stack([
        MLP_all_probs_list[seed_idx],
        CNN_all_probs_list[seed_idx],
        GRU_all_probs_list[seed_idx],
        KAN_probs_list_all[seed_idx]
    ])

    y_train_set_all = scaled_data_all[seeds[seed_idx]]["y_train_set"].values
    y_test_set_all  = scaled_data_all[seeds[seed_idx]]["y_test_set"].values

    model_col_idx = {name: i for i, name in enumerate(base_models)}


    for r in range(2, len(base_models)+1):  
        for combo in combinations(base_models, r):
            cols = [model_col_idx[m] for m in combo]

            #Train the model 
            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=seed
            )
            xgb.fit(X_train_meta_all[:, cols], y_train_set_all)


            #Test the model
            y_prob_stack_all = xgb.predict_proba(X_test_meta_all[:, cols])[:, 1]
            y_pred_stack_all = (y_prob_stack_all > 0.5).astype(int)


            acc_stacking_all = accuracy_score(y_test_set_all, y_pred_stack_all)
            recall_stacking_all = recall_score(y_test_set_all, y_pred_stack_all)
            precision_stacking_all = precision_score(y_test_set_all, y_pred_stack_all, zero_division=0)
            f1_stacking_all = f1_score(y_test_set_all, y_pred_stack_all)
            auc_stacking_all = roc_auc_score(y_test_set_all, y_prob_stack_all)

            stacking_results_all.append({
                "Seed": seed,
                "Combination": combo,
                "Accuracy": acc_stacking_all,
                "Recall": recall_stacking_all,
                "Precision": precision_stacking_all,
                "F1": f1_stacking_all,
                "AUC": auc_stacking_all
            })
            Accuracy_stacking_all.append(acc_stacking_all)
            Recall_stacking_all.append(recall_stacking_all)
            Precision_stacking_all.append(precision_stacking_all)
            F1_stacking_all.append(f1_stacking_all)
            AUC_stacking_all.append(auc_stacking_all)
            combo_stacking_all.append(combo)
            seed_stacking_all.append(seed)


results_df_all = pd.DataFrame(stacking_results_all)

summary_list_all = []
for comb in results_df_all['Combination'].unique():
    subset = results_df_all[results_df_all['Combination'] == comb]

    summary_list_all.append({
        "Combination": comb,
        "Accuracy": f"{subset['Accuracy'].mean():.3f} ± {subset['Accuracy'].std():.3f}",
        "Recall": f"{subset['Recall'].mean():.3f} ± {subset['Recall'].std():.3f}",
        "Precision": f"{subset['Precision'].mean():.3f} ± {subset['Precision'].std():.3f}",
        "F1": f"{subset['F1'].mean():.3f} ± {subset['F1'].std():.3f}",
        "AUC": f"{subset['AUC'].mean():.3f} ± {subset['AUC'].std():.3f}"
    })

summary_df_all = pd.DataFrame(summary_list_all).sort_values('AUC', ascending=False)
print(summary_df_all)


# In[63]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']

data_numeric_all = summary_df_all.copy()
for m in metrics:
    data_numeric_all[m] = data_numeric_all[m].apply(lambda x: float(x.split(' ± ')[0]))

scaler = MinMaxScaler()
data_scaled_all = pd.DataFrame(
    scaler.fit_transform(data_numeric_all[metrics]),
    index=data_numeric_all['Combination'],
    columns=metrics
)

plt.figure(figsize=(10, max(4, len(data_scaled) * 0.4)))
sns.heatmap(
    data_scaled_all,
    annot=summary_df_all[metrics],  
    fmt="",
    cmap="RdYlGn_r",
    linewidths=0.5,
    cbar_kws={'label': 'Normalized score'}
)

plt.title("Stacking Model Performance Heatmap", fontsize=14)
plt.xlabel("Metrics")
plt.ylabel("Model Combination")
plt.tight_layout()
plt.show()


# In[79]:


#The ablation experiment of BorutaShap
from scipy.stats import ttest_rel, wilcoxon, shapiro

metrics_dict = {
    'KAN': {
        'AUC': {'before': AUC_KAN_all, 'after': AUC_KAN},
        'Accuracy': {'before': Accuracy_KAN_all, 'after': Accuracy_KAN},
        'F1' : {'before':F1_KAN_all, 'after':F1_KAN}
    },
    'MLP': {
        'AUC': {'before': AUC_MLP_all, 'after': AUC_MLP},
        'Accuracy': {'before': Accuracy_MLP_all, 'after': Accuracy_MLP},
        'F1' : {'before':F1_MLP_all, 'after':F1_MLP}
    },
    'CNN': {
        'AUC': {'before': AUC_CNN_all, 'after': AUC_CNN},
        'Accuracy': {'before': Accuracy_CNN_all, 'after': Accuracy_CNN},
        'F1' : {'before':F1_CNN_all, 'after':F1_CNN}
    },
    'GRU': {
        'AUC': {'before': AUC_GRU_all, 'after': AUC_GRU},
        'Accuracy': {'before': Accuracy_GRU_all, 'after': Accuracy_GRU},
        'F1' : {'before':F1_GRU_all, 'after':F1_GRU}
    }
}

metrics = ["Accuracy", "AUC" , "F1"]

results = []

for model, model_metrics in metrics_dict.items():
    for metric in metrics:

        before_vals = np.array(model_metrics[metric]["before"])
        after_vals  = np.array(model_metrics[metric]["after"])
        diff = after_vals - before_vals

        mean_before = before_vals.mean()
        mean_after  = after_vals.mean()

        direction = "Increase" if mean_after > mean_before else "Decrease"

        #Normality test
        if len(diff) >= 3:
            _, p_norm = shapiro(diff)
        else:
            p_norm = 0
        if p_norm > 0.05:
            stat, p_value = ttest_rel(after_vals, before_vals)
            test_used = "Paired t-test"
        else:
            try:
                stat, p_value = wilcoxon(after_vals, before_vals)
                test_used = "Wilcoxon"
            except ValueError:
                stat, p_value = np.nan, np.nan
                test_used = "Wilcoxon"

        results.append({
            "Model": model,
            "Metric": metric,
            "Mean_Before": mean_before,
            "Mean_After": mean_after,
            "Normality_p": p_norm,
            "Test": test_used,
            "p_value": p_value,
        })
results_df_1 = pd.DataFrame(results)
results_df_1 = results_df_1.sort_values(["Model", "Metric"]).reset_index(drop=True)

print(results_df_1)


# In[73]:


#The ablation experiment of BorutaShap

df_before = pd.DataFrame(stacking_results_all)      
df_after  = pd.DataFrame(stacking_results)   

df_before['Combination'] = df_before['Combination'].apply(
    lambda x: '+'.join([m.replace('_all','') for m in x]) if isinstance(x, (tuple,list)) else str(x)
)
df_after['Combination']  = df_after['Combination'].apply(
    lambda x: '+'.join([m.replace('_all','') for m in x]) if isinstance(x, (tuple,list)) else str(x)
)

metrics = ["Accuracy", "AUC", "F1"]
diff_results = []

common_combos = sorted(set(df_before['Combination']).intersection(df_after['Combination']))
if len(common_combos) == 0:
    raise ValueError("No aligned combinations found. Please check your stacking results.")

for combo in common_combos:
    before_subset = df_before[df_before['Combination'] == combo].sort_values('Seed')
    after_subset  = df_after[df_after['Combination'] == combo].sort_values('Seed')

    seeds_before = before_subset['Seed'].values
    seeds_after  = after_subset['Seed'].values
    if not all(seeds_before == seeds_after):
        print(f"Warning: combo {combo} seed mismatch! Skipping.")
        continue

    for metric in metrics:
        before_vals = before_subset[metric].values
        after_vals  = after_subset[metric].values

        diff_vals = after_vals - before_vals

        if len(diff_vals) < 3:  
            use_ttest = False
        else:
            stat, p_normal = shapiro(diff_vals)
            use_ttest = p_normal > 0.05  

        if use_ttest:
            stat_val, p_val = ttest_rel(after_vals, before_vals)
            test_used = "Paired t-test"
        else:
            try:
                stat_val, p_val = wilcoxon(after_vals, before_vals)
            except ValueError:
                stat_val, p_val = None, None
            test_used = "Wilcoxon"

        mean_before = before_vals.mean()
        mean_after  = after_vals.mean()
        direction = "Increase" if mean_after > mean_before else "Decrease"

        diff_results.append({
            "Combination": combo,
            "Metric": metric,
            "Mean_Before": mean_before,
            "Mean_After": mean_after,
            "Normality_p": p_normal,
            "Test": test_used,
            "P_value": p_val
        })

diff_df = pd.DataFrame(diff_results)
diff_df = diff_df.sort_values(["Combination", "Metric"]).reset_index(drop=True)
print(diff_df)


# In[81]:


#CNN + KAN Stacking Ensemble Learning Ablation Experiment
metrics_dict = {
    "MLP": {"Accuracy": Accuracy_MLP, "AUC": AUC_MLP, "F1": F1_MLP},
    "CNN": {"Accuracy": Accuracy_CNN, "AUC": AUC_CNN, "F1": F1_CNN},
    "GRU": {"Accuracy": Accuracy_GRU, "AUC": AUC_GRU, "F1": F1_GRU},
    "KAN": {"Accuracy": Accuracy_KAN, "AUC": AUC_KAN, "F1": F1_KAN}
}

df_stack = pd.DataFrame(stacking_results)

df_stack['Combination'] = df_stack['Combination'].apply(
    lambda x: '+'.join(x) if isinstance(x,(list,tuple)) else str(x)
)

stack_subset = df_stack[df_stack["Combination"] == "CNN+KAN"].sort_values("Seed")

stacking = {
    "Accuracy": stack_subset["Accuracy"].values,
    "AUC": stack_subset["AUC"].values,
    "F1": stack_subset["F1"].values
}
results = []

for model, metrics in metrics_dict.items():

    for metric in ["Accuracy","AUC","F1"]:

        base_vals = np.array(metrics[metric])
        stack_vals = np.array(stacking[metric])

        diff = stack_vals - base_vals

        mean_base = base_vals.mean()
        mean_stack = stack_vals.mean()
        if len(diff) >= 3:
            _, p_norm = shapiro(diff)
        else:
            p_norm = 0
        if p_norm > 0.05:
            stat, p_value = ttest_rel(stack_vals, base_vals)
            test_used = "Paired t-test"
        else:
            try:
                stat, p_value = wilcoxon(stack_vals, base_vals)
                test_used = "Wilcoxon"
            except ValueError:
                stat, p_value = np.nan, np.nan
                test_used = "Wilcoxon"

        results.append({
            "Comparison": f"CNN+KAN vs {model}",
            "Metric": metric,
            "Mean_Stacking": mean_stack,
            "Mean_Base": mean_base,
            "Normality_p": p_norm,
            "Test": test_used,
            "p_value": p_value
        })
results_df_2 = pd.DataFrame(results)

results_df_2 = results_df_2.sort_values(["Comparison","Metric"]).reset_index(drop=True)

print(results_df_2)


# In[ ]:




