# import relevant packages
import pandas as pd
import numpy as np 
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# reading the dataset
df = pd.read_csv('/Users/chloeong/Downloads/diabetes_prediction_dataset.csv')

# data processing and feature engineering
# merging features 
df["smoking_history"] = np.where(df["smoking_history"].str.contains("No Info", na=False),"never",
                        np.where(df["smoking_history"].str.contains("ever", na=False), "former",
                        np.where(df["smoking_history"].str.contains("not current", na=False), "former",
                        df["smoking_history"])))

# removing outliers
z_scores = df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']].apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
df = df[~outliers]

# encode categorical variables 
df = pd.get_dummies(df, columns=['gender', 'smoking_history']).astype(int)

# define model dataset and features 
x = df.drop(columns=["diabetes"])
y = df["diabetes"]

# -------------------------------------------------------------------------
# Model 1: Random Forest

# use of Optuna tuning with hyperparameters suggestions
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 32)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    criterion = trial.suggest_categorical('criterion', ["gini", "entropy"]) 

    # use Stratified K-Fold to split into training and testing sets
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    f1_scores = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # apply SMOTE to balance the data
        minority_class = sum(y_train == 1)
        smote_strategy = {1: int(minority_class * 1.4), 0: sum(y_train == 0)}
        oversample = SMOTEENN(sampling_strategy=smote_strategy, random_state=22)
        x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

        # apply Tomek Links to remove majority instances that are near the minority instances
        tomek_links = TomekLinks(sampling_strategy='auto')
        x_final, y_final = tomek_links.fit_resample(x_resampled, y_resampled)

        # define the Random Forest model with hyperparameters from Optuna
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=22
        )

        # fit the model and make predictions
        model.fit(x_final, y_final)
        y_pred = model.predict(x_test)

        # use f1_score to evaluate since class distribution is imbalanced
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

# search for the best params for RF
study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=5)

# output best param
print("Best trial:")
trial = study.best_trial
print("Value: {:.4f}".format(trial.value))
print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# -------------------------------------------------------------------------
# Model 2:  AdaBoost

def objective_gradient_boosting(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    f1_scores = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # apply SMOTE to balance the data
        minority_class = sum(y_train == 1)
        smote_strategy = {1: int(minority_class * 1.4), 0: sum(y_train == 0)}
        oversample = SMOTEENN(sampling_strategy=smote_strategy, random_state=22)
        x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

        # apply Tomek Links to remove majority instances that are near the minority instances
        tomek_links = TomekLinks(sampling_strategy='auto')
        x_final, y_final = tomek_links.fit_resample(x_resampled, y_resampled)

        # define the Gradient Boosting model with hyperparameters from Optuna
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=22
        )

        # fit the model and make predictions
        model.fit(x_final, y_final)
        y_pred = model.predict(x_test)

        # use f1_score to evaluate since class distribution is imbalanced
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

# search for the best params for Gradient Boosting
study_gradient_boosting = optuna.create_study(direction='maximize')  
study_gradient_boosting.optimize(objective_gradient_boosting, n_trials=5)

# output best params
print("Best trial for Gradient Boosting:")
trial_gradient_boosting = study_gradient_boosting.best_trial
print("Value: {:.4f}".format(trial_gradient_boosting.value))
print("Params: ")
for key, value in trial_gradient_boosting.params.items():
    print("    {}: {}".format(key, value))
# -------------------------------------------------------------------------
# Model 3: XGBoost 

def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0, 5)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    f1_scores = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # apply SMOTE to balance the data
        minority_class = sum(y_train == 1)
        smote_strategy = {1: int(minority_class * 1.4), 0: sum(y_train == 0)}
        oversample = SMOTEENN(sampling_strategy=smote_strategy, random_state=22)
        x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)

        # apply Tomek Links to remove majority instances that are near the minority instances
        tomek_links = TomekLinks(sampling_strategy='auto')
        x_final, y_final = tomek_links.fit_resample(x_resampled, y_resampled)

        # define the XGB with hyperparameters from Optuna
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=22,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        # fit the model and make predictions
        model.fit(x_final, y_final)
        y_pred = model.predict(x_test)

        # use f1_score to evaluate since class distribution is imbalanced
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

# search for the best params for XGBoost
study_xgb = optuna.create_study(direction='maximize')  
study_xgb.optimize(objective_xgb, n_trials=5)

# output best param
print("Best trial for XGBoost:")
trial_xgb = study_xgb.best_trial
print("Value: {:.4f}".format(trial_xgb.value))
print("Params: ")
for key, value in trial_xgb.params.items():
    print("    {}: {}".format(key, value))
# -------------------------------------------------------------------------
# Meta Model: LightGBM

best_rf_params = trial.params
best_grad_params = trial_gradient_boosting.params
best_xgb_params = trial_xgb.params

# build models
rf_model = RandomForestClassifier(**best_rf_params)
grad_model = GradientBoostingClassifier(**best_grad_params)
xgb_model = xgb.XGBClassifier(**best_xgb_params)

# split data into training and test sets
x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=22, stratify=y)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
stacked_train = np.zeros((x_train_full.shape[0], 3)) 
stacked_test = np.zeros((x_test.shape[0], 3)) 

# training the base models and generating predictions
for fold, (train_idx, valid_idx) in enumerate(kf.split(x_train_full, y_train_full)):
    x_train, x_valid = x_train_full.iloc[train_idx], x_train_full.iloc[valid_idx]
    y_train, y_valid = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

    # RF
    rf_model.fit(x_train, y_train)
    stacked_train[valid_idx, 0] = rf_model.predict_proba(x_valid)[:, 1]  # Probability for class 1
    stacked_test[:, 0] += rf_model.predict_proba(x_test)[:, 1] / kf.n_splits  # Averaging for test set

    # GradientBoost
    grad_model.fit(x_train, y_train)
    stacked_train[valid_idx, 1] = grad_model.predict_proba(x_valid)[:, 1]
    stacked_test[:, 1] += grad_model.predict_proba(x_test)[:, 1] / kf.n_splits

    # XGBoost
    xgb_model.fit(x_train, y_train)
    stacked_train[valid_idx, 2] = xgb_model.predict_proba(x_valid)[:, 1]
    stacked_test[:, 2] += xgb_model.predict_proba(x_test)[:, 1] / kf.n_splits

# train meta-model using the stacked predictions
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(stacked_train, y_train_full)  

# make predictions with the meta-model
y_meta_pred = lgb_model.predict(stacked_test)

# model evaluation
report = classification_report(y_test, y_meta_pred)
print(report)

# -------------------------------------------------------------------------
import joblib

# save model 
joblib.dump(rf_model, '/Users/chloeong/diabetes prediction/rf_model.pkl')
joblib.dump(grad_model, '/Users/chloeong/diabetes prediction/grad_model.pkl')
joblib.dump(xgb_model, '/Users/chloeong/diabetes prediction/xgb_model.pkl')
joblib.dump(lgb_model, '/Users/chloeong/diabetes prediction/lightgbm_meta_model.pkl')


  