import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import scipy.sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score, make_scorer
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN

from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

from evidently import Report
from evidently.presets import DataDriftPreset

import optuna

# --- Custom Transformers ---
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X,y=None):
        self.col = X.name
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(X)
        return self
    def transform(self, X):
        return self.mlb.transform(X)
    def get_feature_names_out(self, input_features=None):
        return [f"{self.col}_{cls}" for cls in self.mlb.classes_]
    def get_params(self, deep=True):
        return {}
    def set_params(self, **params):
        return self

class AnomalyScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X):
        scores = -self.model.decision_function(X)
        return np.hstack([X, scores.reshape(-1, 1)])

# --- Step 1: Data Preparation ---
def prepare_data(df, is_train=True, model_dir="model_artifacts"):
    df = df.copy()
    
    if is_train:
        df['status'] = df['status'].astype(str).str.upper()
        df = df[df['status'].isin(['CLOSED', 'TERMINATED'])]
        df['label'] = df['status'].map({'CLOSED': 0, 'TERMINATED': 1})
        assert df['label'].notna().all(), "Label column still has NaNs!"

    multilabel_fields = [
        'list_country', 'list_activityType', 'list_deliverableType',
        'list_availableLanguages', 'list_euroSciVocTitle'
    ]

    def extract_intermediate_levels(paths):
        tokens = []
        if isinstance(paths, list):
            for p in paths:
                parts = p.strip('/').split('/')
                tokens.extend(parts[:-1])
        return list(set(tokens))
    df['euroSciVoc_intermediate'] = df['list_euroSciVocPath'].apply(extract_intermediate_levels)
    multilabel_fields.append('euroSciVoc_intermediate')
    
    for col in multilabel_fields:
        df[col] = df[col].apply(lambda x: [] if x is None else (x.tolist() if hasattr(x, 'tolist') else x))
        df[col] = df[col].apply(lambda x: list(x) if not isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: [item for item in x if item is not None])
        df[col] = df[col].apply(lambda x: [str(item).upper() for item in x])

    
    def split_languages(lang_list):
        if not isinstance(lang_list, list):
            return []
        result = []
        for entry in lang_list:
            if isinstance(entry, str):
                result.extend(entry.split(","))
        return result

    df["list_availableLanguages"] = df["list_availableLanguages"].apply(split_languages)
        

    for col in ['title', 'objective']:
        df[col] = df[col].fillna("").astype(str)

    df['n_partners'] = df['list_name'].apply(
        lambda x: len(x.tolist()) if x is not None and hasattr(x, 'tolist') else (len(x) if isinstance(x, list) else 0)
    )

    df['n_country'] = df['list_country'].apply(
        lambda x: len(x.tolist()) if x is not None and hasattr(x, 'tolist') else (len(x) if isinstance(x, list) else 0)
    )

    df['n_sme'] = df['list_SME'].apply(
        lambda x: sum(1 for i in (x.tolist() if hasattr(x, 'tolist') else x) if i is True)
        if x is not None and (hasattr(x, 'tolist') or isinstance(x, list)) else 0
    )

    return df

# --- Step 2: Text Embedding ---
def compute_embeddings(df, text_columns, model_name='sentence-transformers/LaBSE', svd_dim=50):
    model = SentenceTransformer(model_name)
    os.makedirs("/content/drive/MyDrive/model_artifacts", exist_ok=True)
    os.makedirs("/content/drive/MyDrive/embeddings", exist_ok=True)
    for col in text_columns:
        embedding_file = f"/content/drive/MyDrive/embeddings/{col}_embeddings.npy"
        svd_file = f"/content/drive/MyDrive/model_artifacts/{col}_svd.pkl"
        if os.path.exists(embedding_file):
            print(f"Loading saved embeddings for column '{col}'...")
            embeddings = np.load(embedding_file)
        else:
            print(f"Computing embeddings for column '{col}'...")
            embeddings = model.encode(df[col].tolist(), show_progress_bar=True)
            np.save(embedding_file, embeddings)

        print(f"Fitting SVD for column '{col}'...")
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        svd.fit(embeddings)
        joblib.dump(svd, svd_file)

        reduced = svd.transform(embeddings)
        embed_df = pd.DataFrame(reduced, columns=[f'{col}_embed_{i}' for i in range(reduced.shape[1])])
        embed_df.index = df.index  # Force matching index
        df = pd.concat([df, embed_df], axis=1)
    return df


# --- Step 3: Build Preprocessor ---
def build_preprocessor(numeric_features, categorical_features, multilabel_fields):
    numeric_pipeline = SKPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())], memory="cache_dir"
    )

    categorical_pipeline = SKPipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))], memory="cache_dir"
    )

    transformers = [
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
      *[(f'mlb_{col}', MultiLabelBinarizerTransformer(), col) for col in multilabel_fields]]
        

    return ColumnTransformer(transformers, sparse_threshold=0.0)

# --- Step 4: Build Pipeline ---
def build_pipeline(preprocessor, base_model, k=250):
    return ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('anomaly', AnomalyScoreTransformer()),
        ('resample', ADASYN()),
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ('feature_select', SelectKBest(score_func=f_classif, k=k)),
        ('classifier', CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3))
    ])

# --- Step 5: Drift Monitoring ---
def monitor_drift(reference, current, feature_names, output_html='drift_report.html'):
    ref_df = pd.DataFrame(reference, columns=feature_names)
    cur_df = pd.DataFrame(current, columns=feature_names)
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html(output_html)
    print(f"✅ Drift report saved to {output_html}")


# --- Step 6: Evaluation + SHAP ---
def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Evaluation")
    plt.tight_layout()
    plt.show()

    X_proc = model.named_steps['preprocessor'].transform(X_test)
    if scipy.sparse.issparse(X_proc):
        X_proc = X_proc.toarray()

    selector = model.named_steps['feature_select']
    X_selected = selector.transform(X_proc)

    explainer = shap.Explainer(model.named_steps['classifier'].base_estimator, feature_names=feature_names)
    shap_values = explainer(X_selected)
    shap.summary_plot(shap_values, X_selected)

# --- Final Orchestration ---
def status_prediction_model(df):
    os.makedirs("model_artifacts", exist_ok=True)
    print("🧹 Preparing data...")
    df = prepare_data(df, is_train=True)
    print("💡 Embedding text...")
    df = compute_embeddings(df, ['title', 'objective'])

    text_embed_cols = [col for col in df.columns if '_embed_' in col]
    numeric_features = ['durationDays', 'ecMaxContribution', 'totalCost',
                        'n_partners', 'n_country', 'n_sme'] + text_embed_cols
    categorical_features = ['fundingScheme', 'legalBasis', 'nature']
    multilabel_fields =  ['list_country', 'list_activityType', 'list_deliverableType',
        'list_availableLanguages', 'list_euroSciVocTitle','euroSciVoc_intermediate']
    
    
    df = df[numeric_features + categorical_features + multilabel_fields + ['label']]
    X = df.drop(columns='label')
    y = df['label']


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("🧱 Building pipeline...")
    preprocessor = build_preprocessor(numeric_features, categorical_features, multilabel_fields)
    base_model = XGBClassifier(eval_metric='logloss', n_jobs=-1)

    print("🎯 Training model with Optuna...")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 2.0, 10.0)
        }
        base_model.set_params(**params)
        pipeline = build_pipeline(preprocessor, base_model)
        scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                 scoring=make_scorer(f1_score, pos_label=1),n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=24,n_jobs=6)
    best_params = study.best_trial.params
    base_model.set_params(**best_params)

    print("✅ Training final model and evaluating...")
    final_pipeline = build_pipeline(preprocessor, base_model)
    selector = final_pipeline.named_steps['feature_select']
    if hasattr(selector, 'get_support'):
        feature_names = np.array(final_pipeline.named_steps['preprocessor'].get_feature_names_out())[selector.get_support()]
    else:
        feature_names = np.array(final_pipeline.named_steps['preprocessor'].get_feature_names_out())
    evaluate_model(final_pipeline, X_train, X_test, y_train, y_test, feature_names)

    print("📊 Monitoring drift...")
    ref_data = preprocessor.transform(X_train)
    cur_data = preprocessor.transform(X_test)
    if scipy.sparse.issparse(ref_data): ref_data = ref_data.toarray()
    if scipy.sparse.issparse(cur_data): cur_data = cur_data.toarray()
    monitor_drift(pd.DataFrame(ref_data), pd.DataFrame(cur_data), feature_names)
    print("💾 Saving model and artifacts...")
    joblib.dump(final_pipeline, "model_artifacts/model.pkl")
    joblib.dump(preprocessor, "model_artifacts/preprocessor.pkl")
    X_train.to_csv("model_artifacts/X_train_processed.csv", index=False)
    y_train.to_csv("model_artifacts/y_train.csv", index=False)
    feature_config = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "multilabel_fields": multilabel_fields
    }
    json.dump(feature_config, open("model_artifacts/feature_config.json", "w"))
    print("✅ Training complete. Model artifacts saved.")

def score(new_df, model_dir="model_artifacts"):
    # 1) Load artifacts
    pipe = joblib.load(os.path.join(model_dir, "model.pkl"))
    config = json.load(open(os.path.join(model_dir, "feature_config.json")))
    
    # 2) Prepare & embed exactly as in training
    df = prepare_data(new_df.copy(), is_train=False)
    text_cols = ['title', 'objective']
    sbert = SentenceTransformer('sentence-transformers/LaBSE')
    for col in text_cols:
        # load the SVD you trained
        svd = joblib.load(os.path.join(model_dir, f"{col}_svd.pkl"))
        emb = sbert.encode(df[col].tolist(), show_progress_bar=False)
        reduced = svd.transform(emb)
        emb_df = pd.DataFrame(reduced,
                              columns=[f"{col}_embed_{i}" for i in range(reduced.shape[1])],
                              index=df.index)
        df = pd.concat([df, emb_df], axis=1)

    # 3) Build the final feature set
    X = df[ config["numeric_features"]
          + config["categorical_features"]
          + config["multilabel_fields"] ]

    # 4) Predict & attach to DataFrame
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)[:, 1]   # assume binary and positive class = index 1
    df["predicted_label"] = preds
    df["predicted_prob"]  = probs

    # 5) SHAP explanations on the *selected* features
    #    (we need to re-run preprocessing + feature_selection)
    preproc = pipe.named_steps["preprocessor"]
    select  = pipe.named_steps["feature_select"]
    clf     = pipe.named_steps["classifier"].base_estimator

    X_proc = preproc.transform(X)
    if scipy.sparse.issparse(X_proc):
        X_proc = X_proc.toarray()
    X_sel = select.transform(X_proc)

    feature_names = select.get_feature_names_out(
        preproc.get_feature_names_out()
    )

    # Use a TreeExplainer directly on the XGB base estimator
    explainer = shap.Explainer(clf, X_sel, feature_names=feature_names)
    shap_vals = explainer(X_sel)   # returns a ShapleyValues object

    # 6) For each row, pick top-3 absolute contributors
    shap_df = pd.DataFrame(shap_vals.values, columns=feature_names, index=df.index)
    abs_shap = shap_df.abs()

    top_feats = abs_shap.apply(lambda row: row.nlargest(4).index.tolist(), axis=1)
    top_vals  = abs_shap.apply(lambda row: row.nlargest(4).values.tolist(), axis=1)

    df[["top1_feature","top2_feature","top3_feature","top4_feature"]] = pd.DataFrame(
        top_feats.tolist(), index=df.index
    )
    df[["top1_shap","top2_shap","top3_shap","top4_shap"]] = pd.DataFrame(
        top_vals.tolist(),  index=df.index
    )

    return df

if __name__ == "__main__":
    df = pd.read_csv("your_data.csv")

    status_prediction_model(df)

    new_df = pd.read_csv("new_data.csv")
    scored_df = score(new_df)
    print(scored_df.head())