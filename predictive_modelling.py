import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import scipy.sparse
import polars as pl
import re
import gcsfs

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

# Transformer for binarizing multi-label columns (lists of categories per row)
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to binarize multi-label (list of strings) columns.

    Attributes:
        col (str): The column name being transformed.
        mlb (MultiLabelBinarizer): Fitted binarizer.
    """
    def fit(self, X, y=None):
        """Fit the binarizer to the data.

        Args:
            X (pd.Series): Series of lists to binarize.
            y (ignored): Not used.

        Returns:
            self
        """
        self.col = X.name
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(X)
        return self

    def transform(self, X):
        """Transform the input Series to a binary matrix.

        Args:
            X (pd.Series): Series of lists to transform.

        Returns:
            np.ndarray: Binary matrix for multi-label data.
        """
        return self.mlb.transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for the binarized columns.

        Args:
            input_features (ignored): Not used.

        Returns:
            list: List of output feature names.
        """
        return [f"{self.col}_{cls}" for cls in self.mlb.classes_]

    def get_params(self, deep=True):
        """Get parameters (stub for sklearn compatibility)."""
        return {}

    def set_params(self, **params):
        """Set parameters (stub for sklearn compatibility)."""
        return self


# Adds anomaly score from IsolationForest as a feature (for noise/outlier detection)
class AnomalyScoreTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to compute anomaly scores using IsolationForest
    and add them as a new feature.
    """
    def __init__(self):
        self.model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)

    def fit(self, X, y=None):
        """Fit the IsolationForest on input data.

        Args:
            X (array-like): Input features.
            y (ignored): Not used.

        Returns:
            self
        """
        self.model.fit(X)
        return self

    def transform(self, X):
        """Transform the data by appending the anomaly scores.

        Args:
            X (array-like): Input features.

        Returns:
            np.ndarray: Input features with anomaly score column appended.
        """
        scores = -self.model.decision_function(X)
        return np.hstack([X, scores.reshape(-1, 1)])

# --- Step 1: Data Preparation ---
def prepare_data(df, is_train=True, model_dir="model_artifacts"):
    """Prepare and clean the raw input DataFrame for modeling.

    - Maps status to binary label if training.
    - Handles multilabel columns.
    - Cleans and expands list fields.
    - Adds count features.

    Args:
        df (pd.DataFrame): Raw data.
        is_train (bool): Whether the data is for training.
        model_dir (str): Directory to store artifacts (not used here).

    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame.
    """
    df = df.copy()

    if is_train:
        # Filter for only labeled classes and map to numeric
        df['status'] = df['status'].astype(str).str.upper()
        df = df[df['status'].isin(['CLOSED', 'TERMINATED'])]
        df['label'] = df['status'].map({'CLOSED': 0, 'TERMINATED': 1})
        assert df['label'].notna().all(), "Label column still has NaNs!"

    # Define fields that are lists of values (multi-label columns)
    multilabel_fields = [
        'list_country', 'list_activityType', 'list_deliverableType',
        'list_availableLanguages', 'list_euroSciVocTitle'
    ]

    # Helper to extract intermediate paths (for hierarchy/ontology features)
    def extract_intermediate_levels(paths):
        tokens = []
        if isinstance(paths, list):
            for p in paths:
                parts = p.strip('/').split('/')
                tokens.extend(parts[:-1])
        return list(set(tokens))
    df['euroSciVoc_intermediate'] = df['list_euroSciVocPath'].apply(extract_intermediate_levels)
    multilabel_fields.append('euroSciVoc_intermediate')

    # Normalize and clean multi-label fields
    for col in multilabel_fields:
        df[col] = df[col].apply(lambda x: [] if x is None else (x.tolist() if hasattr(x, 'tolist') else x))
        df[col] = df[col].apply(lambda x: list(x) if not isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: [item for item in x if item is not None])
        df[col] = df[col].apply(lambda x: [str(item).upper() for item in x])

    # Split language field (if comma-separated)
    def split_languages(lang_list):
        if not isinstance(lang_list, list):
            return []
        result = []
        for entry in lang_list:
            if isinstance(entry, str):
                result.extend(entry.split(","))
        return result
    df["list_availableLanguages"] = df["list_availableLanguages"].apply(split_languages)

    # Fill NA and convert to string for text columns
    for col in ['title', 'objective']:
        df[col] = df[col].fillna("").astype(str)

    # Count number of partners, countries, SMEs (for feature engineering)
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
    """Compute SBERT embeddings for text columns, reduce with SVD, and add to DataFrame.

    Embeddings are cached to disk for re-use. SVD is fitted per column.

    Args:
        df (pd.DataFrame): DataFrame with text columns.
        text_columns (list of str): Columns to embed.
        model_name (str): HuggingFace model name.
        svd_dim (int): Number of SVD components.

    Returns:
        pd.DataFrame: DataFrame with added embedding columns.
    """
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
        embed_df.index = df.index  # Keep index aligned for merge
        df = pd.concat([df, embed_df], axis=1)
    return df

# --- Step 3: Build Preprocessor ---
def build_preprocessor(numeric_features, categorical_features, multilabel_fields):
    """Create a ColumnTransformer that preprocesses numeric, categorical, and multilabel fields.

    Args:
        numeric_features (list of str): Names of numeric columns.
        categorical_features (list of str): Names of categorical columns.
        multilabel_fields (list of str): Names of multi-label columns.

    Returns:
        ColumnTransformer: Configured sklearn transformer.
    """
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
        # Add a binarizer transformer for each multilabel field
        *[(f'mlb_{col}', MultiLabelBinarizerTransformer(), col) for col in multilabel_fields]]

    return ColumnTransformer(transformers, sparse_threshold=0.0)

# --- Step 4: Build Pipeline ---
def build_pipeline(preprocessor, base_model, k=250):
    """Build the full ML pipeline including preprocessing, anomaly detection, 
    resampling, feature selection, and a calibrated classifier.

    Args:
        preprocessor (ColumnTransformer): Preprocessing transformer.
        base_model (sklearn estimator): Base classifier (e.g. XGBClassifier).
        k (int): Number of features to select.

    Returns:
        ImbPipeline: Configured imbalanced-learn pipeline.
    """
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
    """Generate a drift report using Evidently comparing reference and current data.

    Args:
        reference (np.ndarray or pd.DataFrame): Reference feature matrix.
        current (np.ndarray or pd.DataFrame): Current feature matrix.
        feature_names (list of str): Feature names.
        output_html (str): Output path for HTML report.

    Returns:
        None
    """
    ref_df = pd.DataFrame(reference, columns=feature_names)
    cur_df = pd.DataFrame(current, columns=feature_names)

    report = Report(metrics=[DataDriftPreset()])
    evaluated_report = report.run(reference_data=ref_df, current_data=cur_df)
    evaluated_report.save_html(output_html)
    print(f"✅ Drift report saved to {output_html}")

# --- Step 6: Evaluation + SHAP ---
def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Print evaluation metrics, show confusion matrix, and plot SHAP summary.

    Args:
        model (sklearn Pipeline): Trained pipeline.
        X_train (np.ndarray or pd.DataFrame): Training features.
        X_test (np.ndarray or pd.DataFrame): Test features.
        y_train (np.ndarray or pd.Series): Training labels.
        y_test (np.ndarray or pd.Series): Test labels.
        feature_names (list of str): Feature names after selection.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Evaluation")
    plt.tight_layout()
    plt.show()

    # Prepare data for SHAP after feature selection
    X_proc = model.named_steps['preprocessor'].transform(X_test)
    if scipy.sparse.issparse(X_proc):
        X_proc = X_proc.toarray()

    selector = model.named_steps['feature_select']
    X_selected = selector.transform(X_proc)
    explainer = shap.TreeExplainer(model.named_steps['classifier'].calibrated_classifiers_[0].estimator, feature_names=feature_names)
    shap_values = explainer(X_selected)
    shap.summary_plot(shap_values, X_selected)

# --- Final Orchestration ---
def status_prediction_model(df):
    """Orchestrate end-to-end model training, evaluation, drift detection, and saving artifacts.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        None
    """
    os.makedirs("model_artifacts", exist_ok=True)
    print("🧹 Preparing data...")
    df = prepare_data(df, is_train=True)
    print("💡 Embedding text...")
    df = compute_embeddings(df, ['title', 'objective'])

    # Feature lists for the pipeline
    text_embed_cols = [col for col in df.columns if '_embed_' in col]
    numeric_features = ['durationDays', 'ecMaxContribution', 'totalCost',
                        'n_partners', 'n_country', 'n_sme'] + text_embed_cols
    categorical_features = ['fundingScheme', 'legalBasis', 'nature']
    multilabel_fields =  ['list_country', 'list_activityType', 'list_deliverableType',
        'list_availableLanguages', 'list_euroSciVocTitle','euroSciVoc_intermediate']

    # Restrict to used features + label
    df = df[numeric_features + categorical_features + multilabel_fields + ['label']]
    X = df.drop(columns='label')
    y = df['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("🧱 Building pipeline...")
    preprocessor = build_preprocessor(numeric_features, categorical_features, multilabel_fields)
    base_model = XGBClassifier(eval_metric='logloss', n_jobs=-1)

    # Optuna hyperparameter tuning (24 trials, 6 jobs parallel)
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
        scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                                 scoring=make_scorer(f1_score, pos_label=1),n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=24,n_jobs=6)
    best_params = study.best_trial.params
    base_model.set_params(**best_params)

    # Final fit and evaluation
    print("✅ Training final model and evaluating...")
    final_pipeline = build_pipeline(preprocessor, base_model)
    final_pipeline.fit(X_train, y_train)
    selector = final_pipeline.named_steps['feature_select']
    if hasattr(selector, 'get_support'):
        feature_names = np.array(final_pipeline.named_steps['preprocessor'].get_feature_names_out())[selector.get_support()]
    else:
        feature_names = np.array(final_pipeline.named_steps['preprocessor'].get_feature_names_out())
    evaluate_model(final_pipeline, X_train, X_test, y_train, y_test, feature_names)

    # Drift monitoring (train vs test data)
    print("📊 Monitoring drift...")
    X_train_p = preprocessor.transform(X_train)
    X_test_p  = preprocessor.transform(X_test)
    if scipy.sparse.issparse(X_train_p): X_train_p = X_train_p.toarray()
    if scipy.sparse.issparse(X_test_p):  X_test_p  = X_test_p.toarray()
    selector = final_pipeline.named_steps["feature_select"]
    X_train_sel = selector.transform(X_train_p)
    X_test_sel  = selector.transform(X_test_p)
    monitor_drift(X_train_sel, X_test_sel, feature_names)
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
    """Score new/unseen data using the trained pipeline and explain predictions via SHAP.

    Steps:
    - Loads model and configs
    - Prepares data and embeds text
    - Predicts label and probability
    - Computes SHAP values and identifies top contributing features

    Args:
        new_df (pd.DataFrame): Data to score.
        model_dir (str): Path to saved model artifacts.

    Returns:
        pd.DataFrame: DataFrame with predictions, probabilities, and SHAP features.
    """
    # 1) Load trained model and config
    pipe = joblib.load(os.path.join(model_dir, "model.pkl"))
    config = json.load(open(os.path.join(model_dir, "feature_config.json")))

    # 2) Prepare & embed exactly as in training
    df = prepare_data(new_df.copy(), is_train=False)
    text_cols = ['title', 'objective']
    sbert = SentenceTransformer('sentence-transformers/LaBSE')
    os.makedirs("/content/drive/MyDrive/embeddings_test", exist_ok=True)
    for col in text_cols:
        embedding_file = f"/content/drive/MyDrive/embeddings_test/{col}_embeddings.npy"
        # load the SVD you trained
        svd = joblib.load(os.path.join(model_dir, f"{col}_svd.pkl"))
        if os.path.exists(embedding_file):
            print(f"Loading saved embeddings for column '{col}'...")
            embeddings = np.load(embedding_file)
        else:
            print(f"Computing embeddings for column '{col}'...")
            embeddings = sbert.encode(df[col].tolist(), show_progress_bar=True)
            np.save(embedding_file, embeddings)
        reduced = svd.transform(embeddings)
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
    clf     = pipe.named_steps["classifier"].calibrated_classifiers_[0].estimator

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

    # 6) For each row, pick top-6 absolute contributors
    shap_df = pd.DataFrame(shap_vals.values, columns=feature_names, index=df.index)

    # 7) get absolute values
    abs_shap = shap_df.abs()

    # 8) for each row, record the top‐6 feature names by absolute magnitude
    top_feats = abs_shap.apply(lambda row: row.nlargest(6).index.tolist(), axis=1)

    # 9) convert that to six separate columns
    feat_cols = [f"top{i}_feature" for i in range(1,7)]
    df[feat_cols] = pd.DataFrame(top_feats.tolist(), index=df.index)

    # 10) now build the *true* SHAP values by looking up each name in shap_df
    #    for each row, shap_df.loc[idx, feat] is the signed value
    top_vals = [
        [ shap_df.loc[idx, feat] for feat in feats ]
        for idx, feats in top_feats.items()
    ]

    # 11) store them in your six shap‐value columns
    val_cols = [f"top{i}_shap" for i in range(1,7)]
    df[val_cols] = pd.DataFrame(top_vals, index=df.index)

    return df

def clean_feature_name(raw: str) -> str:
    """
    - cat__:   "cat__feature_value"       → "Feature: Value"
    - num__:   "num__some_count"           → "Some Count"
    - mlb_:    "mlb_list_activityType__list_activityType_Research"
                 → "List Activity Type: Research"
    """
    if not raw:
        return ""

    # 1) cat__
    if raw.startswith("cat__"):
        s = raw[len("cat__"):]
        col, val = (s.split("__", 1) + [None])[:2]
        col_c = col.replace("_", " ").title()
        if val:
            val_c = val.replace("_", " ").title()
            return f"{col_c}: {val_c}"
        return col_c

    # 2) num__
    if raw.startswith("num__"):
        s = raw[len("num__"):]
        return s.replace("_", " ").replace('n ','Number of ')

    # 3) mlb_
    if raw.startswith("mlb_"):
        s = raw[len("mlb_"):]
        col_part, val_part = (s.split("__", 1) + [None])[:2]
        # drop leading "list_" on the column
        if col_part.startswith("list_"):
            col_inner = col_part[len("list_"):]
        else:
            col_inner = col_part
        col_c = col_inner.replace("_", " ").title()
        col_c = "List " + col_c

        if val_part:
            # drop "list_{col_inner}_" or leading "list_"
            prefix = f"list_{col_inner}_"
            if val_part.startswith(prefix):
                val_inner = val_part[len(prefix):]
            elif val_part.startswith("list_"):
                val_inner = val_part[len("list_"):]
            else:
                val_inner = val_part
            val_c = val_inner.replace("_", " ").title()
            return f"{col_c}: {val_c}"
        return col_c

    # fallback: replace __ → ": ", _ → " "
    return raw.replace("__", ": ").replace("_", " ").title()


def preprocess_feature_names(df: pl.DataFrame) -> pl.DataFrame:
    transforms = []

    # clean and round top-6 features & shap values
    for i in range(1, 7):
        fcol = f"top{i}_feature"
        scol = f"top{i}_shap"

        if fcol in df.columns:
            transforms.append(
                pl.col(fcol)
                  .map_elements(clean_feature_name, return_dtype=pl.Utf8)
                  .alias(fcol)
            )
        if scol in df.columns:
            transforms.append(
                pl.col(scol)
                  .round(4)
                  .alias(scol)
            )

    # round overall predicted probability
    if "predicted_prob" in df.columns:
        transforms.append(
            pl.col("predicted_prob")
              .round(4)
              .alias("predicted_prob")
        )

    # 1) build the full list of embed-columns
    embed_cols = [f"title_embed_{i}"     for i in range(50)] + \
                [f"objective_embed_{i}" for i in range(50)]

    # 2) keep only the ones that actually exist in df.columns
    to_drop = [c for c in embed_cols if c in df.columns]

    # 3) drop them
    df = df.drop(to_drop)
    return df.with_columns(transforms)

if __name__ == "__main__":
    # Entry point for training and scoring: loads data from Google Cloud Storage,
    # builds model and artifacts, then scores the same data as a test.
    bucket = "mda_eu_project"
    path   = "data/consolidated_clean.parquet"
    uri    = f"gs://{bucket}/{path}"

    fs = gcsfs.GCSFileSystem()

    with fs.open(uri, "rb") as f:
        df = pl.read_parquet(f).to_pandas()
    
    status_prediction_model(df)
    df_clean = preprocess_feature_names(pl.from_pandas(score(df)))
    df_clean.head(10)