from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "student_dropout_risk_2026.csv"
MODEL_PATH = APP_DIR / "student_dropout_model_9_features.pkl"

FEATURE_COLUMNS = [
    "age",
    "gender",
    "cgpa",
    "attendance_rate",
    "mental_health_support",
    "sleep_hours",
    "stress_level",
    "extracurricular_activities",
    "study_mode",
]
NUMERIC_FEATURES = [
    "age",
    "cgpa",
    "attendance_rate",
    "mental_health_support",
    "sleep_hours",
    "stress_level",
    "extracurricular_activities",
]
CATEGORICAL_FEATURES = ["gender", "study_mode"]


def main():
    data = pd.read_csv(DATA_PATH)
    data["cgpa"] = (
        data[["gpa_semester_1", "gpa_semester_2"]].mean(axis=1) * 2.5
    )
    features = data[FEATURE_COLUMNS]
    target = data["dropout"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    model.fit(train_features, train_target)

    dropout_class_index = list(model.classes_).index(1)
    test_probabilities = model.predict_proba(test_features)[:, dropout_class_index]
    metrics = {
        "test_accuracy": float(
            accuracy_score(test_target, model.predict(test_features))
        ),
        "test_roc_auc": float(roc_auc_score(test_target, test_probabilities)),
        "test_rows": int(len(test_target)),
    }

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics,
            "cgpa_definition": (
                "Mean of semester GPAs on a 0-4 scale, multiplied by 2.5 "
                "to match the app's 0-10 model scale"
            ),
        },
        MODEL_PATH,
    )
    print(f"Saved nine-feature model to {MODEL_PATH}")
    print(f"Holdout evaluation: {metrics}")


if __name__ == "__main__":
    main()
