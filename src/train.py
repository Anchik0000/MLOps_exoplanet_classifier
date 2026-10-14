import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
EXPERIMENT_NAME = "Kepler_Astrobiology_Hybrid_Model"
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_PATH = Path("data/raw/cumulative.csv")
OUTPUT_CATALOG_PATH = Path("data/processed/discovered_exoplanets.csv")


class AstrobiologyEngine:
    @staticmethod
    def calculate_esi(radius, teq):
        esi_r = (1 - np.abs((radius - 1.0) / (radius + 1.0))) ** 0.57
        esi_t = (1 - np.abs((teq - 255.0) / (teq + 255.0))) ** 1.07
        return np.where((radius > 0) & (teq > 0), np.sqrt(esi_r * esi_t), 0)

    @staticmethod
    def is_habitable_zone(teq):
        return (teq >= 200) & (teq <= 320)

    @staticmethod
    def is_rocky(radius):
        return radius < 1.6

    @staticmethod
    def assign_category(row):
        if row["esi_percent"] >= 75 and row["is_habitable_zone"] and row["is_rocky"]:
            return "Землеподобный супер-кандидат"
        elif row["is_habitable_zone"]:
            return "В зоне жизни (субнептун/газовый)"
        elif row["is_rocky"]:
            return "Каменистая суперземля вне зоны жизни"
        else:
            return "Газовый гигант / горячий юпитер"


def load_and_engineer_features():
    print("Загрузка сырых данных NASA Kepler...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["koi_disposition"]).copy()

    meta_cols = [
        "kepoi_name",
        "kepler_name",
        "koi_disposition",
        "koi_prad",
        "koi_teq",
        "koi_period",
    ]
    meta_df = df[[c for c in meta_cols if c in df.columns]].copy()

    le = LabelEncoder()
    y = le.fit_transform(df["koi_disposition"])
    class_names = [str(c) for c in le.classes_]

    X = df.copy()

    if "koi_depth" in X and "koi_depth_err1" in X and "koi_depth_err2" in X:
        depth_safe = X["koi_depth"].replace(0, np.nan)
        X["depth_rel_err"] = (
            np.abs(X["koi_depth_err1"]) + np.abs(X["koi_depth_err2"])
        ) / (2 * depth_safe)

    if "koi_prad" in X and "koi_prad_err1" in X:
        prad_safe = X["koi_prad"].replace(0, np.nan)
        X["prad_rel_err"] = np.abs(X["koi_prad_err1"]) / prad_safe

    if "koi_srad" in X and "koi_steff" in X:
        X["star_luminosity_proxy"] = (X["koi_srad"] ** 2) * (
            (X["koi_steff"] / 5778) ** 4
        )

    cols_to_drop = [
        col
        for col in X.columns
        if col.startswith(
            (
                "kepid",
                "rowid",
                "kepoi_name",
                "kepler_name",
                "koi_disposition",
                "koi_pdisposition",
                "koi_tce_delivname",
                "koi_score",
            )
        )
    ]
    X = X.drop(columns=cols_to_drop, errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

    valid_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[valid_cols]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta_df, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, class_names, meta_test


def generate_exoplanet_catalog(model, X_test, meta_test, class_names):
    print("Формирование научного каталога экзопланет...")

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    confidences = np.max(y_probs, axis=1) * 100

    catalog = meta_test.copy().reset_index(drop=True)
    catalog["predicted_status"] = [class_names[idx] for idx in y_pred]
    catalog["confidence_percent"] = np.round(confidences, 1)

    raw_esi = AstrobiologyEngine.calculate_esi(catalog["koi_prad"], catalog["koi_teq"])
    catalog["esi_percent"] = np.round(raw_esi * 100, 1)
    catalog["is_habitable_zone"] = AstrobiologyEngine.is_habitable_zone(
        catalog["koi_teq"]
    )
    catalog["is_rocky"] = AstrobiologyEngine.is_rocky(catalog["koi_prad"])
    catalog["planet_category"] = catalog.apply(
        AstrobiologyEngine.assign_category, axis=1
    )

    X_test_reset = X_test.reset_index(drop=True)
    catalog = pd.concat([catalog, X_test_reset], axis=1)

    discovered_planets = catalog[
        catalog["predicted_status"].isin(["CONFIRMED", "CANDIDATE"])
    ].copy()

    discovered_planets = discovered_planets.sort_values(
        by="esi_percent", ascending=False
    )

    OUTPUT_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    discovered_planets.to_csv(OUTPUT_CATALOG_PATH, index=False)
    print(f"Новый датасет сохранен локально: {OUTPUT_CATALOG_PATH}")

    mlflow.log_artifact(str(OUTPUT_CATALOG_PATH), artifact_path="exoplanet_catalog")

    plt.figure(figsize=(9, 5))
    sns.histplot(
        discovered_planets["esi_percent"],
        bins=25,
        color="royalblue",
        kde=True,
    )
    plt.title("Распределение индекса сходства с Землей (ESI %) для найденных планет")
    plt.xlabel("Индекс сходства с Землей (ESI %)")
    plt.ylabel("Число планет")
    plt.axvline(75, color="red", linestyle="--", label="Потенциально обитаемые (>75%)")
    plt.legend()
    chart_file = "esi_distribution.png"
    plt.savefig(chart_file)
    plt.close()
    mlflow.log_artifact(chart_file, "astrobiology_analysis")
    os.remove(chart_file)

    mlflow.log_metrics(
        {
            "total_discovered_planets": len(discovered_planets),
            "habitable_candidates_count": int(
                (discovered_planets["esi_percent"] >= 75).sum()
            ),
            "rocky_planets_count": int(discovered_planets["is_rocky"].sum()),
        }
    )


def log_confusion_matrix(y_true, y_pred, class_names, run_name: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {run_name}")
    plt.ylabel("Истинный класс")
    plt.xlabel("Предсказанный класс")
    file_name = f"cm_{run_name.lower()}.png"
    plt.savefig(file_name)
    plt.close()
    mlflow.log_artifact(file_name, "diagnostic_charts")
    os.remove(file_name)


def run_experiments():
    X_train, X_test, y_train, y_test, class_names, meta_test = (
        load_and_engineer_features()
    )

    with mlflow.start_run(run_name="Run_1_RF_Balanced"):
        X_train_rf = X_train.fillna(X_train.median())
        X_test_rf = X_test.fillna(X_train.median())

        mlflow.log_params({"model": "RandomForest", "class_weight": "balanced"})
        rf = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        )
        rf.fit(X_train_rf, y_train)

        preds = rf.predict(X_test_rf)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(y_test, preds),
                "f1_score": f1,
                "precision": precision_score(
                    y_test, preds, average="weighted", zero_division=0
                ),
                "recall": recall_score(y_test, preds, average="weighted"),
            }
        )
        log_confusion_matrix(y_test, preds, class_names, "rf_balanced")
        mlflow.sklearn.log_model(rf, "model", serialization_format="cloudpickle")
        print("Run 1 (Random Forest) завершен.")

    with mlflow.start_run(run_name="Run_2_HistGradientBoosting"):
        mlflow.log_params(
            {
                "model": "HistGradientBoosting",
                "learning_rate": 0.05,
                "max_iter": 200,
            }
        )
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            class_weight="balanced",
            random_state=42,
        )
        hgb.fit(X_train, y_train)

        preds = hgb.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(y_test, preds),
                "f1_score": f1,
                "precision": precision_score(
                    y_test, preds, average="weighted", zero_division=0
                ),
                "recall": recall_score(y_test, preds, average="weighted"),
            }
        )
        log_confusion_matrix(y_test, preds, class_names, "hist_gbm")

        generate_exoplanet_catalog(hgb, X_test, meta_test, class_names)

        mlflow.sklearn.log_model(hgb, "model", serialization_format="cloudpickle")
        print("Run 2 (HistGradientBoosting + Каталог Экзопланет) завершен.")


if __name__ == "__main__":
    run_experiments()
