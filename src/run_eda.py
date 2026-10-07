import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Kepler_Exoplanet_EDA")

DATA_PATH = Path("data/raw/cumulative.csv")


def calculate_digest(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def main():
    if not DATA_PATH.exists():
        print(f"Ошибка: Файл {DATA_PATH} не найден!")
        return

    print("Загрузка датасета...")
    df = pd.read_csv(DATA_PATH)

    with mlflow.start_run(run_name="Advanced_EDA_Analysis"):
        dataset_source = str(DATA_PATH.absolute())
        dataset_digest = calculate_digest(DATA_PATH)

        dataset = mlflow.data.from_pandas(
            df, source=dataset_source, digest=dataset_digest, name="Kepler_Raw_Data"
        )
        mlflow.log_input(dataset, context="Exploratory Data Analysis")
        print(f"Датасет зарегистрирован. Digest: {dataset_digest}")

        print("Проведение расширенного EDA...")

        df_target_clean = df.dropna(subset=["koi_disposition"]).copy()

        missing_series = df_target_clean.isnull().mean() * 100
        top_missing = (
            missing_series[missing_series > 0].sort_values(ascending=False).head(15)
        )

        fig_missing, ax_missing = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=top_missing.values, y=top_missing.index, palette="mako", ax=ax_missing
        )
        ax_missing.set_title("Топ-15 признаков с наибольшим процентом пропусков (%)")
        ax_missing.set_xlabel("Процент пропусков")
        plt.tight_layout()
        mlflow.log_figure(fig_missing, "eda_plots/missing_values_top15.png")
        plt.close(fig_missing)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.countplot(
            data=df_target_clean, x="koi_disposition", palette="viridis", ax=ax1
        )
        ax1.set_title("Распределение целевого класса (koi_disposition)")
        ax1.set_xlabel("Статус кандидата")
        ax1.set_ylabel("Количество")
        mlflow.log_figure(fig1, "eda_plots/target_distribution.png")
        plt.close(fig1)

        numeric_cols = df_target_clean.select_dtypes(
            include=["float64", "int64"]
        ).columns
        astro_features = [
            col
            for col in numeric_cols
            if not col.endswith(("_err1", "_err2"))
            and not col.startswith(("kepid", "rowid"))
            and col not in ["koi_score"]
        ]

        corr_matrix = df_target_clean[astro_features].corr()
        fig2, ax2 = plt.subplots(figsize=(14, 11))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Корреляция Пирсона"},
            ax=ax2,
        )
        ax2.set_title(
            "Матрица корреляций очищенных астрофизических параметров",
            fontsize=14,
            pad=15,
        )
        ax2.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        mlflow.log_figure(fig2, "eda_plots/correlation_heatmap.png")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_target_clean,
            x="koi_teq",
            y="koi_prad",
            hue="koi_disposition",
            alpha=0.6,
            palette="deep",
            ax=ax3,
        )
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_title(
            "Зависимость радиуса планеты от равновесной температуры (Teq vs Prad)"
        )
        ax3.set_xlabel("Температура равновесия (K, log-scale)")
        ax3.set_ylabel("Радиус планеты (Земные радиусы, log-scale)")
        mlflow.log_figure(fig3, "eda_plots/teq_vs_prad.png")
        plt.close(fig3)

        mlflow.log_param("total_raw_rows", len(df))
        mlflow.log_param("rows_after_target_drop", len(df_target_clean))
        mlflow.log_param("num_features_analyzed", len(astro_features))

        print("Расширенный EDA успешно завершен!")


if __name__ == "__main__":
    main()
