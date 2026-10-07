import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Kepler_Exoplanet_EDA")

DATA_PATH = Path("data/raw/cumulative.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


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

    with mlflow.start_run(run_name="Initial_EDA_Analysis"):
        dataset_source = str(DATA_PATH.absolute())
        dataset_digest = calculate_digest(DATA_PATH)

        dataset = mlflow.data.from_pandas(
            df, source=dataset_source, digest=dataset_digest, name="Kepler_Raw_Data"
        )
        mlflow.log_input(dataset, context="Exploratory Data Analysis")
        print(f"Датасет зарегистрирован. Digest: {dataset_digest}")

        print("Проведение EDA и генерация графиков...")

        df_clean = df.dropna(subset=["koi_disposition"])

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_clean, x="koi_disposition", palette="viridis")
        plt.title("Распределение классов экзопланет (Target Distribution)")
        plt.xlabel("Статус объекта")
        plt.ylabel("Количество")
        plot1_path = ARTIFACTS_DIR / "target_distribution.png"
        plt.savefig(plot1_path)
        plt.close()

        features_to_corr = [
            "koi_score",
            "koi_period",
            "koi_prad",
            "koi_depth",
            "koi_teq",
            "koi_insol",
        ]
        corr_matrix = df_clean[features_to_corr].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1
        )
        plt.title("Тепловая матрица корреляций физических признаков")
        plot2_path = ARTIFACTS_DIR / "correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(plot2_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df_clean,
            x="koi_period",
            y="koi_prad",
            hue="koi_disposition",
            alpha=0.6,
            palette="deep",
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Зависимость радиуса планеты от орбитального периода")
        plt.xlabel("Орбитальный период (дни, log-scale)")
        plt.ylabel("Радиус планеты (радиусы Земли, log-scale)")
        plot3_path = ARTIFACTS_DIR / "period_vs_radius.png"
        plt.savefig(plot3_path)
        plt.close()

        print("Отправка графиков в MLflow...")
        mlflow.log_artifact(str(plot1_path.resolve()), artifact_path="eda_plots")
        mlflow.log_artifact(str(plot2_path.resolve()), artifact_path="eda_plots")
        mlflow.log_artifact(str(plot3_path.resolve()), artifact_path="eda_plots")

        mlflow.log_param("total_rows", len(df_clean))
        mlflow.log_param("total_columns", len(df_clean.columns))

        print("EDA завершен! Графики загружены в MLflow.")


if __name__ == "__main__":
    main()
