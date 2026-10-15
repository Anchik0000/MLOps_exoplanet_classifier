import asyncio
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import structlog
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_LOAD_ATTEMPTS,
    MODEL_LOAD_DELAY_SECONDS,
    MODEL_NAME,
)
from src.features import add_engineered_features

logger = structlog.get_logger()

DEFAULT_CLASS_NAMES = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]


@dataclass
class ModelService:
    model: object
    class_names: list[str]
    model_name: str
    model_version: str

    @property
    def feature_names(self) -> list[str]:
        return [str(name) for name in self.model.feature_names_in_]

    def knows_any_feature(self, features: dict[str, float | None]) -> bool:
        return bool(set(features) & set(self.feature_names))

    def predict(self, features: dict[str, float | None]) -> dict:
        frame = pd.DataFrame([features], dtype="float64")
        frame = add_engineered_features(frame)
        frame = frame.reindex(columns=self.feature_names)

        probabilities = self.model.predict_proba(frame)[0]
        best = int(np.argmax(probabilities))

        return {
            "prediction": self.class_names[int(self.model.classes_[best])],
            "probabilities": {
                self.class_names[int(label)]: round(float(prob), 4)
                for label, prob in zip(self.model.classes_, probabilities, strict=True)
            },
            "missing_features": int(frame.isna().sum().sum()),
        }


def load_model_service() -> ModelService:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_TRACKING_URI
    )
    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

    run = client.get_run(version.run_id)
    names = run.data.params.get("class_names")
    class_names = names.split(",") if names else DEFAULT_CLASS_NAMES

    return ModelService(
        model=model,
        class_names=class_names,
        model_name=MODEL_NAME,
        model_version=str(version.version),
    )


async def load_model_with_retries() -> ModelService | None:
    for attempt in range(1, MODEL_LOAD_ATTEMPTS + 1):
        try:
            service = await asyncio.to_thread(load_model_service)
        except (MlflowException, OSError, ValueError, ImportError) as exc:
            logger.error(
                "model_load_failed",
                attempt=attempt,
                attempts=MODEL_LOAD_ATTEMPTS,
                error=str(exc),
            )
            if attempt < MODEL_LOAD_ATTEMPTS:
                await asyncio.sleep(MODEL_LOAD_DELAY_SECONDS)
            continue

        logger.info(
            "model_loaded",
            model=service.model_name,
            alias=MODEL_ALIAS,
            version=service.model_version,
        )
        return service

    return None
