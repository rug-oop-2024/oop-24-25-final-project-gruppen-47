from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np
import pandas as pd


class Pipeline:
    """Pipeline class."""

    def __init__(
        self,
        metrics: List[Metric],
        dataset: Dataset,
        model: Model,
        input_features: List[Feature],
        target_feature: Feature,
        split: float = 0.8,
    ) -> None:
        """
        Initialize the pipeline with the given parameters.

        Args:
            metrics: List of metrics to evaluate the model
            dataset: Dataset
            model: Model
            input_features: List of features to be used as input
            target_feature: Feature to be used as target
            split: float representing the percentage of the dataset \
                    that will go for training
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (
            target_feature.type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model type must be classification "
                "for categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Used to get the model generated during the \
            pipeline execution to be saved
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during
            the pipeline execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Register an artifact.

        Args:
            name: str representing the name of the artifact \
            artifact: artifact to be registered
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocess the features."""
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for feature_name, data, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """Split the data into training and testing sets."""
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            : int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
        ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenate the input vectors into a single matrix.

        Args:
            vectors: List of input vectors
        Returns:
            Numpy array with the concatenated input vectors
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Train the model."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluate the model."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(Y, predictions)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_train_data(self) -> None:
        """Evaluate the model on the training data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._metrics_train_data_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(Y, predictions)
            self._metrics_train_data_results.append((metric, result))

    def execute(self) -> dict:
        """Execute the pipeline."""
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_train_data()
        if self._target_feature.type == "categorical":
            labels = pd.unique(
                self._dataset.read()[self._target_feature.name].values
            )
        else:
            labels = None
        return {
            "metrics_on_evaluation_set": self._metrics_results,
            "metrics_on_training_set": self._metrics_train_data_results,
            "predictions": self._predictions,
            "labels": labels,
        }

    def to_artifact(self, name: str, version: str) -> Artifact:
        """
        Convert the pipeline to an artifact.

        Args:
            name: str representing the name of the artifact
        Returns:
            Artifact: The artifact
        """
        data = pickle.dumps(self)
        return Artifact(
            name=name,
            version=version,
            data=data,
            asset_path=f"{name}.pkl",
            type="pipeline",
        )
