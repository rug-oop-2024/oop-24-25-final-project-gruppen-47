from abc import ABC, abstractmethod
from typing import Dict, List, Type
import numpy as np


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
]


class Metric(ABC):
    """Abstract Base class for all metrics."""

    @abstractmethod
    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the metric.
        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: Metric value.
        """
        pass

    def __call__(self) -> float:
        """
        Invokes the evaluate method when the instance is called as a function.

        Returns:
            The result of the evaluate method.
        """
        return self.evaluate()


class Accuracy(Metric):
    """Description: Metric to evaluate accuracy."""

    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the accuracy.

        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: Accuracy.
        """
        true = np.argmax(true, axis=1)
        return np.mean(np.array(true) == np.array(pred))


class Precision(Metric):
    """Description: Metric to evaluate precision."""

    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the precision.
        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: Precision.
        """
        true_positives: Dict[int, int] = {}
        false_positives: Dict[int, int] = {}
        true = np.argmax(true, axis=1)
        for t, p in zip(true, pred):
            if t == p:
                if t not in true_positives:
                    true_positives[t] = 0
                true_positives[t] += 1
            else:
                if p not in false_positives:
                    false_positives[p] = 0
                false_positives[p] += 1

        precisions = []
        for cls in true_positives:
            tp = true_positives.get(cls, 0)
            fp = false_positives.get(cls, 0)
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))

        return sum(precisions) / len(precisions) if precisions else 0.0


class Recall(Metric):
    """Description: Metric to evaluate recall."""

    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the recall.
        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: Recall.
        """
        true_positives: Dict[int, int] = {}
        false_negatives: Dict[int, int] = {}
        true = np.argmax(true, axis=1)

        for t, p in zip(true, pred):
            if t == p:
                if t not in true_positives:
                    true_positives[t] = 0
                true_positives[t] += 1
            else:
                if t not in false_negatives:
                    false_negatives[t] = 0
                false_negatives[t] += 1

        recalls = []
        for cls in true_positives:
            tp = true_positives.get(cls, 0)
            fn = false_negatives.get(cls, 0)
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))

        return sum(recalls) / len(recalls) if recalls else 0.0


class MeanSquaredError(Metric):
    """Description: Metric to evaluate mean squared error."""

    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the mean squared error.
        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: Mean squared error.
        """
        return np.mean((np.array(true) - np.array(pred)) ** 2)


class MeanAbsoluteError(Metric):
    """Description: Metric to evaluate mean absolute error."""

    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the mean absolute error.
        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: Mean absolute error.
        """
        return np.mean(np.abs(np.array(true) - np.array(pred)))


class RootMeanSquaredError(Metric):
    """Description: Metric to evaluate root mean squared error."""

    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the root mean squared error.
        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: Root mean squared error.
        """
        return np.sqrt(np.mean((np.array(true) - np.array(pred)) ** 2))


metrics_map: Dict[str, Type] = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "mean_squared_error": MeanSquaredError,
    "mean_absolute_error": MeanAbsoluteError,
    "root_mean_squared_error": RootMeanSquaredError,
}


def get_metric(name: str) -> Metric:
    metric_class = metrics_map.get(name)
    if metric_class:
        return metric_class()
    else:
        raise ValueError(f"Unknown metric name: {name}")
