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
    "root_mean_squared_error"
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
    # your code here
    # remember: metrics take ground truth and prediction as input and return a real number
    def __call__(self):
        return self.evaluate()
# add here concrete implementations of the Metric class
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
        true_positives = sum([1 for t, p in zip(true, pred) if t == 1 and p == 1])
        false_positives = sum([1 for t, p in zip(true, pred) if t == 0 and p == 1])
        return true_positives / (true_positives + false_positives)
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
        true_positives = sum([1 for t, p in zip(true, pred) if t == 1 and p == 1])
        false_negatives = sum([1 for t, p in zip(true, pred) if t == 1 and p == 0])
        return true_positives / (true_positives + false_negatives)
class F1Score(Metric):
    """Description: Metric to evaluate F1 score."""
    def evaluate(self, true: List, pred: List) -> float:
        """
        Evaluate the F1 score.
        Args:
            true: Ground truth values.
            pred: Predicted values.
        Returns:
            float: F1 score.
        """
        precision = Precision().evaluate(true, pred)
        recall = Recall().evaluate(true, pred)
        return 2 * (precision * recall) / (precision + recall)
    
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
    "f1_score": F1Score,
    "mean_squared_error": MeanSquaredError,
    "mean_absolute_error": MeanAbsoluteError,
    "root_mean_squared_error": RootMeanSquaredError
}
def get_metric(name: str):
    metric_class = metrics_map.get(name)
    if metric_class:
        return metric_class()
    else:
        raise ValueError(f"Unknown metric name: {name}")