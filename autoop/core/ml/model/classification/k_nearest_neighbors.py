from collections import Counter
from typing import Any

import numpy as np

from autoop.core.ml.model.model import Model


class KNearestNeighbors(Model):
    """
    Class representing a K-Nearest Neighbors model.

    Attributes:
        k: int representing the number of neighbors
            to consider when making a prediction
        _parameters: dictionary representing parameters
    """
    def __init__(self, k:int = 3):
        super().__init__()
        self.k = k  # noqa <VNE001>
        self._type = "classification"

    @property
    def k(self) -> int:
        """
        Return the value of k.

        Returns:
            int representing the number of neighbors to consider
        """
        return self._k

    @k.setter
    def k(self, value: int) -> None:  # noqa <N805>
        """
        Validate that k is greater than 0. and set the value of k.

        Args:
            value: int representing the number of neighbors to consider

        Returns:
            None
        Raises:
            ValueError: if k is not greater than 0
        """
        if value <= 0:
            raise ValueError("k must be greater than 0")
        self._k = value

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Fit the model to the training data.

        Args:
            observations: 2d numpy array representing observations of
                a training dataset
            ground_truths: 1d numpy array representing ground truths
                relating to observations

        Returns:
            None
        Raises:
            ValueError: if number of samples in training data do not match.
        """
        ground_truths = np.argmax(ground_truths, axis=1)
        print(ground_truths)
        number_of_samples = observations.shape[0]
        if number_of_samples != ground_truths.size:
            raise ValueError(
                "Number of samples in training data do not match."
            )

        self._parameters = {
            "k" : self._k,
            "observations": observations,
            "ground_truths": ground_truths,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make a prediction for each observation based on parameters.

        Args:
            observations: 2d numpy array representing observations
                to be predicted

        Returns:
            Numpy array with predictions for each sample.
        """
        self._is_fitted()

        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> Any:
        """
        Make a prediction for a single observation.

        Args:
            observation: 1d numpy array representing a single observation

        Returns:
            Prediction for the observation.
        """
        distances = np.linalg.norm(
            self._parameters["observations"] - observation, axis=1
        )
        sorted_indices = np.argsort(distances)
        k_indices = sorted_indices[: self.k]
        k_nearest_labels = [
            self._parameters["ground_truths"][i] for i in k_indices
        ]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
