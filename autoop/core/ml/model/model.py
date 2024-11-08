
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal
    
class Model(ABC):
    """
    Class represenintg a machine learning model.

    Attributes:
        _parameters: dictionary representing parameters
    """

    def __init__(self):
        self._parameters: dict = {}
        self._type: Literal["regression", "classification"] = None

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Calculate the parameters based on a training dataset.

        Args:
            observations: numpy array representing observations of
                a training dataset
            ground_truths: numpy array representing ground truths
                relating to observations

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make a prediciton for each observation based on parameters.

        Args:
            observations: numpy array representing observations
                to be predicted

        Returns:
            Numpy array with predictions for each sample.
        """
        pass

    def _is_fitted(self) -> None:
        """
        Check if a model was fitted.

        Returns:
            None
        Raises:
            ValueError: if model was not fitted
        """
        if not self._parameters:
            raise ValueError("Model not fitted. Call fit method first.")

    @property
    def parameters(self) -> dict:
        """
        Return deep copy of parameters.

        Returns:
            Deep copy of parameters
        """
        return deepcopy(self._parameters)
    
    @property
    def type(self) -> str:
        """
        Return type of model.

        Returns:
            Type of model
        """
        return self._type
    
    @type.setter
    def type(self, value: Literal["regression", "classification"]) -> None:
        """
        Set type of model.

        Args:
            value: type of model
        Returns:
            None
        """
        if value not in ["regression", "classification"]:
            raise ValueError(f"Invalid model type: {value}")
        self._type = value