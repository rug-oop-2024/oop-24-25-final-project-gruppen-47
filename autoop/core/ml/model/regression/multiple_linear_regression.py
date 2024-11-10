import numpy as np

from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """
    Class representing a multiple linear regression model.

    Attributes:
        _parameters (dict): dictionary representing parameters
        _type (str): type of the model
    """

    def __init__(self) -> None:
        """Initialize the multiple linear regression model."""
        super().__init__()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Calculate the parameters based on a given training dataset.

        Args:
            observations: 2d array representing observations of
                a training dataset where the number of samples
                is in the row dimension and the variables
                are in the column dimension
            ground_truths: 1d array representing ground truths relating
                to observations
        Returns:
            None
        Raises:
            ValueError: if number of samples in training data do not match
            TypeError: if either arguemnt is not a required type
        """
        if observations.ndim != 2:
            raise TypeError(
                "First argument should be a 2 dimensional numpy array."
            )
        if ground_truths.ndim != 1:
            if ground_truths.ndim == 2 and ground_truths.shape[1] == 1:
                ground_truths = ground_truths.flatten()
            else:
                raise TypeError(
                    "Second argument should be a 1 dimensional numpy array."
                )
        number_of_samples = observations.shape[0]
        if number_of_samples != ground_truths.size:
            raise ValueError(
                "Number of samples in training data do not match."
            )

        row_of_ones = np.array([1] * number_of_samples)
        observations_tilde = np.column_stack([observations, row_of_ones])

        temp_params = (
            np.linalg.inv(observations_tilde.T @ observations_tilde)
            @ observations_tilde.T
            @ ground_truths
        )

        self._parameters = {"parameters": temp_params}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make a prediction for each sample in observations.

        Args:
            observations: 2d numpy array representing observations to be
                predicted where the number of samples is in the row
                dimension and the variables are in the column dimension

        Returns:
            Numpy array with predictions for each sample.
        Raises:
            ValueError: if number of samples in training data do not match
            TypeError: if either arguemnt is not a required type
        """
        self._is_fitted()
        if observations.ndim != 2:
            raise TypeError("Argument should be a 2 dimensional numpy array.")
        number_of_parameters = observations.shape[1]
        if number_of_parameters + 1 != self._parameters["parameters"].size:
            raise ValueError(
                "Number of samples in training data do not match."
            )

        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> float:
        """
        Make a prediciton for a single observation based on parameters.

        Args:
            observation: 1d numpy array representing a single
                observation

        Returns:
            A single prediction for a single observation.
        """
        observation = np.append(observation, 1)
        return np.sum(observation * self._parameters["parameters"])
