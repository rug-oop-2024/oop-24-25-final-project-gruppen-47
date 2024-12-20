import numpy as np

from sklearn.naive_bayes import GaussianNB as SKLearnGaussianNB

from autoop.core.ml.model.model import Model


class NaiveBayesModel(Model):
    """
    Class representing a Lasso model.

    Attributes:
        _parameters (dict): dictionary representing parameters
        _naive_bayes_gaussian (SKLearnGaussianNB): SKLearnGaussianNB object
            representing the Naive Bayes model
        _type (str): string representing the type of model
    """

    def __init__(self) -> None:
        """Initialize the Naive Bayes model."""
        super().__init__()
        self._naive_bayes_gaussian = SKLearnGaussianNB()
        self._type = "classification"

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
        ground_truths = np.argmax(ground_truths, axis=1)
        number_of_samples = observations.shape[0]
        if number_of_samples != ground_truths.size:
            raise ValueError("Number of samples in training data do not match")

        self._naive_bayes_gaussian.fit(observations, ground_truths)
        self._parameters = self._naive_bayes_gaussian.get_params()
        self._parameters["classes"] = self._naive_bayes_gaussian.classes_
        self._parameters["epsilon"] = self._naive_bayes_gaussian.epsilon_

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

        return self._naive_bayes_gaussian.predict(observations)
