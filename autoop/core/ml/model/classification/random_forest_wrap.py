import numpy as np

from sklearn.ensemble import RandomForestClassifier as SKlearnRandomForest

from autoop.core.ml.model.model import Model


class Lasso(Model):
    """
    Class representing a Lasso model.

    Attributes:
        _parameters: dictionary representing parameters
        lasso: SklearnLasso object representing the Lasso model
    """
    def __inti__(self):
        super.__init__()
        self._random_forest: SKlearnRandomForest  = SKlearnRandomForest()

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
        number_of_samples = observations.shape[0]
        if number_of_samples != ground_truths.size:
            raise ValueError(
                "Number of samples in training data do not match."
            )

        self._random_forest.fit(observations, ground_truths)
        self._parameters = self._random_forest.get_params()
        self._parameters["classes"] = self._random_forest.classes_
        self._parameters["estimator"] = self._random_forest.estimator_

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

        return self._random_forest.predict(observations)
