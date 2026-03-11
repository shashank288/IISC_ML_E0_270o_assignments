import numpy as np

class GaussianNaiveBayes:

    """
    A Gaussian Naive Bayes classifier.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.priors_ = None        # shape: (n_classes,)
        self.means_ = None         # shape: (n_classes, n_features)
        self.vars_ = None          # shape: (n_classes, n_features)
        self.classes_ = None


    def fit(self, X: np.ndarray, y: np.ndarray, update: bool = False) -> None:
        """
        Fit the model on the training data
        :param X: np.ndarray
            The training data
        :param y: np.ndarray
            The training labels
        :param update: bool
            Whether to the model is being updated with new data
            or trained from scratch
        :return: None
        """
        raise NotImplementedError

    def _gaussian_log_likelihood(self, x, mean, var):
        """
        Log of Gaussian probability density.
        """
        return NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data
        :param X: np.ndarray
            The input data
        :return: np.ndarray
            The predicted labels
        """
        preds = []
        for i in range(X.shape[0]):
            log_probs = []
            raise NotImplementedError
        return np.array(preds)