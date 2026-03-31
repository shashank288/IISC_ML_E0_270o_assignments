import numpy as np

class GaussianNaiveBayes:

    """
    A Gaussian Naive Bayes classifier.
    """

    def __init__(self, var_smoothing: float = 1e-9, uniform_prior: bool = False):
        self.var_smoothing = var_smoothing
        self.uniform_prior = uniform_prior
        self.priors_ = None        # shape: (n_classes,)
        self.means_ = None         # shape: (n_classes, n_features)
        self.vars_ = None          # shape: (n_classes, n_features)
        self.classes_ = None
        self.n_samples_ = None     # shape: (n_classes,) — per-class counts for incremental update


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
        if not update or self.classes_ is None:
            # Train from scratch
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            n_features = X.shape[1]

            self.means_ = np.zeros((n_classes, n_features))
            self.vars_ = np.zeros((n_classes, n_features))
            self.n_samples_ = np.zeros(n_classes)

            for i, c in enumerate(self.classes_):
                X_c = X[y == c]
                self.n_samples_[i] = X_c.shape[0]
                self.means_[i] = X_c.mean(axis=0)
                self.vars_[i] = X_c.var(axis=0)
        else:
            # Incremental update using sufficient statistics
            for i, c in enumerate(self.classes_):
                X_c = X[y == c]
                if X_c.shape[0] == 0:
                    continue

                n_new = X_c.shape[0]
                mean_new = X_c.mean(axis=0)
                var_new = X_c.var(axis=0)

                n_old = self.n_samples_[i]
                mean_old = self.means_[i]
                var_old = self.vars_[i]

                n_total = n_old + n_new
                # Combined mean
                mean_combined = (n_old * mean_old + n_new * mean_new) / n_total
                # Combined variance using parallel algorithm
                var_combined = (
                    n_old * (var_old + mean_old ** 2) +
                    n_new * (var_new + mean_new ** 2)
                ) / n_total - mean_combined ** 2

                self.n_samples_[i] = n_total
                self.means_[i] = mean_combined
                self.vars_[i] = var_combined

        # Set priors
        if self.uniform_prior:
            self.priors_ = np.ones(len(self.classes_)) / len(self.classes_)
        else:
            self.priors_ = self.n_samples_ / self.n_samples_.sum()

    def _gaussian_log_likelihood(self, x, mean, var):
        """
        Log of Gaussian probability density.
        Computes log N(x | mean, var) for each feature and sums.
        """
        smoothed_var = var + self.var_smoothing
        log_prob = -0.5 * np.sum(
            np.log(2 * np.pi * smoothed_var) + (x - mean) ** 2 / smoothed_var
        )
        return log_prob

    def _compute_log_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior for each sample and each class.
        Returns shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            log_prior = np.log(self.priors_[i])
            smoothed_var = self.vars_[i] + self.var_smoothing
            # Vectorized log-likelihood for all samples at once
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * smoothed_var) +
                (X - self.means_[i]) ** 2 / smoothed_var,
                axis=1
            )
            log_posteriors[:, i] = log_prior + log_likelihood

        return log_posteriors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the input data
        :param X: np.ndarray
            The input data
        :return: np.ndarray
            The predicted labels
        """
        log_posteriors = self._compute_log_posteriors(X)
        return self.classes_[np.argmax(log_posteriors, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities for each class.
        Uses log-sum-exp trick for numerical stability.
        :return: np.ndarray of shape (n_samples, n_classes)
        """
        log_posteriors = self._compute_log_posteriors(X)
        # Log-sum-exp trick
        log_max = np.max(log_posteriors, axis=1, keepdims=True)
        log_sum = log_max + np.log(
            np.sum(np.exp(log_posteriors - log_max), axis=1, keepdims=True)
        )
        log_proba = log_posteriors - log_sum
        return np.exp(log_proba)