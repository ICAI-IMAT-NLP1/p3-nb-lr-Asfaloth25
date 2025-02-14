import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = None # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        
        length = labels.shape[0]
        
        class_priors: Dict[int, torch.Tensor] = {lab.item(): sum(labels==lab)/length for lab in labels.unique()} # could've used a collections.Counter...
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        smoothed_features = features + delta
        smoothed_features_by_label = {
            lab.item(): smoothed_features[labels == lab] for lab in labels.unique()
        }

        class_word_counts : Dict[int, torch.Tensor] = {
            lab: smoothed_features_by_label[lab]/smoothed_features_by_label[lab].sum() for lab in smoothed_features_by_label
        }

        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words

        # {label: log probability}
        Log_P_d_given_c : Dict[int, float] = {
            label: (feature * torch.log(
                # The probability of a word appearing N times is P(word|label)^N. Thus, its log prob is N * log(P(word|label)).
                # That is why I multiply feature (N) by log(word|label)
                self.conditional_probabilities[label]
                )).sum() # The logarithm of a product is equal to the sum of logarithms
            for label in self.conditional_probabilities
            }
        
        Log_P_c : Dict[int, float]= {label: torch.log(self.class_priors[label]) for label in self.class_priors}

        # Since I already have Log_P_d_given_c, I can obtain Log_P_d by performing:
        #       Log_P_d = Log(sum for all i([e^(log(P(d|class i))+log(P(class i)))]))
        # This is what the Pytorch function "logsumexp" performs in a numerically stabilized way (see docs)
        Log_P_d : float = torch.tensor([
            Log_P_d_given_c[label] + Log_P_c[label]
            for label in Log_P_c
        ]).logsumexp(dim=0).item()

        log_posteriors: torch.Tensor = torch.tensor([
            Log_P_d_given_c[label] + Log_P_c[label] - Log_P_d for label in Log_P_c
        ])

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        pred: int = torch.argmax(self.estimate_class_posteriors(feature)).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        probs: torch.Tensor = torch.softmax(self.estimate_class_posteriors(feature), dim=0)
        return probs
