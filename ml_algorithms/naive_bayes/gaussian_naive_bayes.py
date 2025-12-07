import numpy as np

class GaussianNaivBayes:
    """
    A gaussian naive bayes classifier implementation from scratch
    
    This classifier assumes that features follow a Gaussian (normal) distribution
    and uses Bayes theorm with 'naive' assumption that all features are independent
    of each other given the class label.
    """
    def __init__(self):
        """
        Initialize the classifier
        
        Attributes:
            classes: array-like
                unique class labels in the training data
            class_priors: dict
                Prior praobability P(y) for each class
            parameters: dict
                Dictionary containing mean and variance for each feature per class
                Structure: {class_label: {'mean': array, 'var': array}}
        """
        self.classes = None
        self.class_priors = {}
        self.paramters = {}
        
    def fit(self, X, y):
        """
        Train the naive bayes classifier by calculating class priors and feature
        feature parameters (mean and variance) for each class.
        
        The method implements:
        1. Calculate prior probability P(y) for each class
        2. Calculate mean and variance for each feature per class
        
        Mathematical foundation:
            Prior probability: P(y_k) = count(y_k) / total_samples
            Mean: m_k = (1 / n_k) * ∑(x_i) for all x_i in class k
            Variance: var_k = (1 / n_k) * ∑(x_i - m_k)**2 for all x_i in class k
        """
        
        X, y = np.array(X), np.array(y)
        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate parameters for each class
        for class_label in self.classes:
            # Extract samples belonging to this class
            X_class = X[y==class_label]
            
            # Calculate prior probability P(y = class_label)
            # This is the proportion of samples belonging to this class
            self.class_priors[class_label] = len(X_class) / n_samples
            
            # Calculate mean and variance for each feature in this class
            self.paramters[class_label] = {
                'mean': X_class.mean(axis=0),
                'var': X_class.var(axis=0) + 1e-9 # Add small constant to avoid divison by zero
            }
            
    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate the gaussian probability density function (PDF)
        
        This method computes the likelihood P(x|y) using the gaussian distribution formula.
        It represents the probability of oberving feature value x given a particular class 
        
        Mathematical formula:
            P(x|y) = (1 / √(2π∂**2)) * exp(-(x - µ)**2 / (2∂**2))
            
        This formula calculates how likely a feature value is to occur given the mean and
        variance of a Gaussinan distribution. (∂**2 means variance)
        """
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp( -(x - mean)**2 / (2 * var))
        return coeff * exponent
    
    def _calculate_posterior(self, x):
        """
        Calculate posterior probabilities P(y|X) for all classes.
        
        This method applies bayes theorm to calculate the probabilty of each class given the
        input features. It combines prior probabilities with feature likelihoods.
        
        Mathematical foundation:
            Bayes Theorm: P(y|X) = P(X|y) * P(y) / P(X)
            
        since we are comparing classes, we can ignore P(X) as it's constant across all the classes.
        The formula becomes:
            P(y|X) ≈ P(y) * product of [P(x_i|y)] for all features x_i
            
        The 'naive' assumption: features are independent given the class, allowing us to multiply
        individual feature probabilities.
        """
        posteriors = {}
        
        for class_label in self.classes:
            # Start with prior probability P(y)
            prior = np.log(self.class_priors[class_label])
            
            mean = self.paramters[class_label]['mean']
            var = self.paramters[class_label]['var']
            
            # Calculate likelihood for all features
            # we use log probabilities to avoid numerical underflow
            likelihood = self._calculate_likelihood(x, mean, var)
            
            # Sum log likelihoods (equivalent to multiplying probabilities)
            log_likelihood = np.sum(np.log(likelihood + 1e-9)) # Add small value to avoid log(0)
            
            # Calculate posterior: log P(y|X) = log P(X|y) + ∑ log P(x_i|y)
            posteriors[class_label] = prior + log_likelihood
        return posteriors
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Decision rule:
            y_pred = argmax P(y|X) over all classes
        
            This is called Maximum A Posteriori (MAP) estimation, where we choose the class that has
            highest posterior probability given the features.
        """
        X = np.array(X)
        predictions = []
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        This method returns the probability distribution over classes for each sample, rather than
        just the most likely class
        
        Process:
        1. Calculate log posteriors for numerical stability
        2. Convert log probabilities back to regular probabilities
        3. Normalize so probabilities sums to 1.0
        
        Note: we use the log-sum-exp trick to avoid numerical overflow/underflow when dealing with
        very small probability values.
        """
        X = np.array(X)
        
        probabilities = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Convert to array for easier manipulation
            class_labels = list(posteriors.keys())
            log_probs = np.array([posteriors[c] for c in class_labels])
            
            # Apply softmax to convert log probabilities to probabilities
            # This normalizes so they su to 1.0
            # softmax(x) = exp(x) / ∑ exp(x)
            # Using log-sum-exp trick for numerical stability
            log_probs_shifted = log_probs - np.max(log_probs)
            probs = np.exp(log_probs_shifted)
            probs = probs / sum (probs)
            
            probabilities.append(probs)
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate the accuracy of the classifier on test data.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions==y)
        return accuracy
    
import numpy as np

class GaussianNaiveBayes:
    """
    A Gaussian Naive Bayes classifier implementation from scratch.
    
    This classifier assumes that features follow a Gaussian (normal) distribution
    and uses Bayes' theorem with the 'naive' assumption that all features are
    independent of each other given the class label.
    """
    
    def __init__(self):
        """
        Initialize the classifier.
        
        Attributes:
        -----------
        classes : array-like
            Unique class labels in the training data
        class_priors : dict
            Prior probabilities P(y) for each class
        parameters : dict
            Dictionary containing mean and variance for each feature per class
            Structure: {class_label: {'mean': array, 'var': array}}
        """
        self.classes = None
        self.class_priors = {}
        self.parameters = {}
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier by calculating class priors and 
        feature parameters (mean and variance) for each class.
        
        The method implements:
        1. Calculate prior probability P(y) for each class
        2. Calculate mean (μ) and variance (σ²) for each feature per class
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data where n_samples is the number of samples and 
            n_features is the number of features
        y : array-like, shape (n_samples,)
            Target values (class labels)
        
        Mathematical Foundation:
        ------------------------
        Prior Probability: P(y_k) = count(y_k) / total_samples
        Mean: μ_k = (1/n_k) * Σ(x_i) for all x_i in class k
        Variance: σ²_k = (1/n_k) * Σ(x_i - μ_k)² for all x_i in class k
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate parameters for each class
        for class_label in self.classes:
            # Extract samples belonging to this class
            X_class = X[y == class_label]
            
            # Calculate prior probability P(y = class_label)
            # This is the proportion of samples belonging to this class
            self.class_priors[class_label] = len(X_class) / n_samples
            
            # Calculate mean and variance for each feature in this class
            # Mean: average value of each feature for this class
            # Variance: measure of spread/dispersion of feature values
            self.parameters[class_label] = {
                'mean': X_class.mean(axis=0),
                'var': X_class.var(axis=0) + 1e-9  # Add small constant to avoid division by zero
            }
    
    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate the Gaussian probability density function (PDF).
        
        This method computes the likelihood P(x|y) using the Gaussian distribution
        formula. It represents the probability of observing feature value x given
        a particular class.
        
        Parameters:
        -----------
        x : float or array-like
            Feature value(s) to calculate likelihood for
        mean : float or array-like
            Mean of the Gaussian distribution
        var : float or array-like
            Variance of the Gaussian distribution
        
        Returns:
        --------
        likelihood : float or array-like
            Probability density value(s)
        
        Mathematical Formula:
        ---------------------
        P(x|y) = (1 / √(2πσ²)) * exp(-(x - μ)² / (2σ²))
        
        Where:
        - x is the feature value
        - μ (mu) is the mean
        - σ² (sigma squared) is the variance
        - π (pi) is approximately 3.14159
        - exp is the exponential function
        
        The formula calculates how likely a feature value is to occur
        given the mean and variance of a Gaussian distribution.
        """
        # Calculate the coefficient: 1 / sqrt(2 * pi * variance)
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        
        # Calculate the exponent: -(x - mean)² / (2 * variance)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        
        # Return the Gaussian probability density
        return coeff * exponent
    
    def _calculate_posterior(self, x):
        """
        Calculate posterior probabilities P(y|X) for all classes.
        
        This method applies Bayes' theorem to calculate the probability
        of each class given the input features. It combines prior probabilities
        with feature likelihoods.
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            A single sample with feature values
        
        Returns:
        --------
        posteriors : dict
            Posterior probability for each class
            Structure: {class_label: probability}
        
        Mathematical Foundation:
        ------------------------
        Bayes' Theorem: P(y|X) = P(X|y) * P(y) / P(X)
        
        Since we're comparing classes, we can ignore P(X) (the evidence) as it's
        constant across all classes. We calculate:
        
        P(y|X) ∝ P(y) * Π P(x_i|y) for all features x_i
        
        Where:
        - P(y) is the prior probability (class frequency)
        - Π represents the product (multiplication)
        - P(x_i|y) is the likelihood of feature x_i given class y
        
        The 'naive' assumption: features are independent given the class,
        allowing us to multiply individual feature probabilities.
        """
        posteriors = {}
        
        for class_label in self.classes:
            # Start with the prior probability P(y)
            prior = np.log(self.class_priors[class_label])
            
            # Get parameters for this class
            mean = self.parameters[class_label]['mean']
            var = self.parameters[class_label]['var']
            
            # Calculate likelihood for all features: Π P(x_i|y)
            # We use log probabilities to avoid numerical underflow
            # log(a * b) = log(a) + log(b)
            likelihood = self._calculate_likelihood(x, mean, var)
            
            # Sum log likelihoods (equivalent to multiplying probabilities)
            log_likelihood = np.sum(np.log(likelihood + 1e-10))  # Add small value to avoid log(0)
            
            # Calculate posterior: log P(y|X) = log P(y) + Σ log P(x_i|y)
            posteriors[class_label] = prior + log_likelihood
        
        return posteriors
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        For each sample, this method calculates posterior probabilities for all
        classes and selects the class with the highest probability (MAP estimation:
        Maximum A Posteriori).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels
        
        Decision Rule:
        --------------
        y_pred = argmax P(y|X) over all classes y
        
        This is called Maximum A Posteriori (MAP) estimation, where we choose
        the class that has the highest posterior probability given the features.
        """
        X = np.array(X)
        predictions = []
        
        # Predict for each sample
        for x in X:
            # Calculate posterior probabilities for all classes
            posteriors = self._calculate_posterior(x)
            
            # Select class with highest posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        This method returns the probability distribution over classes for each
        sample, rather than just the most likely class.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict probabilities for
        
        Returns:
        --------
        probabilities : array-like, shape (n_samples, n_classes)
            Probability of each class for each sample
        
        Process:
        --------
        1. Calculate log posteriors for numerical stability
        2. Convert log probabilities back to regular probabilities
        3. Normalize so probabilities sum to 1.0
        
        Note: We use the log-sum-exp trick to avoid numerical overflow/underflow
        when dealing with very small probability values.
        """
        X = np.array(X)
        probabilities = []
        
        for x in X:
            # Get log posteriors
            posteriors = self._calculate_posterior(x)
            
            # Convert to array for easier manipulation
            class_labels = list(posteriors.keys())
            log_probs = np.array([posteriors[c] for c in class_labels])
            
            # Apply softmax to convert log probabilities to probabilities
            # This normalizes so they sum to 1
            # softmax(x) = exp(x) / Σ exp(x)
            # Using log-sum-exp trick for numerical stability
            log_probs_shifted = log_probs - np.max(log_probs)
            probs = np.exp(log_probs_shifted)
            probs = probs / np.sum(probs)
            
            probabilities.append(probs)
        
        return np.array(probabilities)
    
    def score(self, X, y):
        """
        Calculate the accuracy of the classifier on test data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels for X
        
        Returns:
        --------
        accuracy : float
            Proportion of correct predictions (between 0 and 1)
        
        Formula:
        --------
        Accuracy = (Number of correct predictions) / (Total predictions)
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


# Example Usage and Testing
if __name__ == "__main__":
    # Create sample dataset (Iris-like)
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=2000, 
        n_features=4, 
        n_informative=3, 
        n_redundant=1, 
        n_classes=3, 
        random_state=42
    )
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize and train the classifier
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    
    # Make predictions
    predictions = nb.predict(X_test)
    
    # Get probability predictions
    probabilities = nb.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = nb.score(X_test, y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nFirst 5 predictions: {predictions[:5]}")
    print(f"First 5 true labels: {y_test[:5]}")
    print(f"\nProbabilities for first sample:\n{probabilities[0]}")
