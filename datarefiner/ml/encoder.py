#!/usr/bin/env python
# Created by "Thieu" at 15:17, 25/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    @staticmethod
    def check_y(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise ValueError("y label should have shape like 1-D vector.")
        return y

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : array-like
            Labels to encode.
        """
        y = self.check_y(y)
        self.unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : array-like (1-D vector)
            Labels to encode.

        Returns:
        --------
        encoded_labels : array-like
            Encoded integer labels.
        """
        y = self.check_y(y)
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.label_to_index[label] for label in y])

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters:
        -----------
        y : array-like
            Encoded integer labels.

        Returns:
        --------
        original_labels : array-like
            Original labels.
        """
        y = self.check_y(y)
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y])


class OneHotEncoder:

    @staticmethod
    def check_X(X):
        X = np.squeeze(np.asarray(X))
        if X.ndim == 1:
            return np.reshape(X, (-1, 1))
        return X

    def fit(self, X):
        X = self.check_X(X)
        # Find unique categories for each column
        self.categories_ = [np.unique(col) for col in X.T]
        # Create dictionaries to map each category to a unique index for each column
        self.category_to_index_ = [{category: index for index, category in enumerate(categories)}
                                   for categories in self.categories_]
        # Create dictionaries to map each index back to the category for each column
        self.index_to_category_ = [{index: category for index, category in enumerate(categories)}
                                   for categories in self.categories_]

    def transform(self, X):
        X = self.check_X(X)
        # Create a list of one-hot encoded matrices for each column
        one_hot_columns = []
        for i, col in enumerate(X.T):
            one_hot = np.zeros((len(col), len(self.categories_[i])), dtype=int)
            for j, category in enumerate(col):
                one_hot[j, self.category_to_index_[i][category]] = 1
            one_hot_columns.append(one_hot)
        # Concatenate the one-hot encoded columns along the second axis
        return np.hstack(one_hot_columns)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # Split the one-hot encoded data into chunks corresponding to the original columns
        indices = np.cumsum([len(categories) for categories in self.categories_])
        one_hot_columns = np.hsplit(X, indices[:-1])
        # Convert each one-hot encoded column back to the original categories
        original_columns = []
        for i, one_hot_col in enumerate(one_hot_columns):
            original_col = np.array([self.index_to_category_[i][np.argmax(row)] for row in one_hot_col])
            original_columns.append(original_col)
        # Stack the original columns together along the second axis
        return np.column_stack(original_columns)
