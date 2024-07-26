#!/usr/bin/env python
# Created by "Thieu" at 22:27, 25/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from datarefiner import OneHotEncoder

# Sample data
data = np.array(['red', 'green', 'blue', 'green', 'red', 0, 1.5, 0., 5.1, 'blue'])
data = np.array([['cat'], ['dog'], ['cat'], ['fish'], ['dog']])
data = np.array([['red', 'small'], ['green', 'large'], ['blue', 'medium'],
                 ['green', 'small'], ['red', 'large'], ['blue', 'medium']])

# Create an instance of OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the data
one_hot_encoded = encoder.fit_transform(data)
print("One-Hot Encoded:\n", one_hot_encoded)

# Inverse transform the data
decoded_categories = encoder.inverse_transform(one_hot_encoded)
print("Decoded Categories:", decoded_categories)

print(encoder.categories_)
print(encoder.category_to_index_)
print(encoder.index_to_category_)
