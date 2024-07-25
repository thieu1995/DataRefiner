#!/usr/bin/env python
# Created by "Thieu" at 15:43, 25/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from datarefiner import LabelEncoder


# Sample data
categories = np.array(['red', 'green', 'blue', 'green', 'red', 'blue'])
categories = ['red', 'green', 'blue', 'green', 'red', 'blue', 0, 8.5]

# Create an instance of OneHotEncoder
encoder = LabelEncoder()

# Fit and transform the data
one_hot_encoded = encoder.fit_transform(categories)
print("Label Encoded:\n", one_hot_encoded)

# Inverse transform the data
decoded_categories = encoder.inverse_transform(one_hot_encoded)
print("Decoded Categories:", decoded_categories)

print(encoder.unique_labels)
print(encoder.label_to_index)

