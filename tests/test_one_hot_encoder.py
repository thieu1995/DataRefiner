#!/usr/bin/env python
# Created by "Thieu" at 22:29, 25/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from datarefiner import OneHotEncoder


@pytest.fixture
def one_hot_encoder():
    data = np.array([['cat'], ['dog'], ['cat'], ['fish'], ['dog']])
    ohe = OneHotEncoder()
    ohe.fit(data)
    return ohe

def test_one_hot_encoder_transform(one_hot_encoder):
    data = np.array([['cat'], ['dog'], ['cat'], ['fish'], ['dog']])
    encoded_data = one_hot_encoder.transform(data)
    assert np.array_equal(encoded_data, np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]))

def test_one_hot_encoder_categories(one_hot_encoder):
    categories = one_hot_encoder.categories_
    assert np.array_equal(categories, [np.array(['cat', 'dog', 'fish'], dtype=object)])

if __name__ == "__main__":
    pytest.main()
