#!/usr/bin/env python
# Created by "Thieu" at 13:43, 24/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from datarefiner import sliding_window_transform, SlidingTransformer


# Example usage
X = np.array([[i] for i in range(1, 21)])
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

X = np.array([
        [112, 118, 132],
        [118, 130, 133],
        [132, 140, 140],
        [129, 138, 144],
        [121, 133, 136],
        [135, 150, 160],
        [148, 160, 169],
        [148, 158, 172],
        [136, 148, 160],
        [119, 133, 149],
        [104, 120, 140],
        [118, 135, 154]
    ])

lags = [1, 2]
future_steps = [0]
dropnan = True

st = SlidingTransformer(lags=lags, future_steps=future_steps, dropnan=dropnan)
print(st.transform(X))
print(st.data)

# agg = sliding_window_transform(X, lags, future_steps, dropnan)
# print(agg)
