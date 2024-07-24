#!/usr/bin/env python
# Created by "Thieu" at 10:29, 24/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from datarefiner import DifferenceScaler

# Example usage
if __name__ == "__main__":

    data = np.array([
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

    data = [10, 15, 20, 18, 22, 12, 8, 5]

    scaler = DifferenceScaler(order=1)

    transformed_data = scaler.transform(data)
    print("Differenced Data:", transformed_data)

    recovered_data = scaler.inverse_transform(transformed_data)
    print("Recovered Data:", recovered_data)
