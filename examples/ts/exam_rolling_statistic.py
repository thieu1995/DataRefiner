#!/usr/bin/env python
# Created by "Thieu" at 18:48, 24/07/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Example Usage
import pandas as pd
from datarefiner import RollingStatisticTransformer


if __name__ == "__main__":
    # Example time series data with multiple columns
    data = {'date': pd.date_range(start='2023-01-01', periods=15, freq='D'),
            'value1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'value2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]}
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)

    # Compute rolling statistics with custom lags and future steps
    lags = [1, 3, 5]
    future_steps = [0, 1,]
    transformer = RollingStatisticTransformer(lags=lags, future_steps=future_steps, statistics=['mean', 'std'])
    features, labels = transformer.transform(df)

    print("Features:\n", features)
    print("\nLabels:\n", labels)
