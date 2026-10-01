from sklearn.preprocessing import MinMaxScaler
import numpy as np


x_train = np.array([
    [20, 30000],
    [25, 50000],
    [30, 80000]
])


scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(x_train)


print("Original:")
print(x_train)

print("\nScaled:")
print(x_scaled)