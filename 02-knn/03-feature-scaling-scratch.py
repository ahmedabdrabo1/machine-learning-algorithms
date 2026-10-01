def min_max_scaling(data):
  scaled_data  = []

  for row in data:
    scaled_data.append([0] * len(row))

  for column in range(len(data[0])):

    column_values = []

    for row in data:
      column_values.append(row[column])

    min_value = min(column_values)
    max_value = max(column_values)

    for row in range(len(data)):  
       scaled_data[row][column] = ((data[row][column] - min_value) / (max_value - min_value))
  
  return scaled_data
    

x_train = [
    [20, 30000],
    [25, 50000],
    [30, 80000]
]

x_scaled = min_max_scaling(x_train)

print("Original:")
print(x_train)

print("\nScaled:")
print(x_scaled)