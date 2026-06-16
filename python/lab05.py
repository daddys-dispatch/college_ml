import numpy as np
from collections import Counter

data = np.random.rand(100)
train_data, test_data = data[:50], data[50:]
train_labels = ["Class 1" if x <= 0.5 else "Class 2" for x in train_data]


def knn(train_data, train_labels, point, k):
    i = np.argsort(np.abs(train_data - point))[:k]
    return Counter(np.array(train_labels)[i]).most_common(1)[0][0]


k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    print(f"\nk = {k}")
    for i, x in enumerate(test_data, start=51):
        print(f"x{i} ({x:.4f}) → {knn(train_data, train_labels, x, k)}")
