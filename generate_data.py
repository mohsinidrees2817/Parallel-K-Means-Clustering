# generate_data.py

import random

def generate_data(N, n, K, filename):
    with open(filename, 'w') as f:
        f.write(f"{N} {n}\n")
        for i in range(1, N+1):
            product = f"product{i}"
            features = [round(random.uniform(0, 100), 2) for _ in range(n)]
            f.write(f"{product} {' '.join(map(str, features))}\n")

if __name__ == "__main__":
    N = 10000  # Number of data points
    n = 3      # Number of dimensions
    K = 5      # Number of clusters
    filename = "data.txt"
    generate_data(N, n, K, filename)
