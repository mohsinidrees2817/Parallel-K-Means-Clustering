# plot_clusters.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

def read_clusters(filename):
    centers = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # Use regex to match lines starting with 'C' followed by digits
            if re.match(r'^C\d+', line):
                parts = line.split()
                # Convert feature values to float
                try:
                    center = [float(coord) for coord in parts[1:]]
                    centers.append(center)
                except ValueError as e:
                    print(f"Error parsing line: {line}")
                    print(e)
    return centers

def read_assignments(filename):
    assignments = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '->' in line:
                parts = line.split('->')
                product = parts[0].strip()
                cluster_part = parts[1].strip()
                # Extract cluster number
                match = re.search(r'Cluster\s+(\d+)', cluster_part)
                if match:
                    cluster = int(match.group(1))
                    assignments[product] = cluster
    return assignments

def read_data(filename):
    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        if not lines:
            print("Error: Data file is empty.")
            return data
        try:
            N, n = map(int, lines[0].strip().split())
        except ValueError as e:
            print("Error parsing the first line of data file.")
            print(e)
            return data
        for line in lines[1:N+1]:
            parts = line.strip().split()
            if len(parts) < n + 1:
                print(f"Error: Incomplete data for line: {line.strip()}")
                continue
            product = parts[0]
            try:
                features = list(map(float, parts[1:n+1]))
                data[product] = features
            except ValueError as e:
                print(f"Error parsing features for {product}: {line.strip()}")
                print(e)
    return data

def plot_clusters(data_file, assignments_file, clusters_file):
    data = read_data(data_file)
    assignments = read_assignments(assignments_file)
    centers = read_clusters(clusters_file)

    if not centers:
        print("Error: No cluster centers found.")
        return
    if not data:
        print("Error: No data points found.")
        return

    # Determine number of dimensions
    n_dims = len(next(iter(data.values())))
    if n_dims == 2:
        plt.figure(figsize=(10, 8))
        # Assign unique colors for each cluster
        colors = {}
        for product, features in data.items():
            cluster = assignments.get(product, 0)
            colors[product] = f'C{cluster - 1}'  # Adjust color index
            plt.scatter(features[0], features[1], c=colors[product], marker='o', label=f'Cluster {cluster}' if product == list(data.keys())[0] else "")
        # Plot cluster centers
        for idx, center in enumerate(centers):
            plt.scatter(center[0], center[1], c='black', marker='X', s=200, label=f'Cluster {idx + 1} Center' if idx == 0 else "")
        plt.title('K-Means Clustering Results (2D)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif n_dims == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        # Assign unique colors for each cluster
        colors = {}
        for product, features in data.items():
            cluster = assignments.get(product, 0)
            colors[product] = f'C{cluster - 1}'  # Adjust color index
            ax.scatter(features[0], features[1], features[2], c=colors[product], marker='o')
        # Plot cluster centers
        for idx, center in enumerate(centers):
            ax.scatter(center[0], center[1], center[2], c='black', marker='X', s=200, label=f'Cluster {idx + 1} Center' if idx == 0 else "")
        ax.set_title('K-Means Clustering Results (3D)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.legend()
        plt.show()
    else:
        print("Visualization is only supported for 2D or 3D data.")

if __name__ == "__main__":
    data_file = 'data.txt'
    assignments_file = 'assignments.txt'
    clusters_file = 'clusters.txt'
    plot_clusters(data_file, assignments_file, clusters_file)
