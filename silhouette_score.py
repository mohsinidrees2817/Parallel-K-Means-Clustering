# silhouette_score.py

from sklearn.metrics import silhouette_score
import numpy as np

def read_data(filename):
    data = []
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
            try:
                features = list(map(float, parts[1:n+1]))
                data.append(features)
            except ValueError as e:
                print(f"Error parsing features for a data point: {line.strip()}")
                print(e)
    return data

def read_assignments(filename):
    assignments = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '->' in line:
                parts = line.split('->')
                cluster_part = parts[1].strip()
                # Extract cluster number
                match = re.search(r'Cluster\s+(\d+)', cluster_part)
                if match:
                    cluster = int(match.group(1))
                    assignments.append(cluster)
    return assignments

if __name__ == "__main__":
    import re
    data = read_data('data.txt')
    assignments = read_assignments('assignments.txt')
    if not data:
        print("Error: No data points to calculate Silhouette Score.")
    elif len(set(assignments)) > 1:  # Silhouette score requires at least 2 clusters
        score = silhouette_score(data, assignments)
        print(f"Silhouette Score: {score}")
    else:
        print("Silhouette Score: Not defined for a single cluster.")
