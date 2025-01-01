
# Parallel K-Means Clustering

A high-performance **Parallel K-Means Clustering** algorithm implemented in C++ with **OpenMP** for parallelization. This project demonstrates the use of advanced clustering techniques with efficient computation for large datasets.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Input File Format](#input-file-format)
- [Output Files](#output-files)
- [Performance Metrics](#performance-metrics)
- [Visualization](#visualization)
- [License](#license)

---

## Introduction

K-Means is a popular clustering algorithm used in machine learning and data analysis. This implementation leverages **parallel computing** with OpenMP to efficiently handle large datasets. The project also includes:
- **K-Means++ initialization** for better cluster center selection.
- **Silhouette Score** and **WCSS (Within-Cluster Sum of Squares)** for evaluating clustering performance.
- Python scripts for data visualization.

---

## Features

- Parallelized computation using OpenMP.
- K-Means++ for optimal initial cluster center selection.
- Normalization of data for consistent clustering.
- Evaluation metrics:
  - **WCSS** (Within-Cluster Sum of Squares)
  - **Silhouette Score**
- Python scripts for cluster visualization.
- Supports custom number of clusters and dimensions.

---

## Prerequisites

Before running this project, ensure you have the following installed:

1. **GCC** (with OpenMP support):
   - Install via Homebrew:
     ```bash
     brew install gcc
     ```
2. **Python** (for visualization scripts):
   - Required libraries: `matplotlib`, `numpy`, `sklearn`.

3. **C++ Compiler**:
   - Ensure GCC 11+ or Clang with OpenMP support is installed.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mohsinidrees2817/Parallel-K-Means-Clustering.git
   cd Parallel-K-Means-Clustering
   ```

2. Compile the C++ code:
   ```bash
   make
   ```

3. Install Python dependencies for visualization:
   ```bash
   pip install matplotlib numpy sklearn
   ```

---

## Usage

### Running the Program
Execute the compiled program with the following format:
```bash
./kmeans_parallel <input_file> <output_file> <K>
```
- `<input_file>`: Path to the input data file.
- `<output_file>`: Path to save the cluster centers.
- `<K>`: Number of clusters.

**Example:**
```bash
./kmeans_parallel data.txt clusters.txt 3
```

### Visualizing Clusters
Run the Python visualization script:
```bash
python3 plot_clusters.py
```

---

## Input File Format

The input file should be a plain text file with the following format:

1. The first line specifies:
   ```
   N D
   ```
   - `N`: Number of data points.
   - `D`: Number of dimensions (features).

2. Each subsequent line represents a data point:
   ```
   <id> <feature_1> <feature_2> ... <feature_D>
   ```

**Example Input File:**
```
5 3
product1 1.0 2.0 3.0
product2 1.5 2.5 3.5
product3 5.0 6.0 7.0
product4 8.0 9.0 10.0
product5 1.0 1.0 1.0
```

---

## Output Files

1. **Cluster Centers File (`clusters.txt`)**:
   - Contains the final cluster centers.
   - Example:
     ```
     Number of clusters: 3

     C1 1.2 2.3 3.4
     C2 5.6 6.7 7.8
     C3 9.0 10.1 11.2
     ```

2. **Cluster Assignments File (`assignments.txt`)**:
   - Maps each data point to a cluster.
   - Example:
     ```
     Cluster Assignments:

     product1 -> Cluster 1
     product2 -> Cluster 1
     product3 -> Cluster 2
     product4 -> Cluster 3
     product5 -> Cluster 1
     ```

---

## Performance Metrics

1. **Within-Cluster Sum of Squares (WCSS)**:
   - Measures cluster compactness. Lower WCSS indicates better clustering.
   - Output example:
     ```
     Within-Cluster Sum of Squares (WCSS): 22502
     ```

2. **Silhouette Score**:
   - Measures cluster separation and cohesion. Higher values indicate better-defined clusters.
   - Can be computed using the Python script `silhouette_score.py`.

---

## Visualization

### Cluster Visualization (Python Script)
- Use `plot_clusters.py` to visualize the clusters.
- The script generates a 3D scatter plot for datasets with three dimensions.

**Example Command:**
```bash
python3 plot_clusters.py
```



---

## Author

**Mohsin Idrees**  
GitHub: [mohsinidrees2817](https://github.com/mohsinidrees2817)  
Email: contact@mohsin-idrees.com
```
