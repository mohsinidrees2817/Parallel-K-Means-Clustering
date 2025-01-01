// main.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <algorithm>
#include <random>

// Structure to hold a data point
struct Point {
    std::string id;
    std::vector<double> features;
    int cluster;
};

// Function to read data from a file
std::vector<Point> readData(const std::string& filename, int& numDims) {
    std::vector<Point> data;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open input file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }

    int N;
    infile >> N >> numDims;
    data.reserve(N);

    for (int i = 0; i < N; ++i) {
        Point p;
        infile >> p.id;
        p.features.resize(numDims);
        for (int d = 0; d < numDims; ++d) {
            infile >> p.features[d];
        }
        p.cluster = -1;
        data.push_back(p);
    }

    infile.close();
    return data;
}

// Function to normalize data to have zero mean and unit variance
void normalizeData(std::vector<Point>& data, int numDims) {
    std::vector<double> mean(numDims, 0.0);
    std::vector<double> stddev(numDims, 0.0);
    int N = data.size();

    // Calculate mean for each dimension
    for (const auto& point : data) {
        for (int d = 0; d < numDims; ++d) {
            mean[d] += point.features[d];
        }
    }
    for (int d = 0; d < numDims; ++d) {
        mean[d] /= N;
    }

    // Calculate standard deviation for each dimension
    for (const auto& point : data) {
        for (int d = 0; d < numDims; ++d) {
            stddev[d] += pow(point.features[d] - mean[d], 2);
        }
    }
    for (int d = 0; d < numDims; ++d) {
        stddev[d] = sqrt(stddev[d] / N);
        if (stddev[d] == 0) stddev[d] = 1; // Prevent division by zero
    }

    // Normalize data
    for (auto& point : data) {
        for (int d = 0; d < numDims; ++d) {
            point.features[d] = (point.features[d] - mean[d]) / stddev[d];
        }
    }
}

// Function to initialize cluster centers using K-Means++ algorithm
std::vector<Point> initializeClustersKMeansPP(const std::vector<Point>& data, int K, int numDims) {
    std::vector<Point> clusters;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    // Choose the first cluster center randomly
    int first_idx = dis(gen);
    clusters.push_back(data[first_idx]);
    clusters.back().cluster = 0;

    std::vector<double> distances(data.size(), std::numeric_limits<double>::max());

    // Choose the remaining K-1 centers
    for (int c = 1; c < K; ++c) {
        double total = 0.0;

        // Calculate the distance squared of each point to the nearest existing cluster center
        #pragma omp parallel for reduction(+:total)
        for (size_t i = 0; i < data.size(); ++i) {
            double dist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < clusters.size(); ++j) {
                double current_dist = 0.0;
                for (int d = 0; d < numDims; ++d) {
                    double diff = data[i].features[d] - clusters[j].features[d];
                    current_dist += diff * diff;
                }
                if (current_dist < dist) {
                    dist = current_dist;
                }
            }
            distances[i] = dist;
            total += dist;
        }

        if (total == 0.0) {
            // All points are identical; choose a random point as the new center
            int new_idx = dis(gen);
            clusters.push_back(data[new_idx]);
            clusters.back().cluster = c;
            continue;
        }

        // Choose a new center with probability proportional to the distance squared
        std::uniform_real_distribution<> dis_real(0, total);
        double r = dis_real(gen);
        double cumulative = 0.0;
        int new_center = -1;
        for (size_t i = 0; i < data.size(); ++i) {
            cumulative += distances[i];
            if (cumulative >= r) {
                new_center = i;
                break;
            }
        }

        if (new_center == -1) {
            new_center = data.size() - 1;
        }

        clusters.push_back(data[new_center]);
        clusters.back().cluster = c;
    }

    return clusters;
}

// Function to compute Euclidean distance between two points
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
    double dist = 0.0;
    for (size_t d = 0; d < a.size(); ++d) {
        double diff = a[d] - b[d];
        dist += diff * diff;
    }
    return sqrt(dist);
}

// Function to assign points to the nearest cluster (parallelized with OpenMP)
int assignClusters(std::vector<Point>& data, const std::vector<Point>& clusters) {
    int changes = 0;

    #pragma omp parallel for reduction(+:changes)
    for (size_t i = 0; i < data.size(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        int bestCluster = -1;
        for (size_t c = 0; c < clusters.size(); ++c) {
            double dist = euclideanDistance(data[i].features, clusters[c].features);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }
        if (data[i].cluster != bestCluster) {
            changes += 1;
            data[i].cluster = bestCluster;
        }
    }

    return changes;
}

// Function to update cluster centers (optimized with OpenMP)
void updateClustersOptimized(std::vector<Point>& data, std::vector<Point>& clusters, int numDims) {
    int K = clusters.size();
    std::vector<std::vector<double>> newFeatures(K, std::vector<double>(numDims, 0.0));
    std::vector<int> counts(K, 0);

    #pragma omp parallel
    {
        // Thread-local storage for features and counts
        std::vector<std::vector<double>> localFeatures(K, std::vector<double>(numDims, 0.0));
        std::vector<int> localCounts(K, 0);

        #pragma omp for nowait
        for (size_t i = 0; i < data.size(); ++i) {
            int cluster = data[i].cluster;
            for (int d = 0; d < numDims; ++d) {
                localFeatures[cluster][d] += data[i].features[d];
            }
            localCounts[cluster] += 1;
        }

        // Combine thread-local results into global sums
        #pragma omp critical
        {
            for (int c = 0; c < K; ++c) {
                for (int d = 0; d < numDims; ++d) {
                    newFeatures[c][d] += localFeatures[c][d];
                }
                counts[c] += localCounts[c];
            }
        }
    }

    // Update cluster centers
    for (int c = 0; c < K; ++c) {
        if (counts[c] > 0) {
            for (int d = 0; d < numDims; ++d) {
                clusters[c].features[d] = newFeatures[c][d] / counts[c];
            }
        } else {
            // Reinitialize the empty cluster to a random data point
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, data.size() - 1);
            int new_idx = dis(gen);
            clusters[c] = data[new_idx];
            clusters[c].cluster = c;
            std::cout << "Reinitialized empty cluster " << c + 1 << " to data point " << data[new_idx].id << "\n";
        }
    }
}

// Function to write cluster centers to a file
void writeClusters(const std::string& filename, const std::vector<Point>& clusters) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open output file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }

    outfile << "Number of clusters: " << clusters.size() << "\n\n";
    outfile << "# Cluster Centers:\n\n"; // Prefixed with '#' to avoid parsing as a cluster center
    for (size_t c = 0; c < clusters.size(); ++c) {
        outfile << "C" << c + 1 << " ";
        for (size_t d = 0; d < clusters[c].features.size(); ++d) {
            outfile << clusters[c].features[d] << " ";
        }
        outfile << "\n\n";
    }

    outfile.close();
}

// Function to write cluster assignments to a file
void writeClusterAssignments(const std::string& filename, const std::vector<Point>& data) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open cluster assignments file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }

    outfile << "Cluster Assignments:\n\n";
    for (const auto& point : data) {
        outfile << point.id << " -> Cluster " << point.cluster + 1 << "\n";
    }

    outfile.close();
}

// Function to calculate Within-Cluster Sum of Squares (WCSS)
double calculateWCSS(const std::vector<Point>& data, const std::vector<Point>& clusters) {
    double wcss = 0.0;
    #pragma omp parallel for reduction(+:wcss)
    for (size_t i = 0; i < data.size(); ++i) {
        double dist = euclideanDistance(data[i].features, clusters[data[i].cluster].features);
        wcss += dist * dist;
    }
    return wcss;
}

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc != 4) {
        std::cerr << "Usage: ./kmeans_parallel <input_file> <output_file> <K>\n";
        return EXIT_FAILURE;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    int K = std::atoi(argv[3]);

    // Read data
    int numDims;
    std::vector<Point> data = readData(inputFile, numDims);

    // Normalize data
    normalizeData(data, numDims);

    // Initialize clusters using K-Means++ algorithm
    std::vector<Point> clusters = initializeClustersKMeansPP(data, K, numDims);

    // K-Means iterations
    int maxIterations = 100;
    int iter = 0;
    int changes = 0;

    do {
        changes = assignClusters(data, clusters);
        updateClustersOptimized(data, clusters, numDims);
        iter += 1;
        std::cout << "Iteration " << iter << ": " << changes << " changes\n";
    } while (changes > 0 && iter < maxIterations);

    // Write cluster centers to output file
    writeClusters(outputFile, clusters);

    // Write cluster assignments to assignments.txt
    writeClusterAssignments("assignments.txt", data);

    // Calculate and print WCSS
    double wcss = calculateWCSS(data, clusters);
    std::cout << "Clustering completed in " << iter << " iterations.\n";
    std::cout << "Within-Cluster Sum of Squares (WCSS): " << wcss << "\n";

    return 0;
}
