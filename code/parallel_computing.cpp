#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <omp.h>
#include <fstream>
using namespace std;

// Number of cities
#define N 48

// Define the maximum and minimum frequency ranges
#define fmin 0
#define fmax 1

// Define maximum velocity
#define vmax 0.2

// Define loudness and pulse rate
#define A0 0.5
#define r0 0.5

// Define number of bats
#define NBATS 10

// Define number of iterations
#define ITERATIONS 100

// City structure to store x and y coordinates of a city
struct City {
    double x, y;
};

// Function to calculate Euclidean distance between two cities
double distance(City city1, City city2) {
    return sqrt((city1.x - city2.x) * (city1.x - city2.x) + (city1.y - city2.y) * (city1.y - city2.y));
}

// Function to generate a random solution (random permutation of cities)
vector<int> generateRandomSolution() {
    vector<int> solution;
    for (int i = 0; i < N; ++i) {
        solution.push_back(i);
    }
    random_shuffle(solution.begin() + 1, solution.end());
    return solution;
}

// Function to evaluate the total distance of a solution
double evaluateSolution(vector<int> solution, vector<vector<double>> cities) {
    double totalDistance = 0;
    for (int i = 0; i < N - 1; ++i) {
        totalDistance += distance({cities[solution[i]][0], cities[solution[i]][1]}, {cities[solution[i + 1]][0], cities[solution[i + 1]][1]});
    }
    // Add distance from last city to the starting city
    totalDistance += distance({cities[solution[N - 1]][0], cities[solution[N - 1]][1]}, {cities[solution[0]][0], cities[solution[0]][1]});
    return totalDistance;
}

// Function to initialize bats with random solutions
vector<vector<int>> initializeBats() {
    vector<vector<int>> bats;
    for (int i = 0; i < NBATS; ++i) {
        bats.push_back(generateRandomSolution());
    }
    return bats;
}

// Function to apply local search to improve a solution
vector<int> localSearch(vector<int> solution, vector<vector<double>> cities) {
    vector<int> bestSolution = solution;
    double bestDistance = evaluateSolution(solution, cities);

    for (int i = 1; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            // Swap cities i and j
            swap(solution[i], solution[j]);
            double newDistance = evaluateSolution(solution, cities);
            if (newDistance < bestDistance) {
                bestDistance = newDistance;
                bestSolution = solution;
            }
            // Swap back to original solution
            swap(solution[i], solution[j]);
        }
    }

    // Remove repeated cities
    vector<bool> visited(N, false);
    vector<int> newSolution;
    for (int i = 0; i < N; ++i) {
        if (!visited[bestSolution[i]]) {
            newSolution.push_back(bestSolution[i]);
            visited[bestSolution[i]] = true;
        }
    }

    return newSolution;
}

// Function to perform Bat Algorithm to solve TSP
vector<int> batAlgorithm(vector<vector<int>> bats, vector<vector<double>> cities, int numThreads) {
    vector<int> globalBestSolution;
    double globalBestDistance = numeric_limits<double>::max();

    vector<double> frequencies(NBATS, fmin);
    vector<double> velocities(NBATS, 0.0);

    omp_set_num_threads(numThreads);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        #pragma omp parallel for
        for (int i = 0; i < NBATS; ++i) {
            vector<int> newSolution = bats[i];
            // Generate a new solution by combining with global best solution
            newSolution = localSearch(newSolution, cities);
            // Evaluate new solution
            double newDistance = evaluateSolution(newSolution, cities);

            // Update bat's solution
            bats[i] = newSolution;

            // Update global best solution
            if (newDistance < globalBestDistance) {
                #pragma omp critical
                {
                    if (newDistance < globalBestDistance) {
                        globalBestDistance = newDistance;
                        globalBestSolution = newSolution;
                    }
                }
            }

            // Update frequency
            frequencies[i] = fmin + (fmax - fmin) * ((double)rand() / RAND_MAX);

            // Update velocity
            velocities[i] += (newDistance - globalBestDistance) * frequencies[i];

            // Apply velocity clamping
            if (velocities[i] > vmax) velocities[i] = vmax;
            else if (velocities[i] < -vmax) velocities[i] = -vmax;

            // Move towards the global best solution
            for (int j = 0; j < N; ++j) {
                if ((double)rand() / RAND_MAX < r0) {
                    double randFactor = (double)rand() / RAND_MAX;
                    velocities[i] *= randFactor;
                    // Check if the city at index j is already present in newSolution
                    bool cityAlreadyPresent = false;
                    for (int k = 0; k < newSolution.size(); ++k) {
                        if (newSolution[k] == globalBestSolution[j]) {
                            cityAlreadyPresent = true;
                            break;
                        }
                    }
                    // If the city is not already present, add it to newSolution
                    if (!cityAlreadyPresent) {
                        newSolution[j] = globalBestSolution[j];
                    }
                }
            }
        }
    }
    return globalBestSolution;
}

int main() {
    // Initialize random seed
    srand(time(0));

    // Define cities
    ifstream cityCoordinate ("../dataset/dataset1/att48_xy.txt");
    vector<vector<double>> cities(N, vector<double>(2));
    double value;
    int i = 0;
    while (cityCoordinate >> value) {
        cities[i][0] = value;
        cityCoordinate >> value;
        cities[i][1] = value;
        i++;
    }

    // Close the input file
    cityCoordinate.close();

    

    // Initialize bats
    vector<vector<int>> bats = initializeBats();

    int maxThreads = omp_get_max_threads();
    vector<int> best_solution;
    for(int cur = 1; cur<=maxThreads; cur++){
        // cur is the current number of threads
        double start, end;
        start = omp_get_wtime();
        vector<int> solution = batAlgorithm(bats, cities, cur);
        best_solution = solution;
        // Solve TSP using Bat Algorithm
        end = omp_get_wtime();
        cout << "Parallel approach with " << cur << " threads: " << (end - start) << " seconds" << endl;

        // Print the solution
    }

    cout << "Best solution found by Bat Algorithm: ";
    for (int i = 0; i < N; ++i) {
        cout << best_solution[i] << " ";
    }
    cout << endl;

    // Print the total distance of the best solution
    cout << "Total distance of the best solution: " << evaluateSolution(best_solution, cities) << endl;

    return 0;
}
