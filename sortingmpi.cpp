#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace std::chrono;

void swap(int& a, int& b) {
    int t = a;
    a = b;
    b = t;
}

int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int SIZE = 100000;
    vector<int> arr(SIZE);
    srand(time(0)); // Seed for random numbers

    // Filling the array with random numbers
    if (rank == 0) {
        for (int i = 0; i < SIZE; ++i) {
            arr[i] = rand() % 100; // Modulus 100 for numbers between 0 and 99
        }
        cout << "Array filled with random numbers.\n";
    }

    // Scatter the array data among processes
    vector<int> localArr(SIZE / size);
    MPI_Scatter(arr.data(), SIZE / size, MPI_INT, localArr.data(), SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Timing the sorting process
    auto start = high_resolution_clock::now();
    cout << "Process " << rank << " sorting array...\n";
    quickSort(localArr, 0, localArr.size() - 1);
    cout << "Process " << rank << " sorting completed.\n";
    auto end = high_resolution_clock::now();

    // Gather the sorted portions of the array from all processes
    MPI_Gather(localArr.data(), SIZE / size, MPI_INT, arr.data(), SIZE / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto duration = duration_cast<microseconds>(end - start);
        cout << "Execution Time: " << duration.count() << " microseconds\n";
    }

    MPI_Finalize();
    return 0;
}
        