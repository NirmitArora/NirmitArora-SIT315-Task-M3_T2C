#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>
#include <mpi.h>
#include <CL/cl.h>
#include <cstring>

using namespace std;
using namespace std::chrono;

void swap(int &a, int &b) {
    int t = a;
    a = b;
    b = t;
}

int partition(vector<int> &arr, int low, int high) {
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

void quickSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void execute_quicksort_parallel(int array[], int array_size, int node_rank, int node_count) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint device_count;
    cl_uint platform_count;
    cl_int status;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem buffer;

    status = clGetPlatformIDs(1, &platform_id, &platform_count);
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &device_count);
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &status);

    // Create memory buffer
    buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * array_size, NULL, &status);
    status = clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE, 0, sizeof(int) * array_size, array, 0, NULL, NULL);

    // Create OpenCL program from source
    const char *code =
        "__kernel void parallel_quicksort(__global int* array, int low, int high) {"
        " if (low < high) {"
        " int p = array[high];"
        " int i = low - 1;"
        " for (int j = low; j <= high - 1; j++) {"
        " if (array[j] < p) {"
        " i++;"
        " int temp = array[i];"
        " array[i] = array[j];"
        " array[j] = temp;"
        " }"
        " }"
        " int k = array[i + 1];"
        " array[i + 1] = array[high];"
        " array[high] = k;"
        ""
        " int partition_index = i + 1;"
        " parallel_quicksort(array, low, partition_index - 1);"
        " parallel_quicksort(array, partition_index + 1, high);"
        " }"
        "}";
    size_t size = strlen(code);
    program = clCreateProgramWithSource(context, 1, (const char **)&code, (const size_t *)&size, &status);
    status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "parallel_quicksort", &status);

    // Setting OpenCL kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffer);
    status = clSetKernelArg(kernel, 1, sizeof(int), (void *)&array[0]);
    status = clSetKernelArg(kernel, 2, sizeof(int), (void *)&array[array_size - 1]);

    // Execute OpenCL kernel
    size_t global_item_size = array_size;
    size_t local_item_size = 1;
    status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    status = clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, sizeof(int) * array_size, array, 0, NULL, NULL);

    // Cleaning up
    status = clFlush(command_queue);
    status = clFinish(command_queue);
    status = clReleaseKernel(kernel);
    status = clReleaseProgram(program);
    status = clReleaseMemObject(buffer);
    status = clReleaseCommandQueue(command_queue);
    status = clReleaseContext(context);
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
    execute_quicksort_parallel(localArr.data(), localArr.size(), rank, size);
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
