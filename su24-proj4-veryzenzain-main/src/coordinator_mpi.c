#include <mpi.h>
#include "coordinator.h"

#define READY 0
#define NEW_TASK 1
#define TERMINATE -1

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Error: not enough arguments\n");
        printf("Usage: %s [path_to_task_list]\n", argv[0]);
        return -1;
    }

    task_t **task = NULL;
    int num_tasks = 0;

    if (read_tasks(argv[1], &num_tasks, &task) != 0)
    {
        return -1;
    }

    MPI_Init(&argc, &argv);
    int procId = 0;
    int totalProcs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procId);

    if (procId == 0)
    {
        // Manager code
        int nextTask = 0;
        int32_t message = 0;
        MPI_Status status;
        int activeWorkers = totalProcs - 1;

        while (activeWorkers > 0)
        {
            MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (nextTask < num_tasks)
            {
                MPI_Send(&nextTask, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                nextTask++;
            }
            else
            {
                message = TERMINATE;
                MPI_Send(&message, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                activeWorkers--;
            }
        }
    }
    else
    {
        // Worker code
        int32_t message = READY;
        bool keepRunning = true;

        while (keepRunning)
        {
            MPI_Send(&message, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&message, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (message == TERMINATE)
            {
                keepRunning = false;
            }
            else
            {
                if (execute_task(task[message]) != 0)
                {
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
        }
    }

    free(task);
    MPI_Finalize();

    return 0;
}

