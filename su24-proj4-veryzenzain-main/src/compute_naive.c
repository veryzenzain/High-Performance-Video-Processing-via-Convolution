#include "compute.h"

void swap(int32_t *a, int32_t *b) {
    int32_t temp = *a;
    *a = *b;
    *b = temp;
}

int32_t* convolution(matrix_t *a_matrix, matrix_t *flipped_b, uint32_t x, uint32_t y) {
    uint32_t p = flipped_b->rows;
    uint32_t q = flipped_b->cols;
    uint32_t d = a_matrix->cols;
    
    int32_t* result_data = (int32_t*)malloc(x * y * sizeof(int32_t));

    for (uint32_t i = 0; i < x; i++) {
        for (uint32_t j = 0; j < y; j++) {
            int32_t sum = 0;
            for (uint32_t r = 0; r < p; r++) {
                for (uint32_t s = 0; s < q; s++) {
                    int32_t a = a_matrix->data[(j + s) + (i + r) * d];
                    int32_t b = flipped_b->data[s + r * q];
                    sum += a * b;
                }
            }
            result_data[j + i * y] = sum;
        }
    }

    return result_data;
}

int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
    uint32_t x = a_matrix->rows - b_matrix->rows + 1;
    uint32_t y = a_matrix->cols - b_matrix->cols + 1;

    *output_matrix = (matrix_t*)malloc(sizeof(matrix_t));
    (*output_matrix)->rows = x;
    (*output_matrix)->cols = y;
    (*output_matrix)->data = (int32_t*)malloc(x * y * sizeof(int32_t));

    matrix_t *flipped_b = (matrix_t*)malloc(sizeof(matrix_t));
    flipped_b->rows = b_matrix->rows;
    flipped_b->cols = b_matrix->cols;
    flipped_b->data = (int32_t*)malloc(flipped_b->rows * flipped_b->cols * sizeof(int32_t));

    memcpy(flipped_b->data, b_matrix->data, b_matrix->rows * b_matrix->cols * sizeof(int32_t));

    for (uint32_t row = 0; row < flipped_b->rows; row++) {
        for (uint32_t col = 0; col < flipped_b->cols / 2; col++) {
            swap(&flipped_b->data[row * flipped_b->cols + col], &flipped_b->data[row * flipped_b->cols + (flipped_b->cols - col - 1)]);
        }
    }

    for (uint32_t col = 0; col < flipped_b->cols; col++) {
        for (uint32_t row = 0; row < flipped_b->rows / 2; row++) {
            swap(&flipped_b->data[row * flipped_b->cols + col], &flipped_b->data[(flipped_b->rows - row - 1) * flipped_b->cols + col]);
        }
    }
    int32_t* result = convolution(a_matrix, flipped_b, x, y);
    memcpy((*output_matrix)->data, result, x * y * sizeof(int32_t));
    free(result);
    free(flipped_b->data);
    free(flipped_b);

    return 0;
}


// Executes a task
int execute_task(task_t *task) {
  matrix_t *a_matrix, *b_matrix, *output_matrix;

  char *a_matrix_path = get_a_matrix_path(task);
  if (read_matrix(a_matrix_path, &a_matrix)) {
    printf("Error reading matrix from %s\n", a_matrix_path);
    return -1;
  }
  free(a_matrix_path);

  char *b_matrix_path = get_b_matrix_path(task);
  if (read_matrix(b_matrix_path, &b_matrix)) {
    printf("Error reading matrix from %s\n", b_matrix_path);
    return -1;
  }
  free(b_matrix_path);

  if (convolve(a_matrix, b_matrix, &output_matrix)) {
    printf("convolve returned a non-zero integer\n");
    return -1;
  }

  char *output_matrix_path = get_output_matrix_path(task);
  if (write_matrix(output_matrix_path, output_matrix)) {
    printf("Error writing matrix to %s\n", output_matrix_path);
    return -1;
  }
  free(output_matrix_path);

  free(a_matrix->data);
  free(b_matrix->data);
  free(output_matrix->data);
  free(a_matrix);
  free(b_matrix);
  free(output_matrix);
  return 0;
}
