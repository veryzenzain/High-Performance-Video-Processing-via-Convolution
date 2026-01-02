#include "compute.h"
#include <omp.h>
#include <x86intrin.h>


void swap(int32_t *a, int32_t *b) {
    int32_t temp = *a;
    *a = *b;
    *b = temp;
}
    

// Computes the convolution of two matrices
int convolve(matrix_t *a_matrix, matrix_t *b_matrix, matrix_t **output_matrix) {
  // TODO: convolve matrix a and matrix b, and store the resulting matrix in
  // output_matrix
  uint32_t x = a_matrix->rows - b_matrix->rows + 1;
  uint32_t y = a_matrix->cols - b_matrix->cols + 1;

  *output_matrix = (matrix_t*)malloc(sizeof(matrix_t));

  (*output_matrix)-> rows = x;
  (*output_matrix)-> cols = y;
  (*output_matrix)->data = (int32_t*)malloc(x * y * sizeof(int32_t));

  matrix_t *flipped_b = (matrix_t*)malloc(sizeof(matrix_t));
  flipped_b->rows = b_matrix->rows;
  flipped_b->cols = b_matrix->cols;
  flipped_b->data = (int32_t*)malloc(flipped_b->rows * flipped_b->cols * sizeof(int32_t*));

  // Copy matrix
  memcpy(flipped_b->data, b_matrix->data, b_matrix->rows * b_matrix->cols * sizeof(int32_t));

  // Horizontal flip
  #pragma omp parallel for
  for (uint32_t row = 0; row < flipped_b->rows; row++) {
      for (uint32_t col = 0; col < flipped_b->cols / 2; col++) {
        swap(&flipped_b->data[row * flipped_b->cols + col], &flipped_b->data[row * flipped_b->cols + (flipped_b->cols - col - 1)]);
      }
  }

  // Vertical flip
  #pragma omp parallel for
  for (uint32_t col = 0; col < flipped_b->cols; col++) {
    for (uint32_t row = 0; row < flipped_b->rows / 2; row++) {
        swap(&flipped_b->data[row * flipped_b->cols + col], &flipped_b->data[(flipped_b->rows - row - 1) * flipped_b->cols + col]);        
    }
  }

uint32_t p = b_matrix->rows;
uint32_t q = b_matrix->cols;
uint32_t d = a_matrix->cols;

#pragma omp parallel for collapse(2)
for (uint32_t i = 0; i < x; i++) {
    for (uint32_t j = 0; j < y; j++) {
        int32_t sum = 0;
        __m256i sum_vec = _mm256_setzero_si256();  // Initialize sum_vec to zero
        uint32_t r, s;
        
        for (r = 0; r < p; r++) {
            for (s = 0; s + 7 < q; s += 8) {
                __m256i val1 = _mm256_loadu_si256((__m256i*)&a_matrix->data[(j + s) + (i + r) * d]);
                __m256i val2 = _mm256_loadu_si256((__m256i*)&flipped_b->data[s + r * q]);
                __m256i mul = _mm256_mullo_epi32(val1, val2);

                sum_vec = _mm256_add_epi32(sum_vec, mul);  // Accumulate the results in sum_vec
            }
            
            for (; s < q; s++) {
                sum += a_matrix->data[(j + s) + (i + r) * d] * flipped_b->data[s + r * q];
            }
        }

        // Sum up the elements of sum_vec at the end
        __m128i val3 = _mm256_extracti128_si256(sum_vec, 0);
        __m128i val4 = _mm256_extracti128_si256(sum_vec, 1);
        __m128i sum_vals = _mm_add_epi32(val3, val4);
        sum_vals = _mm_hadd_epi32(sum_vals, sum_vals);
        sum_vals = _mm_hadd_epi32(sum_vals, sum_vals);

        sum += _mm_extract_epi32(sum_vals, 0);

        (*output_matrix)->data[j + i * y] = sum;
    }
}


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

