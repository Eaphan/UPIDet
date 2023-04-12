#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>

#include "sampling_gpu.h"

extern THCState *state;


int gather_points_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor points_tensor, at::Tensor idx_tensor, at::Tensor out_tensor){
    const float *points = points_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *out = out_tensor.data<float>();

    gather_points_kernel_launcher_fast(b, c, n, npoints, points, idx, out);
    return 1;
}


int gather_points_grad_wrapper_fast(int b, int c, int n, int npoints, 
    at::Tensor grad_out_tensor, at::Tensor idx_tensor, at::Tensor grad_points_tensor) {

    const float *grad_out = grad_out_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *grad_points = grad_points_tensor.data<float>();

    gather_points_grad_kernel_launcher_fast(b, c, n, npoints, grad_out, idx, grad_points);
    return 1;
}


// int farthest_point_sampling_wrapper(int b, int n, int m,
//     at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

//     const float *points = points_tensor.data<float>();
//     float *temp = temp_tensor.data<float>();
//     int *idx = idx_tensor.data<int>();

//     farthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
//     return 1;
// }

int furthest_point_sampling_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx);
    return 1;
}

int furthest_point_sampling_matrix_wrapper(int b, int n, int m, 
    at::Tensor matrix_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {
    
    const float *matrix = matrix_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_matrix_kernel_launcher(b, n, m, matrix, temp, idx);
    return 1;
}

int furthest_point_sampling_weights_wrapper(int b, int n, int m,
    at::Tensor xyz_tensor, at::Tensor weights_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *xyz = xyz_tensor.data<float>();
    const float *weights = weights_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    furthest_point_sampling_weights_kernel_launcher(b, n, m, xyz, weights, temp, idx);
    return 1;
}


int furthest_point_sampling_with_dist_wrapper(int b, int n, int m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    // cudaStream_t stream = THCState_getCurrentStream(state);
    furthest_point_sampling_with_dist_kernel_launcher(b, n, m, points, temp, idx);
    return 2;
}