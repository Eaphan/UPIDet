#ifndef K_NEAREAST_NEIGHBOR_H
#define K_NEAREAST_NEIGHBOR_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

torch::Tensor k_nearest_neighbor_cuda(torch::Tensor input_xyz, torch::Tensor query_xyz, int k);


#endif
