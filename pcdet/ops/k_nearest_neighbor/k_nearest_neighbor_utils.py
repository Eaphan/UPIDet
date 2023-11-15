import torch
import torch.nn.functional

try:
    from ._k_nearest_neighbor_cuda import _k_nearest_neighbor_cuda
except ImportError as e:
    _k_nearest_neighbor_cuda = None
    print('Failed to load one or more CUDA extensions, performance may be hurt.')
    print('Error message:', e)

def squared_distance(xyz1: torch.Tensor, xyz2: torch.Tensor):
    """
    Calculate the Euclidean squared distance between every two points.
    :param xyz1: the 1st set of points, [batch_size, n_points_1, 3]
    :param xyz2: the 2nd set of points, [batch_size, n_points_2, 3]
    :return: squared distance between every two points, [batch_size, n_points_1, n_points_2]
    """
    assert xyz1.shape[-1] == xyz2.shape[-1] and xyz1.shape[-1] <= 3  # assert channel_last
    batch_size, n_points1, n_points2 = xyz1.shape[0], xyz1.shape[1], xyz2.shape[1]
    dist = -2 * torch.matmul(xyz1, xyz2.permute(0, 2, 1))
    dist += torch.sum(xyz1 ** 2, -1).view(batch_size, n_points1, 1)
    dist += torch.sum(xyz2 ** 2, -1).view(batch_size, 1, n_points2)
    return dist


def k_nearest_neighbor(input_xyz: torch.Tensor, query_xyz: torch.Tensor, k: int, cpp_impl=True):
    """
    Calculate k-nearest neighbor for each query.
    :param input_xyz: a set of points, [batch_size, n_points, 3] or [batch_size, 3, n_points]
    :param query_xyz: a set of centroids, [batch_size, n_queries, 3] or [batch_size, 3, n_queries]
    :param k: int
    :param cpp_impl: whether to use the CUDA C++ implementation of k-nearest-neighbor
    :return: indices of k-nearest neighbors, [batch_size, n_queries, k]
    """
    def _k_nearest_neighbor_py(_input_xyz: torch.Tensor, _query_xyz: torch.Tensor, _k: int):
        dists = squared_distance(_query_xyz, _input_xyz)
        return dists.topk(_k, dim=2, largest=False).indices.to(torch.long)

    if input_xyz.shape[1] <= 3:  # channel_first to channel_last
        assert query_xyz.shape[1] == input_xyz.shape[1]
        input_xyz = input_xyz.transpose(1, 2).contiguous()
        query_xyz = query_xyz.transpose(1, 2).contiguous()

    if cpp_impl and callable(_k_nearest_neighbor_cuda) and input_xyz.is_cuda and query_xyz.is_cuda:
        return _k_nearest_neighbor_cuda(input_xyz.contiguous(), query_xyz.contiguous(), k)
    else:
        return _k_nearest_neighbor_py(input_xyz, query_xyz, k)
