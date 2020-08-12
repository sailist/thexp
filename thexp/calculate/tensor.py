"""

"""
from typing import Tuple, List, Callable

import torch


def rotate_right_angle(x: torch.Tensor, w_dim: int = 2, h_dim: int = 3, degree: int = 90):
    assert degree in {90, 270, 180}
    if degree == 90:
        x = x.transpose(w_dim, h_dim)  # 90
    elif degree == 180:
        x = x.flip(w_dim)
    elif degree == 270:
        x = x.transpose(w_dim, h_dim).flip(h_dim)  # 270

    return x


def split_sub_matrix(mat: torch.Tensor, *sizes):
    """
    将一个[N,M,...,L]的矩阵按 n,m,...l 拆分成 N/n*M/m*...L/l 个 [n,m,...l]的小矩阵

    如果N/n 无法整除，不会报错而是会将多余的裁掉
    example:
        mat = torch.arange(0,24).view(4,6) # shape = [4, 6]
        >> tensor([[ 0,  1,  2,  3,  4,  5],
                   [ 6,  7,  8,  9, 10, 11],
                   [12, 13, 14, 15, 16, 17],
                   [18, 19, 20, 21, 22, 23]])

        split_sub_matrix(mat,2,3) # shape = [2, 2, 2, 3]
        >> tensor([[[[ 0,  1,  2],
                      [ 6,  7,  8]],

                     [[ 3,  4,  5],
                      [ 9, 10, 11]]],

                    [[[12, 13, 14],
                      [18, 19, 20]],

                     [[15, 16, 17],
                      [21, 22, 23]]]])

    :param mat: 一个[N,M,...L] 的矩阵
    :param sizes: n,m,...l 的list, 其长度不一定完全和mat的维数相同
        mat = torch.arange(0,240).view([4,6,10])
        split_sub_matrix(mat,2,3) # shape = [2, 2, 10, 2, 3]
    :return: 一个 [N/row,M/col,row,col] 的矩阵
    """
    for i, size in enumerate(sizes):
        mat = mat.unfold(i, size, size)
    return mat


def onehot(labels: torch.Tensor, label_num):
    """
    convert label to onehot vector
    Args:
        labels:
        label_num:

    Returns:

    """
    return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(1, labels.view(-1, 1), 1)


def cartesian_product(left: torch.Tensor, right: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate cartesian product of the given tensor(s)

    Args:
        left: A pytorch tensor.
        right: A pytorch tensor,
            if None, wile be left X left

    Returns:
        Tuple[torch.Tensor, torch.Tensor]

    Example:
        >>> cartesian_product(torch.arange(0,3),torch.arange(0,5))

    (tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
      tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))

    """
    if right is None:
        right = left

    nleft = left.repeat_interleave(right.shape[0], dim=0)
    nright = right.repeat(*[item if i == 0 else 1 for i, item in enumerate(left.shape)])
    return nleft, nright


def cat_then_split(op: Callable[[torch.Tensor], torch.Tensor], tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Examples:
    >>> ta,tb,tc = cat_then_split(lambda x:model(x),[ta,tb,tc])
    >>> alogits,blogits,clogits = cat_then_split(model,[ta,tb,tc])
    """

    res = op(torch.cat(tensors))  # type: torch.Tensor
    return res.split_with_sizes([i.shape[0] for i in tensors])


def label_smoothing(onthot_labels, epsilon=0.1):
    """
    Applies label smoothing, see https://arxiv.org/abs/1512.00567
    Args:
        onthot_labels:
        epsilon:

    Returns:

    """
    return ((1 - epsilon) * onthot_labels) + (epsilon / onthot_labels.shape[-1])
