"""
    Copyright (C) 2020 Shandong University

    This program is licensed under the GNU General Public License 3.0 
    (https://www.gnu.org/licenses/gpl-3.0.html). 
    Any derivative work obtained under this license must be licensed 
    under the GNU General Public License as published by the Free 
    Software Foundation, either Version 3 of the License, or (at your option) 
    any later version, if this derivative work is distributed to a third party.

    The copyright for the program is owned by Shandong University. 
    For commercial projects that require the ability to distribute 
    the code of this program as part of a program that cannot be 
    distributed under the GNU General Public License, please contact 
            
            sailist@outlook.com
             
    to purchase a commercial license.
"""
from typing import List, Union, Tuple

import torch


def classify(preds: torch.Tensor,
             labels: torch.Tensor,
             cacu_rate: bool = False,
             topk: List[int] = None) -> Union[List[float], Tuple[int, List[int]]]:
    """
    用于在分类问题中计算准确率

    Args:
        preds: [batch,logits]
        labels: [labels,]
        cacu_rate: 计算正确率而不是计数
        topk: list(int) ，表明计算哪些topk，默认计算 top1 和 top5

    Returns:
        if cacu_rate:
            [topk_rate,...]
        else:
            total, [topk_count,...]
    """
    if topk is None:
        topk = (1, 5)
    k = topk
    _, maxk = torch.topk(preds, max(*k), dim=-1)
    total = labels.size(0)
    test_labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

    if cacu_rate:
        return [(test_labels == maxk[:, 0:i]).sum().item() / total for i in k]
    else:
        return total, [(test_labels == maxk[:, 0:i]).sum().item() for i in k]
