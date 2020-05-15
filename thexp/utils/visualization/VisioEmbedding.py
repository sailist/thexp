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

import torch


class VisioEmbedding():
    def __init__(self, writter, global_step=None, tag="default"):
        self.writter = writter
        self.global_step = global_step
        self.tag = tag
        self.mats = []
        self.metadatas = []
        self.label_imgs = []

    def add_embedding(self, mat, metadata=None, label_img=None):
        self.mats.append(mat.detach().cpu())
        self.metadatas.append(metadata.detach().cpu())
        self.label_imgs.append(label_img.detach().cpu())

    def flush(self, max_len=None):
        self.writter.add_embedding(torch.cat(self.mats[:max_len]),
                                   torch.cat(self.metadatas[:max_len]),
                                   self.label_imgs,
                                   tag=self.tag,
                                   global_step=self.global_step)

