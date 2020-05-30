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

    用于还没有正式添加到 thexp 中的方法或类，则测试或实验使用体验良好后会陆续添加啊进去

    目前的部份有
    TODO nddataset
        由于torchvisioin 中提供的图片数据集初始格式很多为 np.ndarray ，而转换却需要用到 PIL 库中的函数，因此图片
        增广时效率会受影响。如果数据集一开始就为 PIL.Image 格式或采用 numpy 系来完成 transform，那么效率应该能够
        大幅提升。目前的测试结果是，简单增广（randomCrop 和 randomFlip）大概有两倍的时间差异，所以这一部分需要解
        决以下

    TODO randaugment
        Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le，
        RandAugment: Practical automated data augmentation with a reduced search space
        https://arxiv.org/abs/1909.13719
"""

