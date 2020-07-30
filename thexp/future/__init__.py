"""


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

