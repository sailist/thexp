thexp是对pytorch的一个高级封装，专门用于简化实验流程，从而可以专注于内容本身。

我想要的结果是所有实验中产生的数据都按照可定制、可管理的方法存储下来，从而实现可任意复现、任意回档、自动记录、自动对比等一系列功能。

 - [ ] 下次更新时间：6月份，赶完这波实验的，会根据这次实验中的不爽点大幅优化多个函数和功能，感兴趣可以先star一下。

# thexp（TorchExperiment）

**thexp**的核心目的管理试验和提供pytorch训练模型的流程的整个框架。在设计该框架的过程中，尽可能的做到了解耦，使得各个模块之间尽可能的独立，从而可以单独使用。


## Features

- 覆盖全训练流程：从数据集准备，到训练，日志输出，模型保存，重新训练等一整套流程的快速操作。
- 充分解耦：可以单独利用该框架中的saver、logger、meter、experiment来帮助你保存模型，输出/美化日志，管理试验。

# 安装
```bash
pip install thexp
```
或者
```
git clone --depth=1 https://github.com/sailist/thexp
cd thexp
```
## 测试
```
python -m pytest
```
（虽然基本没怎么写测试用例

# Quick Start
thexp 主要提供以下几个功能
 - [超参数声明](#超参数声明)
    - [基本使用](#基本使用)
    - [自定义类](#自定义类)
    - [命令行参数支持](#命令行参数支持)
 - [变量记录](#变量记录)
    - [格式化](#格式化)
    - [平均](#平均)
 - [日志输出](#日志输出)
 - [随机种子管理](#随机种子管理)
 - [模型保存](#模型保存)
 - [试验过程记录](#试验过程记录)
    - [排查试验记录](#排查试验记录)
 - 一套训练模型的[完整流程](#完整流程)
    - [callbacks](#callbacks)
 - [全局变量获取](#全局变量获取)

> 下面所有的例子都位于[./examples](./examples)中

## 超参数声明
thexp.frame.params.Params 专注于记录各种超参数和配置变量，并通过Fire库提供便捷的命令行参数支持

### 基本使用
Params类默认带有device、epoch、eidx、idx、global_step等变量，主要用于配合trainer，如果需要有不带任何参数的，请使用`thexp.frame.params.BaseParams`
最简单的使用方式：
```python
from thexp import Params

params = Params()
params.epoch = 400
params.batch_size = 25
print(params)
```
输出
````bash
Params[('epoch', 400),
 ('eidx', 1),
 ('idx', 0),
 ('global_step', 0),
 ('device', 'cpu'),
 ('batch_size', 25)]
````

### 自定义类
自定义Params类来实现默认值：
```python
from thexp import Params

class BaseParams(Params):

    def __init__(self):
        super().__init__()
        self.batch_size = 50
        self.topk = (1,2,3,4)
        self.optim_option(lr=0.009,moment=0.9) 
        # 显示声明变量可以使用：
        # self.optim = self.optim_option(lr=0.009,moment=0.9)

params = BaseParams()
...
```
输出：
```bash
>>> BaseParams[('epoch', 10),
 ('eidx', 1),
 ('idx', 0),
 ('global_step', 0),
 ('device', 'cpu'),
 ('batch_size', 50),
 ('topk', (1, 2, 3, 4)),
 ('optim', {'lr': 0.009, 'moment': 0.9})]
```

### 命令行参数支持
命令行参数支持（命令行参数同时支持多级变量定义，用'.'断开即可）：
```python
from thexp import Params

params = Params()
params.from_args()
print(params)

# 运行（注意optim.lr）：
# python p.py --epoch=400 --optim.lr=0.01
```
```bash
>>> 

>>> Params[('epoch', 400),
 ('eidx', 1),
 ('idx', 0),
 ('global_step', 0),
 ('device', 'cpu'),
 ('optim', {'lr': 0.001})]
```




## 变量记录
thexp.frame.meter.Meter 专注于记录变量（并用于日志输出），并且尽可能的不干扰训练逻辑：
```python
from thexp.frame import Meter
import torch

m = Meter()
m.a = 1
m.b = "2"
m.c = torch.rand(1)[0]

m.c1 = torch.rand(1)
m.c2 = torch.rand(2)
m.c3 = torch.rand(4, 4) # 会只保留第一行
print(m)
```
输出：
```
@a=1  @b=2  @c=0.7072  @c1=0.3892  @c2=tensor([0.7060, 0.4127])  @c3=tensor([[0.2227, 0.8771, 0.7446, 0.7601],...
```
所有在meter中的变量在取出的时候都是正常的，只有在输出的时候会被格式化成各种样子。

### 格式化
有的时候，我们希望更改输出的精度，或者将小数变成百分比的形式输出，这利用meter很容易就可以实现：
```python
from thexp.frame import Meter
m = Meter()
m.a = 0.236
m.percent(m.a_) # 此外还有 m.float m.int 等格式化方法
print(m)
```
输出：
```
@a=23.60%
```
> 这里一个疑惑的地方就是 `m.a_`，该变量实际等价于去掉了下划线的变量名字符串，也就是 `m.a_ == "a"`，在上面的代码中，直接传进去字符串"a" 也是可以的，**但我的这种实现方式对IDE更友好了，前面的变量名可以通过自动补全来完成**，随后加一个下划线即可以表示该变量名自身。

### 平均
一个batch可能没有办法很好的看出来我们的训练效果，我们希望能看到某一个值（如loss）一整个epoch下来的平均值，这由 `thexp.frame.meter.AvgMeter` 结合 `Meter` 来实现：
```python
from thexp.frame import AvgMeter
am = AvgMeter()
for j in range(5):
    for i in range(100):
        m = Meter()
        m.percent(m.c_)
        m.a = 1
        m.b = "2"
        m.c = torch.rand(1)[0]

        m.c1 = torch.rand(1)
        m.c2 = torch.rand(2)
        m.c3 = torch.rand(4, 4)
        m.d = [4]
        m.e = {5: "6"}
        # print(m)
        am.update(m)
    print(am)
``` 
输出：
```
@a=1  @b=2  @c=33.27%(47.41%)  @c1=0.0165(0.4997)  @c2=tensor([0.4340, 0.0787])  @c3=tensor([[0.6270, 0.2181, 0.2154, 0.5420],...  @d=[4]  @e={5: '6'}
@a=1  @b=2  @c=0.42%(48.48%)  @c1=0.5041(0.4849)  @c2=tensor([0.6771, 0.0458])  @c3=tensor([[0.6650, 0.1666, 0.0713, 0.0401],...  @d=[4]  @e={5: '6'}
@a=1  @b=2  @c=88.18%(47.52%)  @c1=0.6152(0.5005)  @c2=tensor([0.5661, 0.6359])  @c3=tensor([[0.0794, 0.7921, 0.5871, 0.7822],...  @d=[4]  @e={5: '6'}
@a=1  @b=2  @c=53.72%(49.20%)  @c1=0.7797(0.5090)  @c2=tensor([0.7977, 0.1521])  @c3=tensor([[0.0595, 0.3507, 0.0569, 0.5075],...  @d=[4]  @e={5: '6'}
@a=1  @b=2  @c=77.53%(49.44%)  @c1=0.6968(0.5027)  @c2=tensor([0.4919, 0.1815])  @c3=tensor([[0.7555, 0.9526, 0.3055, 0.3592],...  @d=[4]  @e={5: '6'}
```

## 日志输出
`thexp.frame.logger.Logger` 专注于日志输出：
```python
import time
from thexp.frame.logger import Logger
from thexp.frame.meter import Meter
from thexp.frame.params import Params
logger = Logger()

meter = Meter()
meter.a = 3.13524635465
meter.b = "long text;long text;long text;long text;long text;long text;long text;long text;long text;long text;"

for i in range(100):
    logger.inline(meter,prefix=examples)
    time.sleep(0.2)
logger.info(1,2,3,{4:4},prefix=examples)
logger.info(meter,prefix=examples)
```
如果你的终端支持，你将会看到在同一行不断滚动的输出。如果不是使用 `logger.inline` 方法输出，超出一行的的输出将不会被压缩到一行而是完全输出出来。

> logger 并不是线程安全的，但一般来说机器学习任务的需求也并不需要满足线程安全。

## 随机种子管理
为了保证复现结果，如数据集的随机采样结果，或者模型的训练结果，需要在训练开始的时候固定随机种子，所以会有 保存、读取、加载等对随机种子的需求，该点可以使用 `thexp.frame.rndmanager.RndManager` 来完成：
```python

from thexp.utils import random
import torch.nn as nn
import torch
from torch.optim import SGD
stt = random.get_state()
for i in range(2):
    data = torch.rand(5,2)
    y = torch.tensor([0,0,0,0,0])
    model = nn.Linear(2,2)
    
    sgd = SGD(model.parameters(),lr=0.01)
    logits = model(data)
    loss = nn.CrossEntropyLoss()(logits,y)
    loss.backward()
    sgd.step()
    sgd.zero_grad()
    
    print(list(model.parameters()))
    random.set_state(stt)
```
可以对比一下两次输出的参数，都是完全相同的。

> 随机种子会设置 内置的random库、numpy、torch一共三个库的随机种子。

## 模型保存
`thexp.frame.saver.Saver` 专注于模型/优化器的保存，对于checkpoint，可以实现最大保持数（`max_to_keep`），另外还可以保存不受最大保持数影响的keypoint（关键点...）和最终的模型model，保存每个模型的时候，还可以附带额外的信息（如当前保存的loss啊，epoch啊等的关键参数）到 {模型文件名.json} 中
```python

from thexp.frame.saver import Saver

saver = Saver("./sav")
for i in range(10):
    saver.save_keypoint(i,{"a":i},{"b":i})
    print(saver.find_keypoints())
for i in range(10):
    saver.save_checkpoint(i,{"a":i},{"b":i})
    print(saver.find_checkpoints())
for i in range(10):
    saver.save_model(i,{"a":i},{"b":i})
    print(saver.find_models())
```
三种的主要参数完全一致，torch.save可以保存在 saver 也可以保存，实质上这三个方法只是在保存特性和后缀名上有不同，文件内容完全看输入参数。

## 试验过程记录
`thexp.frame.experiment.Experiment` 能够保存试验过程中的文件变更（如模型的创建、覆盖，日志的输出，tensorboard文件的生成等），这些默认都在 Traniner类中实现了，但你也可以单独使用该类：
```python
import time
from thexp.frame import Logger,Saver,Experiment

exp = Experiment("./exp")

@exp.keycode()
def train():
    logger = Logger()
    logger.add_log_dir(exp.hold_exp_part("log",[".log"]))
    save = Saver(exp.hold_exp_part("save",[".ckpt",".pth"]))
    for i in range(10):
        for j in range(5):
            save.save_checkpoint(j, {}, {})
            time.sleep(0.2)
            logger.info(i)

exp.start_exp()
train()
```
当本次运行结束时（无论是否正常退出），都会在`./exp`目录下，和`homedir/.thexp/`下，记录本次试验中的文件变动和代码快照（如果和已快照代码不相同）

文件变动通过 `exp.hold_exp_part()` 方法来添加监视目录和后缀（后缀是全局），在上面的代码中，监视的是saver产生的模型和logger产生的日志文件。
```python
logger.add_log_dir(exp.hold_exp_part("log",[".log"]))
save = Saver(exp.hold_exp_part("save",[".ckpt",".pth"]))
```

代码快照既可以通过注解，也可以直接传入参数来实现，类似：
```python
exp.keycode(sys.modules["__main__"])
```

### 排查试验记录
通过`thexp.frame.experiment.ExperimentViewer` 可以还算方便的排查你做过的试验以及你当前保存的模型/日志是哪次试验遗留下来的：
```python
from thexp.frame import ExperimentViewer
from datetime import timedelta
import pprint as pp
expview = ExperimentViewer()
pp.pprint(expview.list())
pp.pprint(expview.recent(timedeltaobj=timedelta(minutes=20)))
pp.pprint(expview.backtracking("ckpt"))
pp.pprint(expview.exp_code("20-03-08121413")) # An key of the list()
pp.pprint(expview.exp_info("20-03-08122447")) # An key of the list()
```

## 完整流程
上述模块完全可以单独在你的旧代码上使用，但是如果你使用该库，并开了一个新项目，那么我推荐你使用将上述模块融合到一起的 trainer 类（`thexp.frame.trainer.Trainer`），以下是一个快速示例：

任意模型
```python
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
```
利用 thexp 建立一个trainer来控制训练流程
```python
from thexp.frame import Meter, Params, Trainer
class MyTrainer(Trainer):
    def initial_trainer(self,params:Params): # 在该方法中初始化模型、优化器、数据集
        from torch.optim import SGD
        from torchvision import transforms
        from torchvision.datasets import FakeData
        from thexp.utils.date.dataloader import DataLoader

        self.model = MyModel()
        self.optim = SGD(self.model.parameters(), lr=params.lr)
        dataset = FakeData(image_size=(28, 28), transform=transforms.ToTensor())
        train_loader = eval_loader = test_loader = DataLoader(dataset, shuffle=True, batch_size=32, drop_last=True)

        self.regist_databundler(
            train=train_loader,
            test=test_loader,
            eval=eval_loader,
        )
        self.cross = nn.CrossEntropyLoss()

    def train_batch(self, eidx, idx, global_step, batch_data, params, device): 
        # 在该方法中写 每个batch 中的模型训练流程
        optim, cross = self.optim, self.cross
        meter = Meter()
        xs, ys = batch_data

        # 训练逻辑
        logits = self.model(xs)
        meter.loss = cross(logits, ys)

        # 反向传播
        meter.loss.backward()
        optim.step()
        optim.zero_grad()

        return meter

```
创建 param 和 trainer，开始训练
```python
params = Params()
params.lr = 0.01
params.build_exp_name("mytrainer", "lr")

trainer = MyTrainer(params)
trainer.initial_exp("./experiment")

trainer.train()
```

随后日志输出 / 模型checkpoint / epoch控制 / 试验代码快照 thexp 全都会自动完成，你所需要的，只是多建立一个子类，然后在训练逻辑中，把需要日志输出的代码放到meter类中并在方法中返回。

上述所有的例子都位于[./examples](./examples)中

### callbacks
callbacks和keras中的回调类相似，在整个训练流程、每次epoch、iteration、eval、test开始前和结束后实现某些处理逻辑。

利用trainer，其构建方法主要由以下几行代码完成
```python
params = Params()
params.build_exp_name("mytrainer",...)
trainer = MyTrainer(params)
trainer.initial_exp("./experiment")

trainer.train()#...
```
其中，在调用 `trainer.initial_exp` 的时候，会在内部调用 `trainer.initial_callback`和`trainer.initial_trainer`方法，因此初始化方法通过重写`initial_callback`来完成，而回调函数推荐在`initial_callback`方法中实现。

使用方法如下（Trainer类中该方法的实现，也即默认情况下的callback）：
```
class Trainer:
    def initial_callback(self):
        from .callbacks import EvalCallback, LoggerCallback
        ec = EvalCallback(1, 5) # 决定几个epoch eval一次，几个eopch test一次
        ec.hook(self)

        lc = LoggerCallback() # 对试验过程进行日志输出
        lc.hook(self)
```

## 全局变量获取
```
Usage:
 thexp init # 初始化
 thexp config -l # 列出全部变量
 thexp config --global <name> <value> # 设置某全局变量 
 thexp config --global -u <name> # 删除某全局变量
 thexp config --global -l # 列出所有全局变量
 thexp config --local <name> <value>  # ... 同上
 thexp config --local -u <name> 
 thexp config --local -l
```
如
```
thexp config --global datasets /path/to/all/datasets
```

在项目中，就可以直接使用：
```python
from thexp.frame.experiment import globs

print(globs[nddatasets])

globs.update("local","a",1)
globs.list_config(globa=True,local=True)
print(globs.glob_fn)
```


# 基本结构
前三个模块是主要模块

## frame
训练框架一套包

## calculate
提供了训练中一些有时会用到但是需要轮子的场景，如正确率啊，rampup啊等

## utils
提供了一些对torch内部的一些功能的补充和包内自用的一些工具

## future
准备在未来添加但是还在测试基本不能正常使用的库

## cli
命令行控制

## base_classes
提供了一些thexp中用到的基类、元类或数据结构


# More
剩下的可能就是测试用例的补全和一些细节或bug的完善了，大的变更基本已经不会再有，后期用到了再修吧。
 - [ ] Plotter：用于将tensorboard中的图像等抽取出来保存到本地，这个等我跑完试验开始写论文的时候再说吧...
 - [ ] utils：添加一些更多的trick或torch操作，这个我用到了就会不断向上添加的
 - [x] （持续更新...）[thexp-implement](https://github.com/sailist/thexp-implement)：以后复现各种模型我会统一用这个框架来复现，争取让每一个我过手的模型都简洁清晰2333
 - [x] （2020年3月11日）全局变量：利用experiment来实现全局变量，比如数据集啊，保存根路径啊之类的
 - [ ] 考虑Trainer多继承时候的params分配问题，方便Trainer能够更好的通过多继承关系组织起来  
 
 - [ ] 以文件夹为单位，实现state_dict接口
 - [ ] 利用jupyter绘制报表和控制训练流程？
 - [ ] 多次重复试验的特定结果记录 
 
## 低优先级
 - [ ] 任务队列？在显存不够的时候，一个试验跑完了另一个试验跑，类似这样的...
 
 
## 细节记录



# Log

- 2020年3月10日：1.0.1：初步完成所有的框架
- 2020年3月11日：使用的过程中出了一些bug，修改了一下，现在正在用来跑我的试验，舒服了...
- 2020年4月16日：为便于试验记录，新增reporter和相应callback AutoReport