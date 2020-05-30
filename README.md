# thexp

一个主要面向 Pytorch 的深度学习试验框架&工具包

## Features

- 覆盖全训练流程：从数据集准备，到训练，日志输出，模型保存，随机种子管理，试验管理与复盘等一整套流程的功能支持，使你能够专注于试验核心实现。
- 充分解耦：可以单独利用该框架中的saver、logger、meter、experiment来帮助你保存模型，输出/美化日志。
- 全局试验管理模式：所有试验全局唯一，从而能够在全局完成管理，能够完美的在不污染原分支的情况下进行试验版本控制。
- 使用前端管理试验：所有按照约定对试验的记录会全部呈现在前端，并进行相关的分析。同时前端保留了diy接口，可以在不修改前端页面的情况下进行统计图表和展示信息的扩展（该部份扩展还不是很完善，目前还没有文档，如有需要，请参考源码进行扩展，位于 [thexp.analyse.web](https://github.com/sailist/thexp/tree/master/thexp/calculate)）。

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

# 快速开始

请参考[Document](https://sailist.github.io/thexp/)

## 简单示例
下面是命令行参数定制和中间变量记录的两个例子，你可以非常顺滑的使用这些功能，并大幅度减少你的相关代码。


如，无需手动定义各参数，只需要三行代码就可以实现一个接受任意命令行参数的程序，参数同时支持多级嵌套，以 `.` 分隔参数名即可，完全符合代码习惯，该部分可以在 [超参数声明](https://sailist.github.io/thexp/tutorial/2-params.html) 部份进行了解。
```python 
from thexp import Params
params = Params()
params.from_args()

print(params.optim.lr)
print(params.dataset)

>>> python ap.py --optim.lr=0.001 --epoch=400 --dataset=cifar10 --k=12
```


以及，在使用变量的同时完成记录变量，并以极少的代码代价更新记录变量的平均值，这极大的简化了变量的记录逻辑，该部分位于 [变量记录]([#变量记录](https://sailist.github.io/thexp/tutorial/3-meter.html))。
```python
from thexp import Meter,AvgMeter

am = AvgMeter()
for j in range(5):
    for i in range(100):
        m = Meter()
        m.percent(m.c_)
        m.a = 1
        m.a += 1
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

上述的代码会输出以下的内容
```
a: 2 | b: 2 | c: 2.73%(43.51%) | c1: 0.9707 | c2: [0.65452176 0.84101534] | c3: [[0.84045666 0.7536433  0.8916327  0.28940183]... | d: [4] | e: {5: '6'}
...
```

更多的功能，参考目前还不是很完善的 [Document](https://sailist.github.io/thexp/)

# 说明
我相信这是一个非常好的，能够对大多数炼丹实验人都有帮助的开源项目，但该项目仍然存在诸多不足，包括：

 - **缺少完善的使用指南和API**。还好 [Tutorial](/tutorial/1-introduction.html) 和 [Cookbook](/cookbook/1-introduction.html) 已经覆盖了大部分的使用说明。此外几乎所有的方法和类我都写了至少一句注释，保证了六个月后我还认识代码（笑。
 - **缺少测试**。所有已修复的 bug 均是在我使用过程中不断迭代修复 bug ，因此可能存在诸多没有发现的问题，由于精力有限以及只学习过理论化的测试方法，因此无法进行十分正规的测试。在不同电脑上的兼容情况可能也没有办法得到很好的保证。
 - **缺少项目开发经验**。这不是一个工程项目，由于精力有限，以及没有实习经验，因此我没有比较好的基于 git 的开发方法和分支管理办法，也没有采用什么工程化的开发流程，该项目更像是我一拍脑袋拿出来的项目。
 - **缺少i18n。** 开发过程中写出来的注释都是中英文混杂的 Orz。

因此，如果该库对你有帮助，或者你在使用过程中发现了问题，或者你希望该库更加的完善，那么希望你也可以参与到贡献中。仍然是由于精力问题，我不太懂贡献流程，因此除了代码问题、bug修复等，如果有人能贡献如何贡献的流程，或者贡献有人贡献出的如何贡献的流程（套娃禁止），也是非常有帮助的。

为一个开源项目进行贡献是十分消耗精力的一件事，因此十分感谢参与贡献的朋友。


祝大家炼丹愉快！