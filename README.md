> WestOnline 第三轮 python 考核（人工智能方向）

# 1 考核任务

做一个手写数字识别器，并在 kaggle 上面进行测试识别率

- 使用 numpy
- 识别率尽可能高

# 2 实现过程

通过对《python 神经网络编程》学习了解，一个神经网络的定义分为三部分，分别是初始化、训练、查询.通过这三部分的功能的实现一个神经网络模型

## 2.1 初始化

> def **init**(self, inputNodes, hiddenNodes, outputNodes, learningrate)

初始化过程需要对输入层节点，隐藏层节点，输出层节点的数目，还有对学习率大小进行定义.并对权重初始化.
学习率的大小影响到了模型的训练时长.但是一般选取 0.01,0.05,0.1,0.2,0.3 等等,不同的模型相应的最适学习率也是不等的.

## 2.2 训练

> def train(self, inputs_list, targets_list)

学习给定的训练集样本后，并优化权重

### 2.2.1

针对给定的训练样本计算输出.

### 2.2.2

将计算得到的输出与所需要输出对比,使用差值来进行网络权重的更新.

### 2.3 查询

> def query(self, inputs_list)

给定输入，从输出节点给出答案.

# 3 实现结果

通过 kaggle 的 Digit Recognizer 中给定的测试集进行训练.
最佳结果为 0.96835.
通过对训练集进行增加 mnist 的训练集,最佳结果为 0.98896.

# 4 总结

在该模型中，学习率以 0.1 优,并且随着训练集的大小有关，训练集大小越大越好.可通过不同形式对模型进行优化如采用 cnn,knn 等.

# 5 优化
通过对图像进行旋转，进行优化。

最佳结果为0.97621


