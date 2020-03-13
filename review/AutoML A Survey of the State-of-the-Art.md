# AutoML: A Survey of the State-of-the-Art

## 摘要

本文引用自 [AutoML & NAS综述](https://zhuanlan.zhihu.com/p/104902661)。

论文地址：[AutoML: A Survey of the State-of-the-Art](https://arxiv.org/pdf/1908.00709.pdf)

深度学习现在已经深入到了各行各业，然而不论是对工业界还是学术界，从头对特定任务建立一个深度学习模型仍然非常耗费时间和资源，以及需要专业经验。

为了缓解这种问题，最近越来越多的研究项目关注于自动化机器学习（Automated Machine Learning, AutoML）这篇文章提供了截止2019年底，AutoML领域最新成果的比较详尽和全面的介绍。

首先，本文根据其在机器学习任务管道（Pipeline）的位置详细介绍AutoML技术。

之后，还总结了现阶段AutoML最热门的话题—NAS(Neural Architecture Search)算法，并将NAS生成的模型与手工设计的模型进行比较。

最后，本文展示了几个未来研究的开放问题。

## 一、Introduction

不论是目标分类/检测这样的计算机视觉问题，还是语言模型这类的自然语言问题，深度学习已经得到了广泛的应用。而深度学习模型也愈发复杂，以VGG16为例，1亿3千万（130 million）个参数，占据了将近500MB的内存空间。

而且这些笨重的模型，也是他人不断试错得到，需要花很长时间和资源去设计。

于是AutoML应运而生，即使用适当的数据预处理、特征工程、模型选择和模型评价，自动化这个端到端流程的过程。

> Taking human out of learning applications: A survey on automated machine learning,

中的定义则是，AutoML是自动化和机器学习的结合，可以在有限的计算资源下自动地构建机器学习管道。

![v2-1bef1b6dc7869e17ceeaacc807d6b18b_720w](https://pic4.zhimg.com/80/v2-1bef1b6dc7869e17ceeaacc807d6b18b_720w.jpg "机器学习管道")

> 图源AutoML: A Survey of the State-of-the-Art

近期，AutoML突然火起来，是因为学术界和工业界发现其动态结合不同技术，已生成一个易用的端到端系统的能力。就像下图中的谷歌的[Cloud AutoML](https://cloud.google.com/automl/)，一些AI公司已经开始提供这种产品了。

![v2-2a9e4c775bced8d2c16aeb70b8afe068_720w](https://pic1.zhimg.com/80/v2-2a9e4c775bced8d2c16aeb70b8afe068_720w.jpg "谷歌的Cloud AutoML")

那么，为什么现在大家突然觉得有了这种可能，就是因为NAS的出现。

> Neural architecture search with reinforcement learning

这篇论文中，一个强化学习训练的RNN可以自动地搜索最优的网络结构。从此之后，AutoML被带火了，而且主要就是NAS的流量，NAS（网络结构搜索）的目标是通过在一个由不同基础元件组成的预定义搜索空间中，通过某种策略自动搜索，以生成一个鲁棒且高性能的神经网络架构。

![img](https://pic4.zhimg.com/80/v2-eb45110765771cbbcb7eac7621ab521b_720w.jpg)

本文将从两个部分介绍NAS

- - 模型结构

1. 整体结构
2. 基于cell的结构
3. 层级结构（hierarchical structure）
4. 基于形态的结构（morphism-based structure）

- 超参数优化方法（Hyperparameter Optimization，HPO）

1. 强化学习（RL）
2. 进化算法（EA）
3. 梯度下降法（GD）
4. 贝叶斯优化

当然除了NAS之外，机器学习管道的不同部分（**数据准备、特征工程、模型生成以及模型评价** 见图机器学习管道），这么多年也涌现出很多AutoML的技术，每一个部分其实都可以单独写一篇长文，但是为了把所有管道的阶段包括进去，就只写一下每个领域最有代表性的工作。

除此之外，不同领域之间的分界线也是模糊的，模型选择（NAS）遇到的问题也可以认为是超参数选择（HPO）。

---

## 二、数据准备

数据准备是机器学习管道中最开始的部分，但对于诸如医学图像分类等问题，数据不足/不够好，因此鲁棒的AutoML也要解决数据收集的问题。

### 2.1 数据准备

一些常用的基线数据集：

- MNIST（手写数字）
- CIFAR10&CIFAR100
- ImageNet

除此之外，更多公开的数据集可以在如下网站找到：

- Kaggle
- Google Dataset Search（GOODS）
- Elsevier Data Search

不过，对于一些特殊任务，特别是医学任务或者其他涉及隐私的任务，通常是很难从上述网站找到。

对于这种情况，有两种解决途径：

**途径一：数据合成**

也可以理解为数据增广：可以使用各种办法进行数据数据增广，比如裁切，翻折，周围补零（padding），旋转，缩放（resize）等（仿射变换和透视变换也挺好用的，特别是分类数据集上）。常用python工具库有torchvision和Augmentor。

> 《Understanding data augmentation for classification: when to warp?》这篇论文还提出了数据翘曲（data warping）和合成过采样(synthetic over-sampling)。前者通过在数据空间上应用转换生成额外的样本，后者在特征空间中创建额外的样本。

**途径二：数据仿真**

除此之外，对还有一些像自动驾驶的特殊任务，这种任务不可能在一开始就直接在现实世界中进行测试和改进模型，因为成本过高且有潜在安全隐患（一辆车好贵的）。这时就要去做数据仿真，通过尽量匹配真实世界的环境生成数据。OpenAI Gym就是提供多种仿真环境的工具。

> Learning to simulate

使用了基于RL的方法最优化了数据仿真器的参数以控制合成数据的分布。

### 2.2 数据搜索

可以爬虫爬取网上的图片，但是首先，查询结果可能和关键词不符合，更重要的是跨标签之间的噪音，有个很简单的办法就是，如果多个标签(分类）都能检索到这张图，那就干脆把这张图扔掉。

> Harnessing noisy web images for deep representation

根据关键词对搜索结果重新排序并线性提供搜索结果。

**自标签：**

除此之外，有些**错误的标签**，甚至**没标签**的东西，就需要基于学习的自标签方法。

> Towards scalable dataset construction: An active learning approach,

提到的Active Learning就是先选出最不确定的样本，交给人标记，剩下的再由算法自己标。

而为了把人从打标签中解放，进一步加速打标签速度，半监督学习自打标签方法。这种方法由

> A survey on data collection for machine learning: a big data - AI integration perspective

分成了三类：

- - 自训练（self-training）
  - 协同训练（co-training）
  - 协同学习（co-learning）

网上图片复杂，一个标签很难描述好图片，所以就需要有多个标签。

> Recognition from web data: a progressive filtering approach，

就采用了对网上的图片赋予多个标签，如果对图片中多个标签的置信度特别相近，或者预测出最高分的标签和检索图片的标签一样，就拿过来用。

此外，网络数据的分布可能和目标数据集有很大不同，所以也有的工作对网页数据进行微调（fine-tune）

> Webly supervised learning of convolutional networks
> Augmenting strong supervision using web data for fine-grained categorization

之前提到过多标签的论文也提出了**模型训练和网络数据过滤的迭代算法。**

最后，数据不平衡也是网页图片常见问题，因为有的分类图多有的图少，

> Synthetic Minority Over-Sampling Technique

(SMOTE)这篇论文合成少的类别，而不是过采样or欠采样。

Learning from imbalanced data sets with boosting and data generation: the databoost-im approach,则对不均衡的样本结合了提升方法和数据生成。

### 2.3 数据清洗

特征提取之前，进行数据清洗也是常规操作。比较简单的例子是表格数据中的缺失值啊，错误的数据类型之类的问题。

常见的数据清洗方法诸如规范化（standardization），放缩（scaling），定量特征的二值化（binarization of quantitative characteristic），独热编码定性特征（one-hot encoding qualitative characteristic），用平均值填充缺失值（filling missing values with mean value）等等。

而对于图片数据，类似于自标签的技术也可以解决标记错误/缺失问题。不过值得注意的是，数据清洗通常需要进行一些人为定义。下面四篇就是有关自动化数据清洗的

> Alphaclean: Automatic generation of data cleaning pipelines,
> Katara: A data cleaning system powered by knowledge bases and crowdsourcing,
> Activeclean: An interactive data cleaning framework for modern machine learning
> Sampleclean: Fast and reliable analytics on dirty data

---

## 三、特征工程

在工业界有一个共识，**数据和特征决定了机器学习的上界，而模型只是逼近上界。**

特征工程有三大子话题：

- 特征选择
- 特征提取
- 特征构建。

特征提取/构建都是特征变换的变体，因为他们都会生成新的特征。特征提取通常是通过一些函数映射对特征降维，特征构建则是扩展特征空间。

特征提取是减少特征冗余，选取重要特征。特征工程的自动化就是动态结合上述三阶段。

### 3.1 特征选择

对原始数据集降低不相关和冗余的特征，创建原始特征集的子集。

![v2-bcd9285da196f70bfde2520940ea41ce_720w](https://pic3.zhimg.com/80/v2-bcd9285da196f70bfde2520940ea41ce_720w.jpg)

上图表示了特征选择的迭代过程。

**搜索策略**

首先，根据**搜索策略**选择特征子集并进行评估。然后，实现一个**验证过程**来检查子集是否有效。然后重复上述步骤，直到满足停止条件。

经过总结，搜索测率可以分为以下三种：

1. 完全搜索
2. 启发式搜索
3. 随机搜索

**完全搜索**，涉及到穷尽（exhaustive）和非穷尽搜索。并可以细分为四种搜索方式：

1. 1. 广度优先搜索（Breath First Search）
   2. 分支定界法（Branch and Bound）
   3. 集束搜索（Beam Search）（貌似是一种常见于NLP的搜索方式呢）
   4. 最优优先搜索（Best First Search）

**启发式搜索**

1. 1. 顺序正向选择(Sequential Forward Selection SFS)
   2. 顺序反向选择(Sequential Backward Selection SBS)
   3. 双向选择（Bidirectional Search BS）

对于前两种情况，分别从空集添加特性或从完整集删除特性，而对于BS，它同时使用SFS和SBS进行搜索，直到这两种算法获得相同的子集。

在**随机搜索**方面，常用的方法有模拟退火算法Simulated Annealing (SA) 和遗传算法 Genetic Algorithms(GA)。

**子集评价方式**

- 过滤法：先打分，然后设阈值，分数都是方差相关系数，卡方检验，互信息啥的
- 包装法（Wrapper method，递归特征消除）：直接拿特征进行分类，选效果好的
- 嵌入法：特征选择作为学习的一部分，正则化，决策树，深度学习都属于这个方法。

### **3.2. 特征构建**

特征构建是从基本特征空间或原始数据中构造新特征以增强模型的鲁棒性和泛化性的过程，其本质是提高原始特征的表达性。

传统上，这一过程高度依赖于人类的专业知识，最常用的方法之一是预处理转换，如规范化（standardization）、标准化（normalization）、特征离散化（feature discretization）等。

此外，转换操作可能因不同类型的特性而不同。

人工尝试所有可能的特征构建方法是不可能的，所以为了进一步提高效率，就有了**自动模型构建：自动化搜索和评估运算组合的过程**。据说可以达到/超过人类专家的效果。

基于树和遗传算法的搜索方法需要预定义的操作空间。而基于注释的办法，可以用训练样本使用域知识。

### **3.3. 特征提取**

**特征提取是个降维的活儿**。特征提取使用一些映射函数，根据特定的度量提取信息丰富的非冗余特征。

与特征选择不同，特征提取将改变原始特征。特征提取的核心是一个映射函数，可以通过多种方式实现。

常见的传统方法：

- - 主成分分析（PCA）
  - 独立成分分析
  - 等距映射（isomap）
  - 非线性降维
  - 线性判别分析（LDA,LDA都拿来降维的？)

近年来，也有用前向神经网络做自编码器（autoencoder）的方法。

## 四、模型选择

数据完了特征，特征也完了，就要生成模型并设置超参数了。

模型选择有两种两方法：

- - 传统模型选择
  - NAS

**传统模型选择**，就是把一些传统的机器学习算法，比如SVM，KNN，决策树等等全拉过来，然后选择效果最好的。

**NAS**则是重点部分，本文将从两个方面介绍NAS：

- - 模型结构
  - 最优化生成出模型参数的方法

### 4.1 模型结构

NAS通过**选择和组合**一组在**搜索空间**中**预先定义的基本操作生成**神经网络模型。

这些操作可以大致分为**卷积、池化、连接、逐元素相加（elemental addition）、跳跃连接（skip connection）**等，逐元素相加和跳接最早可以追溯到ResNet。

这些操作的参数通常也是根据经验预先确定的。例如，卷积的核大小通常设置为3×3和5×5，也有一些人工设计的卷积核，如深度可分离卷积（depthwise separable convolution）、扩张卷积（dilation convolution也有叫空洞卷积的）、可变形卷积（deformable convolution）等。

而模型结构大体可以分为4类：**整体结构，基于cell的结构，层级结构，基于网络态射的结构**

#### 整体结构

最直观的办法就是生成整个神经网络，

> Neural architecture search with reinforcement learning.
> Efficient neural architecture search via parameter sharing

![img](https://pic4.zhimg.com/80/v2-3bf55dfd6c59f92ecce73944973cdc1f_720w.jpg)

作为一个整体生成的链结构神经结构的例子。图中的每个节点表示一个具有特定操作的层，例如N1表示层1。每个节点的操作从搜索空间中选择，包括卷积或最大值池化等。有向的边表示信息流。例如，左图中N2到N3的边表示N3接收N2的输出作为输入。

上图就是两个不同的网络，都是链结构的，右边的相对左边的就更复杂，因为右边有很多诸如跳接或者多分枝网络。

**整体结构的缺点:**当网络层数很深的时候，整体结构就需要很长时间去一点点搜索(因为整个网络的每个小部分都要单独搜索)。除此之外，这种耗时且精细的搜索就会导致网络结构的“过拟合”，即搜出的网络结构只能针对训练时的数据集，应对更大的数据集或者其他任务时就没有足够的可移植性。（这点很要命的，考虑到现在都是拿CIFAR10做搜索）

#### 基于cell的结构

为了解决上述的问题，受到ResNet和DenseNet的启发，

> Efficient neural architecture search via parameter sharing
> Learning transferable architectures for scalable image recognition
> Practical block-wise neural network architecture generation

等工作提出了基于cell的结构，基于cell的结构首先搜索cell的结构，然后再堆叠这些cell已生成最终的网络，这和之前提到的链式结构类似。

![v2-0b65c833d44f8be9f884ec92a28a83ce_720w](https://pic3.zhimg.com/80/v2-0b65c833d44f8be9f884ec92a28a83ce_720w.jpg)

这种设计方法可极大地降低搜索的复杂度，举个例子：

假设现在有6种预定义的基本操作，对第 ![[公式]](https://www.zhihu.com/equation?tex=j) 层 ![[公式]](https://www.zhihu.com/equation?tex=L_j) 来说，有 ![[公式]](https://www.zhihu.com/equation?tex=j-1) 层可以与其链接，于是就有 ![[公式]](https://www.zhihu.com/equation?tex=6%5Ctimes2%5E%7Bj-1%7D) 种可能的决策（6是这一层是啥，乘上的是和前面怎么连接）。可以看出这是一个等比数列，假设一共有12层，即 ![[公式]](https://www.zhihu.com/equation?tex=L%3D12+) ，对这个等比数列求和，就有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+6%5E%7BL%7D+%5Ctimes+2%5E%7BL%28L-1%29+%2F+2%7D%3D6%5E%7B8%7D+%5Ctimes+2%5E%7B12%2812-1%29+%2F+2%7D%3D1.6+%5Ctimes+10%5E%7B29%7D+%5Cend%7Bequation%7D) 种可能的神经网络。

而对于基于cell的网络结构来说，一个cell有 ![[公式]](https://www.zhihu.com/equation?tex=B) 个节点，一个节点由两个子节点(预定义操作)组成。由于**每个结点的决策是相互独立的**，节点有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%286+%5Ctimes%28B-2%29+%21%29%5E%7B2%7D+%5Cend%7Bequation%7D) 种可能。

除此之外，cell的功能也是有先验的，比如卷积 cell （convolution cell）和削减cell（reduction cell），输出会将空间分辨率除以2.

![v2-95d68876142afd71b1050b147d66ea49_720w](https://pic2.zhimg.com/80/v2-95d68876142afd71b1050b147d66ea49_720w.jpg)

除此之外，对于整体结构，每个层是一个操作。而对于基于cell的结构，每个层是很复杂的cell结构。也就是说，如果我们在一个小数据集上搜索出了一个比较小的网络，而需要迁移到更大的数据集上时，整体结构不好迁移，cell结构多怼几个cell就好了。

> Efficient neural architecture search via parameter sharing
> Learning transferable architectures for scalable image recognition
> Practical block-wise neural network architecture generation
> Designing neural network architectures using reinforcement learning,
> Large-scale evolution of image classifiers

回顾上边这几篇基于cell结构的论文，我们可以发现cell结构（逻辑上的）的两个层级：

- - cell层找每个node的操作和链接。
  - 网络层控制分辨率变化。

不过前期的基于cell的模型的效果有点弟弟，因为对网络层级的搜索不到位，就是无脑堆堆堆

就像上面figure7所示的一样。为了联合学习可重复的cell和网络的结构，Auto DeepLab定义了一个网络层级结构的泛化表示，如下figure8。

![v2-3cbb94fdf97ca57561fa26da212f2508_720w](https://pic1.zhimg.com/80/v2-3cbb94fdf97ca57561fa26da212f2508_720w.jpg)

#### 层级结构

层级结构和基于cell的结构很类似，区别在于生成cell的方法。基于cell的结构只有两个层级，即基本操作组成cell，cell拼成网络。而层级结构则有很多个层级，每个层级都可以由一些cell组成。高层的cell就由底层的cell迭代组合而成。

> Hierarchical representations for efficient architecture search

![v2-06b5060d458bd4d07bce42f10d8d4030_720w](https://pic1.zhimg.com/80/v2-06b5060d458bd4d07bce42f10d8d4030_720w.jpg)

如figure9所示，第一层级是基础的操作，比如1×1卷积，3×3卷积/最大值池化。

这些基础的组件组成了第二层级的cell。之后这些第二层的cell又作为基础操作生成第三层级的cell，以此类推……最高层级的cell就是整个结构的有个motif。

除此之外，一个高阶cell可以用一个可学习的邻接上三角矩阵G定义， ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bij%7D%3Dk) 表示第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个操作 ![[公式]](https://www.zhihu.com/equation?tex=O_k) 部署在了第 ![[公式]](https://www.zhihu.com/equation?tex=i) 和 ![[公式]](https://www.zhihu.com/equation?tex=j) 个节点之间。

和基于Cell的结构相比，层级结构更加灵活，可以学习更加复杂和多变的拓扑结构。

#### 基于网络态射（Network Morphism 可以认为是网络“形态”的“映射”（转变））的结构

基于Network Morphism(NM)的结构，可以将已经存在的神经网络结构中的信息迁移到一个新的神经网络，也就是说其可以对现在表现良好的网络进行更新，而不是从头搜索一个神经网络。当然，新网络的性能是要比旧网络好的，至少也不必旧的网络差。

> Net2net: Accelerating learning via knowledge transfer
> Network morphism
> Efficient multi-objective neural architecture search via lamarckian evolution.

![v2-4173273cec1536cbe4656f8275d0088c_720w](https://pic1.zhimg.com/80/v2-4173273cec1536cbe4656f8275d0088c_720w.jpg)

---

### 4.2 超参数优化（HyperParameter Optimization， HPO）

在定义了网络结构的表示方法之后，下一步就是从很大的搜索空间里搜索最优的网络结构。

这样的过程可以看作是对每个节点的操作和连接进行优化

我们可以认为，最终的网络结构也是一个“超参数”，而搜索网络结构和找学习率和batchsize一样，是超参数优化。



#### 网格/随机搜索

如果我们面前摆着很多超参数，最直观的方法就是把所有参数都拿过来试试。

不过遇到参数过多（连续）的情况，一个个试是不现实的，那么就把他们的取值范围列出来，间隔相同的长度画一个网格，之后尝试里面的每一个点，这就是网格搜索。

而随机搜索（论文名Random Search for Hyper-Paramerter Optimization）认为，不是所有的超参数都那么重要，以至于你需要一个个搜。因此不如在整个空间里随机取样。

![v2-253d667bc484cfab0e810bda8aee9e55_720w](https://pic2.zhimg.com/80/v2-253d667bc484cfab0e810bda8aee9e55_720w.jpg)

除此之外，为了找出超参数空间中表现良好的区域，

> A practical guide to support vector classification

建议在整个空间先做一个比较粗粒度的网格搜索，认为那些表现较好的格点的附近是更好的区域，之后再在这些区域做更细的网格搜索。

> Maximum-likelihood estimation with a contracting-grid search algorithm

提出了一个收缩网格搜索算法，其首先计算网格中每个点的似然值，之后一个新的网格就在最大似然值的中心生成。新网格中的点间距减少到旧网格的一半。对于固定次数的迭代重复此过程以收敛到局部最小值。

尽管随机搜索在经验上和理论上都比网格搜索好用，但是有个问题就是我们很难确定搜出来的点是不是效果最好的。只能认为搜的时间越长，越可能是最优解。

为了解决这个问题

> Hyperband: A novel bandit-based approach to hyperparameter optimization

提出了Hyperband算法，Hyperband在资源和性能之间做了权衡。具体做法是一次训一堆，给效果好的模型资源更多。



#### 强化学习

之前说过，NAS的老祖宗Neural architecture search with reinforcement Learning 用的就是RL，其通过RL训练了一个RNN生成网络的结构。

在此之后

> Designing neural network architectures using reinforcement learning

的**MetaQNN**提供了一个“元建模”算法（meta modeling algorithm），该算法使用Q-learning和ε-greedy探索策略和经验回放（experience replay），顺序地搜索神经结构。

总的来说，如之前的Figure1（下图）所示，基于RL的方法由两个部分组成：

- - **控制器**（如RNN），用于生成子网络
  - **奖励网络**（Reward Network），用于评价生成的子网络。并根据子网络的“奖励”（如准确率）更新控制器的参数。

![v2-eb45110765771cbbcb7eac7621ab521b_720w](https://pic4.zhimg.com/80/v2-eb45110765771cbbcb7eac7621ab521b_720w.jpg)

基于RNN和MetaQNN的两个工作都在宾州树库和CIFAR10数据集上达到了SOTA。但他们共同的问题就是太慢，太消耗资源了，大的令人发指（即使较低的MetaQNN也是100GPU days量级）

由于上述两个算法使用的是整体结构，而整体结构的搜索和训练都相当耗时，所以这两个基于RL的算法都不是那么高效。为了进一步提升效率，很多基于RL的算法改用了基于Cell的结构。包括（括号内为模型名）

> （NASNet）Learning transferable architectures for scalable image recognition.
> （BlockQNN）Practical block-wise neural network architecture generation.
> （ENAS）Efficient neural architecture search via parameter sharing,

其中ENAS是基于cell结构的强者，用一块GPU只要十小时就可以搜索出结果了。

ENAS的创新点在于，其每个子网络看作搜索空间的子图，如此一来子模型共享参数，无需从头训练。



#### 演化算法

演化算法是一种基于种群的泛启发式优化算法。据说这是一种对各种问题都比较鲁棒的算法。

基于演化算法的网络编码表达不一样，所以对应工作的基因操作也不一样。具体有两种编码方式：

**1. 直接编码**

直接编码是一种广泛使用的方法，它显式地指定了网络表现型（phenotype）。

比如，

> Genetic CNN

使用了一种二进制编码表示网络结构，比如“1”表示两个节点是相连的。这种二值编码的优势是易于执行，但计算的空间是节点的平方。更恐怖的是，节点的数目在一开始就是定死的。

> A genetic programming approach to designing convolutional neural network architectures.

使用了笛卡尔遗传规划（Cartesian Genetic Programming, CGP），把网络描述为非循环图.有关CGP的知识可以参考

> Cartesian genetic programming
> Redundancy and computational efficiency in cartesian genetic programming

而下文，

> Large-scale evolution of image classifiers.

就是把结构用图编码，点是3阶的张量or激活值，边是恒等映射or卷积等操作。

> Neuroevolution of Augmenting Topologies（NEAT）

也是一种直接编码结构，把所有节点和边编码到了DNA里。

**2. 间接编码**

间接编码规定生成规则构建网络，允许网络结构的紧凑表示。

Cellular Encoding（CE）将一系列神经网络编码成一组基于简单图语法的标记树。原论文：

> Cellular encoding as a graph grammar,

而最近四篇工作

> Efficient multi-objective neural architecture search via lamarckian evolution，
> Convolution by evolution: Differentiable pattern producing networks,
> Deep clustered convolutional kernels,
> Evolving multimodal controllers withhyperneat

也使用了间接编码，比如上述的第一篇，就使用了function（个人认为这里是“功能”）进行编码。每个网络用功能保留网络表型算子编码，保证子网络不比亲代网络差。

一个基本的演化算法包括以下四步：**选择，交叉，突变，更新**。

![v2-e7359ffc15a89903ee6d9d6f92615e68_720w](https://pic1.zhimg.com/80/v2-e7359ffc15a89903ee6d9d6f92615e68_720w.jpg)

### STEP 1 选择（Selection）：

这一步是从所有生成的网络中选择一部分进行交叉。有三种选择单个网络的策略：

-  **Fitness Selection**：所有子网络被选择的概率是适应度(Fitness Value)在所有群体适应度之和的比例。即

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+P%5Cleft%28h_%7Bi%7D%5Cright%29%3D%5Cfrac%7B%5Ctext+%7BFitness%7D%5Cleft%28h_%7Bi%7D%5Cright%29%7D%7B%5Csum_%7Bj%3D1%7D%5E%7BN%7D+%5Ctext+%7BFitness%7D%5Cleft%28h_%7Bj%7D%5Cright%29%7D+%5Cend%7Bequation%7D)

- - **Rank Selection**：和第一种类似，但概率和相对排名有关，而不是绝对的值

- - **Tournament Selection**：是基于EA（演化算法）的**NAS中最常用的选择策略**。在每一轮中，首先在所有子网络中随机选择k（锦标赛大小）个个体。之后最优的个体就有概率 p 被选中，而对第二名的个体，概率就是 p×(1-p)，以此类推……以下是使用Tournament Selection的几篇工作：

> Hierarchical representations for efficient architecture search
> Large-scale evolution of image classifiers
> Regularized evolution for image classifier architecture search
> Efficient multi-objective neural architecture search via lamarckian evolution

### STEP2 交叉（Crossover）：

经过选择的每两个个体都会生成一个新的后代，后代继承了部分亲代的基因信息。这就模拟了生物学中交配的过程。而交叉的方法也根据编码的不同而不同。

对于二进制编码，网络被编码为了一个线性的bit串，因此两个亲代可以通过一点或多点交叉来结合。不过有的时候这种一个个点变动的行为会损害有用的信息。因此Genetic CNN,交叉的单位是一个由部分bit串组成的stage。

而对于**Cellular Encoding（CE）**，一个随机选择的子树从一个父树中被切割，并替换另一个父树的子树。

**NEAT**，执行基于历史标记的人工配对，允许它添加新的结构而不丢失哪个基因是哪个的整个模拟。

### STEP3 突变（Mutation）

当基因信息被遗传给下一代时，基因突变也会发生。下面是三种不同的突变形式

1. 1. **点突变**：随机翻转每一个bit

> A genetic programming approach to designing convolutional neural network architectures
> Genetic CNN

​	    2. **连接突变**：下文有两个突变方法

- 改变两层之间是否连接
- 在两个节点/层之间添加/删除跳接。

> Evolving deep neural networks

 	   3. **自定义突变**：

> Large-scale evolution of image classifiers

预定义了一组突变操作子，包括改变学习率和滤波器的size，移除跳接等等



虽然突变的过程可能看起来像一个错误，会导致网络结构的破坏和功能的丢失，但是探索更多的新结构和确保多样性。

### STEP4 更新（Update）

上述步骤完成后，会生成许多新的网络。通常情况下，由于计算资源有限，需要删除一些网络。

> Large-scale evolution of image classifiers

随机选出两个子网络，其中**最差**的那一个立即从种群中剔除

> Regularized evolution for image classifier architecture search

移除的是**最老**的。

下面的三个工作，每隔一定时间删除所有网络

> A genetic programming approach to designing convolutional neural network architectures.”
> Evolving deep neural networks.
> Genetic CNN



而下文

> Hierarchical representations for efficient architecture search

谁都不删，整个种群随着时间推移慢慢生长。



#### 贝叶斯优化（Bayesian Optimization，BO）

BO建立目标函数概率模型，然后利用该模型选择最有希望的超参数，最后根据真实目标函数对选择的超参数进行评价。

这样做的优势在于，每一轮的网络信息都被记录在了整个概率空间当中，这样即使之前的网络被弃用，它的信息也依然通过更新概率分布得到保留。

而且，直接建立超参数到目标分数的数学映射是很帅的一件事儿。

序列模型优化（Sequential model-based optimization，SMBO）是贝叶斯优化的一种简洁形式。下面是SMBO的伪代码流程。

![v2-72412a4e8217a0ce3192eece02c44ec3_720w](https://pic4.zhimg.com/80/v2-72412a4e8217a0ce3192eece02c44ec3_720w.jpg)

1. 1. 从搜索空间 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cmathcal%7BX%7D+%5Cend%7Bequation%7D) 里取一小部分样本随机初始化。
   2. 其中， ![[公式]](https://www.zhihu.com/equation?tex=D) 是一个由数据对 ![[公式]](https://www.zhihu.com/equation?tex=%28x_i%2Cy_i%29) 组成的数据集
   3. ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+y_%7Bi%7D%3Df%5Cleft%28x_%7Bi%7D%5Cright%29+%5Cend%7Bequation%7D) 的计算过程十分难搞（比如输入 ![[公式]](https://www.zhihu.com/equation?tex=x) 是网络结构，输出 ![[公式]](https://www.zhihu.com/equation?tex=y) 是这个网络最终的准确率）
   4. 而模型 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cmathcal%7BM%7D+%5Cend%7Bequation%7D) 则负责拟合这个数据集 ![[公式]](https://www.zhihu.com/equation?tex=D)
   5. 于是呢，就生成了一堆新的超参数，这些超参数服从模型![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cmathcal%7BM%7D+%5Cend%7Bequation%7D)的分布，并且被一个预定义的采集函数（acquisition function） ![[公式]](https://www.zhihu.com/equation?tex=S) 顺序选择。而这个采集函数 ![[公式]](https://www.zhihu.com/equation?tex=S) 可以看作难搞的目标函数 ![[公式]](https://www.zhihu.com/equation?tex=f) 的廉价近似

顺带一提，有4种不同的**采集函数**

1. 1. 基于改进的（improvement-based policies）策略
   2. 乐观策略
   3. 基于信息的策略
   4. 组合采集函数（portfolios of acquisition functions）

而根据**概率模型**的不同，贝叶斯最优化（BO）可以被分为以下三种：

- **高斯过程（Gaussian Processes，GPs）**

> Practical bayesian optimization of machine learning algorithms

- **树Paezen估计量（Tree Parzen Estimators，TPE）**

> Making a science of model search: Hyperparameter optimization in hundreds of dimensions for vision architectures

- **随机森林（Random Forests）**

> Sequential model-based optimization for general algorithm configuration,

![v2-528e5673244604d3f47126712637b544_720w](https://pic1.zhimg.com/80/v2-528e5673244604d3f47126712637b544_720w.jpg)

上表是一些BO算法的开源库，可以看出基于高斯过程的BO是最有名的。

即使目标函数是随机的、非凸的、甚至非连续的，贝叶斯优化算法也是有效的，但是呢碰到深度学习问题，BO可能就有点不行了。

除此之外，虽然Hyperband（见网格/随机搜索的最后）这类本身就做了资源性能权衡的模型可以有限资源内收敛，但我们并不知道这些收敛结果到底是不是最优的。

为了解决上述问题，

> BOHB: Robust and efficient hyperparameter optimization at scale

提出了**Bayesian Optimization-based Hyperband (BOHB)**。BOHB算法在SVM，神经网络和深度强化学习都可以使用。

而一个更快的BO算法，名为**FABOLAS**，它是之前BOHB的快速版本，大概能快10到100倍。



#### 梯度下降法

尽管之前所有的搜索方法都可以自动生成网络结构，也都可以达到SOTA的水平，但是他们还是太慢了。因为之前的方法大多使用离散的方法搜索网络，而且把整个HPO（超参数优化）问题看作一个黑盒问题，因此需要多次消耗大量的计算资源。

> DARTS: Differentiable architecture search.
> Convolutional neural fabrics
> Connectivity learning in multi-branch networks
> Differentiable neural network architecture search,
> Gradient-based hyperparameter optimization through reversible learning,
> Hyperparameter optimization with approximate gradient

等文章就尝试解决离散和黑盒优化问题。特别是DARTS,它在搜索空间内做的是连续且可微分的搜索，这是由一个softmax函数软化离散的超参数超参数决策实现的：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbar%7Bo%7D%5E%7B%28i%2C+j%29%7D%28x%29%3D%5Csum_%7Bo+%5Cin+%5Cmathcal%7BO%7D%7D+%5Cfrac%7B%5Cexp+%5Cleft%28%5Calpha_%7Bo%7D%5E%7B%28i%2C+j%29%7D%5Cright%29%7D%7B%5Csum_%7Bo%5E%7B%5Cprime%7D+%5Cin+%5Cmathcal%7BO%7D%7D+%5Cexp+%5Cleft%28%5Calpha_%7Bo%5E%7B%5Cprime%7D%7D%5E%7B%28i%2C+j%29%7D%5Cright%29%7D+o%28x%29+%5Cend%7Bequation%7D)

其中 ![[公式]](https://www.zhihu.com/equation?tex=o%28x%29) 表示的是对输入 ![[公式]](https://www.zhihu.com/equation?tex=x) 的操作，比如卷积和池化。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BO%7D) 则是预定义的操作集。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Calpha_%7Bo%7D%5E%7B%28i%2C+j%29%7D+%5Cend%7Bequation%7D) 表示操作 ![[公式]](https://www.zhihu.com/equation?tex=o) 在一个节点对 ![[公式]](https://www.zhihu.com/equation?tex=%28i%2Cj%29) 之间，比如 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+x_%7Bj%7D%3D%5Calpha_%7Bo%7D%5E%7B%28i%2C+j%29%7D+o%5Cleft%28x_%7Bi%7D%5Cright%29+%5Cend%7Bequation%7D)

![v2-ab9edd28ef548d30c96492723df897ef_720w](https://pic4.zhimg.com/80/v2-ab9edd28ef548d30c96492723df897ef_720w.jpg)

![v2-56ed107ae57a4685f7afc2d39ce4a266_720w](https://pic3.zhimg.com/80/v2-56ed107ae57a4685f7afc2d39ce4a266_720w.jpg)

基于梯度下降的办法节省时间，但搜索空间和GPU占用呈现线性关系，搜索空间变大则GPU占用必然变大。再上图Figure13中，边上共有三种不同的操作，为了提高效率，只对最高权重的操作会被重新训练，

为了节约资源消耗，

> PROXYLESSNAS: DIRECT NEURAL ARCHITECTURE SEARCH ON TARGET TASK AND HARDWARE

提出了一种新的**通道层级剪枝方法（path-level pruning method）**，一开始设置一个大网络，逐步去掉没用的部分。（不过这个大网络也占地方）

> sharpDARTS: Faster and More Accurate Differentiable Architecture Search

则解决了大网络的问题。其中采用了一个通用的、平衡的、一致的设计，即**SharpSepConv块**。

他们也提出了**可微分超参网格搜索（Differentiable Hyperparameters Grid Search）**和**超立方搜索空间（HyperCuboid search space）**。相较于DARTS快了50%，且精度也有较大提升。

---

## 五、模型评价

一旦我们生成了一个网络，那么就要评价这个网络好不好。

一个直观的方法就是把这个网络拉出来训一训，从最终的训练结果确定可以不可以拿来用。但这种方法也是最花费时间和计算资源的。

为了加速模型评价的过程，研究者们采取了如下几种办法：

### **5.1 低保真**

训练时间与数据集大小和模型大小相关，所以可以从这两个方向入手。

一方面，我们可以减少数据集中图片的数量以及图片的分辨率（那个快速贝叶斯优化**FABLOAS**用的就是这种办法）

另一方面，可以减少模型的大小，比如每层使用更少的滤波器（减少每个卷积层输入输出通道数）

> Learning transferable architectures for scalable image recognition
> Regularized evolution for image classifier architecture search.

两个工作就用了这种办法。

> Multi-Fidelity Automatic Hyper-Parameter Tuning via Transfer Series Expansion

则采用了类似于集成学习的方法，提出了**Transfer Series Expansion（TSE）**，简单的说就是对一个子网络看**多种低保真度策略**(比如不同的分辨率)，避免某个低保真度的模型效果特别差导致的误杀。

> Towards automated deep learning: Efficient joint neural architecture and hyperparameter search

这篇工作则验证了，只要训练时间足够长，那么就不需要训练太长时间

![v2-2119a44bd499ba95ba23114f316bbae8_720w](https://pic1.zhimg.com/80/v2-2119a44bd499ba95ba23114f316bbae8_720w.jpg)

### **5.2 迁移学习（Transfer Learning）**

老祖宗论文（RNN+RL的那篇论文）每次训完的网络就直接扔掉了，新的网络要从头训练，才花费了这么多资源。因此，如果使用迁移学习的技术，把之前训练过的网络用起来也能加速搜索过程。

> Transfer learning with neural automl,

就是用之前任务的知识加速网络设计。

而**ENAS**则在子网络之间共享权重，比之前的老祖宗快1000倍。

而一些基于网络态射的算法也继承了之前架构的权重，比如↓

> Net2net: Accelerating learning via knowledge transfer
> Network morphism

### **5.3 代理人（Surrogate）**

基于代理人的方法是另一个近似黑盒函数的有效工具，以下三篇是代理人方法的有关论文。

> Surrogate benchmarks for hyperparameter optimization.
> An evaluation of adaptive surrogate modeling based optimization with two benchmark problems
> Efficient benchmarking of hyperparameter optimizers via surrogates,

总的来说，如果我们能找到个不错的近似，那么**先近似后优化**比从头优化快得多。

**PNAS**（原文Progressive neural architecture search）用了代理人模型控制搜索方式，比ENAS能看更多模型。

不过，如果优化的空间过大，代理人模型就遭不住了。

### **5.4 早停**

早停本来是一个解决过拟合的办法，却也可以用来减少学习时间。

早停法最直观的问题是，要停多“早”？如何科学“早停”？

比如可以提前通过停止预测在验证集上表现不佳的评估模型来加快速度。

> Learning curve prediction with bayesian neural networks
> Peephole: Predicting network performance before training
> Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves,

上面三篇就是解决科学早停问题的，而且看起来十分玄学。

比如最后一篇，它提出了一种学习曲线模型，该模型是从文献中选取的一组参数曲线模型的加权组合，拿这个组合预测最终网络的效果。

在这种玄学论文的基础上，

> Early stopping without a validation set,

提出一种基于快速计算梯度的局部统计数据的早停方法，它不再像以前的方法那样依赖于验证集，而是允许优化器充分利用所有的训练数据。

---

## 六、NAS性能总结

NAS是最近很火的研究主题，而且涉及到非常多的算法（之前提到的那些，五花八门）。

为了帮助大家更好地归纳理解NAS，我们将热门的NAS算法分为了四类：

- - 随机搜索（RS）
  - 强化学习（RL）
  - 演化算法（EA）
  - 梯度下降（GD）

之后必须注意，这些模型报告的结果直接比较会非常不公平。就拿搜索出来网络的效果，比如CIFAR10分类的准确度举例子，这些模型的效果其实非常接近，而且也都算是SOTA了。而且我们也知道，不同的数据增广等等Trick也会影响模型的最终效果，因此，准确率这个参数看看就好。

还有花费时间举例，一百个GPU跑一天显然和一个GPU跑一天不一样，所以这里引入 GPU Days去衡量不同算法的准确率，假设由N个GPU，跑起来花了D天，那么GPU Days = N×D

GPU Days也只是个相对精准的量，因为不同模型部署的GPU型号也不一样。

![v2-ee941fa5626a597268bea74cc0e36d8a_720w](https://pic3.zhimg.com/80/v2-ee941fa5626a597268bea74cc0e36d8a_720w.jpg)

![v2-bfd3d2fbaaa983dcae34d4df77ca637e_720w](https://pic3.zhimg.com/80/v2-bfd3d2fbaaa983dcae34d4df77ca637e_720w.jpg)

![v2-5c165eeb1cc3068d7018f25b5ff53f6b_720w](https://pic4.zhimg.com/80/v2-5c165eeb1cc3068d7018f25b5ff53f6b_720w.jpg)

最终我们整理出了不同NAS算法的图表如上所示。可以得出的结论是：

- - 实际上CIFAR10这种数据集上效果很接近，谁也干不过谁。
  - 基于梯度下降的方法是最快的
  - 基于EA的普遍耗时，因为评估所有子网络太难了。
  - 比较有趣的是，就是随机搜索也可以得到不错的结果。

---

## 七、开放问题与未来工作

### **7.1 完整的AutoML管道**

尽管现在已经有了很多开源的AutoML管道，比如TPOT和Auto-Sklearn

> TPOT👉 Automating Biomedical Data Science Through Tree-based Pipeline Optimization
> Auto-Sklearn👉Efficient and robust automated machine learning

不过上述的工作也不能自动收集数据，也囿于计算资源不能做什么自动化的特征工程。

不过长远来看，我们的终极目标就是将机器学习的每个流程全都自动化，并将其放入一个完整的系统。

![v2-1bef1b6dc7869e17ceeaacc807d6b18b_720w](https://pic4.zhimg.com/80/v2-1bef1b6dc7869e17ceeaacc807d6b18b_720w.jpg)

### **7.2 可解释性（Interpretability）**

尽管NAS总是可以找到比手工设计效果好的模型，但是现在也没有人科学/正式地证明为什么在某个地方选择某个操作会比其他的好很多。

比如BlockQNN在最后采用的是Concat操作而不是逐元素加，5×5的卷积也相比其他更少见。但是人们对这些结果的解释都是主观和直觉性的。

因此AutoML的可解释性也是一个重要的研究方向。

### **7.3 可复现性（Reproducibility）**

可复现性一直是ML领域的老大难问题，AutoML也不例外。

特别是NAS的研究，因为很多算法的源码无法下载，或者即使提供了源码，结果仍然不可复现（或者需要很长的时间搜索）。

而NASBench（NAS-Bench-101: Towards Reproducible Neural Architecture Search）缓解了不可复现性的问题，其提供了一个表格，里面有423624种神经网络在标准数据集上（CIFAR10）的结果。

### **7.4 灵活的编码方案（Flexible Encoding Scheme）**

当前的搜索空间碍于人们对神经网络编码的预定义，会遇到以下三个缺点：

1. 1. 不同工作对网络的编码不能通用
   2. 搜索空间的原子是预定义的操作层（卷积，池化），则无法出现崭新的操作层（比如人们提出空洞卷积和深度可分离卷积）
   3. 搜索空间上网络结构也有人为的先验，很新颖的模型结构也无法发现（跳接，Concat等结构）。

### 7.5 其他领域

现在NAS的研究到部分都在CIFAR10上。当然，现在已经有更多的工作去研究NAS+目标检测和语义分割了

> Nas-fpn: Learning scalable feature pyramid architecture for object detection,
> Auto-deeplab:Hierarchical neural architecture search for semantic image segmentation
> Nas-unet: Neural architecture search for medical image segmentation,

探索NAS在其他领域上的应用也是可行的研究方向。

### **7.6 终生学习（Lifelong Learn）**

一旦一个高质量的AutoML系统被实现，那么终生学习就不远了。

首先，这个系统应该可以有办法运用之前的先验知识解决新的问题（Learn to learn），以便将其运用在更多的跨学科领域中。

而最近已经有一些工作使用元学习技术去加速NAS了

> SMASH: ONE-SHOT MODEL ARCHITECTURE SEARCH THROUGH HYPER-NETWORKS

另一方面，如果我们可以自动地持续获取数据，那么模型就可以进行持续地增量学习（Incremental learning）。

这个时候，在接受新数据同时，避免旧数据灾难性遗忘也需要研究。

LWF（Learning without forgetting）以及其基础上的ICARL（ICARL：Incremental classifer and representation learning）就是解决灾难性遗忘问题的。