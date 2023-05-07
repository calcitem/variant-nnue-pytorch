# NNUE

## Preface

This document describes in detail what NNUE is, how it works in theory, how the inference is implemented and how to make it efficient, how to train it with pytorch, and describes some architectural enhancements possible.

## Table of contents

* [Preface](#preface)
* [Table of contents](#table-of-contents)
* [Basics](#basics)
    + [What is NNUE?](#what-is-nnue)
    + [Quantization 101](#quantization-101)
    + [What layers are used in NNUE?](#what-layers-are-used-in-nnue)
        - [Linear layer](#linear-layer)
        - [Linear layer with sparse inputs](#linear-layer-with-sparse-inputs)
        - [Clipped ReLU layer](#clipped-relu-layer)
    + [A simple input feature set.](#a-simple-input-feature-set)
    + [A simple NNUE network](#a-simple-nnue-network)
    + [Accumulator](#accumulator)
    + [HalfKP](#halfkp)
        - [Multiple perspectives, multiple accumulators](#multiple-perspectives-multiple-accumulators)
            * [How to combine them?](#how-to-combine-them)
            * [Which weights to use?](#which-weights-to-use)
        - [HalfKP example and network diagram](#halfkp-example-and-network-diagram)
* [Forward pass implementation](#forward-pass-implementation)
    + [Example network](#example-network)
    + [Layer parameters](#layer-parameters)
    + [Accumulator](#accumulator-1)
        - [Refreshing the accumulator](#refreshing-the-accumulator)
        - [Updating the accumulator](#updating-the-accumulator)
    + [Linear layer](#linear-layer-1)
    + [ClippedReLu](#clippedrelu)
    + [Putting it together](#putting-it-together)
    + [Consideration of networks size and cost.](#consideration-of-networks-size-and-cost)
        - [Feature set](#feature-set)
        - [First set of hidden neurons](#first-set-of-hidden-neurons)
        - [Further layers](#further-layers)
* [Training a net with pytorch](#training-a-net-with-pytorch)
    + [Model specification](#model-specification)
    + [Preparing the inputs](#preparing-the-inputs)
        - [Parsing the training data sets and moving them to the python side](#parsing-the-training-data-sets-and-moving-them-to-the-python-side)
        - [But what is actually sent, how, and how it's made into tensors?](#but-what-is-actually-sent-how-and-how-its-made-into-tensors)
    + [Feature factorization](#feature-factorization)
        - [Virtual feature coalescing](#virtual-feature-coalescing)
        - [Other factors](#other-factors)
            * ["K" factors](#k-factors)
            * ["HalfRelativeKP" factors](#halfrelativekp-factors)
        - [Real effect of the factorizer](#real-effect-of-the-factorizer)
    + [Loss functions and how to apply them](#loss-functions-and-how-to-apply-them)
        - [The Goal](#the-goal)
        - [Converting the evaluation from CP-space to WDL-space](#converting-the-evaluation-from-cp-space-to-wdl-space)
        - [Using results along the evaluation](#using-results-along-the-evaluation)
        - [Mean Squared Error (MSE)](#mean-squared-error-mse)
            * [loss](#loss)
            * [grad](#grad)
        - [Cross entropy](#cross-entropy)
            * [loss](#loss-1)
            * [grad](#grad-1)
* [Quantization](#quantization)
    + [Stockfish quantization scheme](#stockfish-quantization-scheme)
        - [Feature Transformer](#feature-transformer)
        - [Linear layer](#linear-layer-2)
        - [ClippedReLU](#clippedrelu-1)
    + [The math of quantization and how to make it fit](#the-math-of-quantization-and-how-to-make-it-fit)
        - [Feature Transformer](#feature-transformer-1)
        - [Linear layer](#linear-layer-3)
    + [Implementation](#implementation)
    + [Optimized implementation](#optimized-implementation)
        - [Feature Transformer](#feature-transformer-2)
        - [Linear layer](#linear-layer-4)
            * [m256_add_dpbusd_epi32](#m256_add_dpbusd_epi32)
            * [m256_haddx4](#m256_haddx4)
        - [Linear layer with sparse input](#linear-layer-with-sparse-input)
        - [Linear layer with sparse input and blocked sparse output](#linear-layer-with-sparse-input-and-blocked-sparse-output)
        - [ClippedReLU](#clippedrelu-2)
            * [int16 -> int8](#int16---int8)
            * [int32 -> int8](#int32---int8)
    + [Accounting for quantization in the trainer](#accounting-for-quantization-in-the-trainer)
        - [Inside the optimizer](#inside-the-optimizer)
        - [Outside the optimizer](#outside-the-optimizer)
        - [Accounting for virtual layers (factorization)](#accounting-for-virtual-layers-factorization)
* [Optimizing the trainer (CUDA)](#optimizing-the-trainer-cuda)
    + [Using custom CUDA kernels](#using-custom-cuda-kernels)
    + [Feature transformer](#feature-transformer)
        - [Data loader](#data-loader)
        - [Forward](#forward)
        - [Backward](#backward)
        - [FeatureTransformerSlice layer](#featuretransformerslice-layer)
        - [Results](#results)
* [Architectures and new directions](#architectures-and-new-directions)
    + [Simple HalfKP Stockfish architecture](#simple-halfkp-stockfish-architecture)
    + [HalfKAv2 feature set.](#halfkav2-feature-set)
    + [A part of the feature transformer directly forwarded to the output.](#a-part-of-the-feature-transformer-directly-forwarded-to-the-output)
    + [Multiple PSQT outputs and multiple subnetworks](#multiple-psqt-outputs-and-multiple-subnetworks)

## 基础知识

### 什么是NNUE？

NNUE（ƎUИИ Efficiently Updatable Neural Network，高效可更新神经网络）广义上是一种神经网络结构，它可以利用在连续评估之间网络输入的最小变化。它最初是由Yu Nasu为将棋发明的，后来被Motohiro Isozaki在2018年5月整合到[YaneuraOu](https://github.com/yaneurao/YaneuraOu)，并在2019年6月被Hisayori Noda移植到国际象棋的Stockfish引擎中，但它也适用于许多其他棋盘游戏，甚至可能适用于其他领域。NNUE遵循以下原则：

1. 网络应具有相对较低数量的非零输入。
2. 连续评估之间的输入变化应尽可能小。

原则1意味着当网络扩展大小时，输入必须变得稀疏。目前最优的结构具有大约0.1%的输入稀疏度。非零输入数量较少，意味着在必须对整个网络进行评估的情况下，评估网络所需的时间上限较低。这是NNUE网络可以在保持较大规模的同时，仍具有非常快速评估速度的主要原因。

原则2意味着在大多数回合制游戏中，它可以利用单次移动只会稍微改变棋盘状态的事实。这个原则相对于第一个原则的重要性较低，对于实现来说完全是可选的，但是在确实利用这个假设的实现中，它仍然能带来可衡量的改进。

总的来说，NNUE原则也适用于昂贵的深度网络，但它们在快速的浅层网络中表现得更好，这些网络适合用于低延迟的CPU推理，而无需进行批处理。目标性能是每线程每秒百万（或更多）次评估。这是一个极端的用例，需要极端的解决方案。

### 量化基础知识

量化是将神经网络的领域从浮点值更改为整数值的过程。NNUE网络旨在实现快速评估，因此它们充分利用了可用的int8性能。在必要时，还会添加一些int16和int32。浮点运算对于实现最大引擎强度来说并不是一个选项，因为它为了获得较小的精度提升而牺牲了太多速度，尽管它被其他一些方法所使用。这限制了网络可表示值的范围，并引入了误差，防止网络变得过于复杂，否则误差会累积过多。量化将在本文档后面的部分更详细地描述。在此之前，本文档将使用浮点数代替整数，在我们进行实际代码优化之前，这个问题并不重要。

### NNUE中使用了哪些层？

NNUE目前依赖于线性（全连接）和ClippedReLU（clamp(0, 1)）层。

通常，这种网络保持较浅的层数（2-4层），因为量化会引入一些误差，并且隐藏神经元的数量较少。相较于后面的层，第一层通常要大几个数量级，简单地说，这是因为知识必须存储在*某个地方*，而且它是最佳候选者，因为输入可以变得稀疏。

#### 线性层

线性层（全连接层）只是一个简单的矩阵乘法。它可以高效实现，支持稀疏输入，并提供良好的容量。它以`in_features`个值作为输入，并生成`out_features`个值。操作为`y = Ax+b`，其中：

`x` - 大小为`in_features`的输入列向量

`A` - 大小为`(out_features, in_features)`的权重矩阵

`b` - 大小为`out_features`的偏置列向量

`y` - 大小为`out_features`的输出列向量

![矩阵向量乘法](img/mv.png)

#### 具有稀疏输入的线性层

乘法`Ax`可以在概念上简化为"如果`x[i]`不为零，则从`A`中取列`i`，将其乘以`x[i]`并将其添加到结果中"。现在应该很明显，每当输入元素为零时，我们可以跳过处理权重矩阵的整行。这意味着我们只需要处理`A`的列数与输入向量中非零值的数量一样多。尽管权重矩阵中可能有数万个列，但对于每个位置，我们只关心其中的一部分！这就是为什么第一层可以如此庞大。

![Matrix and sparse vector multiplication](img/mvs.png)

#### 截断ReLU层

这是基于常规ReLU的激活函数，区别在于它的上下限都有边界。公式为`y = min(max(x, 0), 1)`。

![截断ReLU](img/clipped_relu.png)

这一层的目的是为网络增加非线性。如果仅仅是线性层，它们都可以合并为一个，因为矩阵可以直接相乘。

理想情况下，截断ReLU将被替换为ReLU，但激进的量化需要减小隐藏层输入的动态范围，因此将值限制在1变得对性能非常重要。

### 简单的输入特征集

大多数成功的引擎使用所谓的"HalfKP"特征或其变体。这将在后面详细描述，但现在让我们关注一个更简单的例子，以便基本了解NNUE实际如何工作。

为了说明，我们将考虑一组基于棋子位置的简单输入。我们将其称为"A"特征，因为它们将代表"所有棋子"，与它们所在的方格以外的任何其他事物无关。

棋盘上有64个方格，6种棋子类型（兵，马，象，车，后，王）和2种颜色（白色，黑色）。我们想要将棋子位置编码为输入，所以每个输入将对应于某个（方格，棋子类型，颜色）元组。共有`64*6*2=768`个这样的元组。如果有颜色为`C`的棋子`P`在方格`S`上，我们将输入`(S, P, C)`设为1，否则设为0。尽管输入的总数为768，但在任何合法的国际象棋局面中，最多只能有32个非零输入，因为棋盘上最多只有32个棋子。此外，任何走法最多只能改变4个输入（王车易位），平均应该低于3。

在将特征传递给神经网络时，利用了输入的二进制和稀疏特性——输入只是特征（索引）的列表，不需要完整的输入向量，因为其他位置的值为0，我们知道每个激活特征都有一个与之关联的值1。

让我们看一个示例局面`1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1`。

![](img/board_0.png)

在上面的棋盘上，我们有4个活动特征：`(A1, king, white)`，`(C3, pawn, white)`，`(B8, king, black)`，`(D4, rook, black)`。

现在让我们考虑走棋c4 - 唯一失效的特征是`(C3, pawn, white)`，它需要被替换为`(C4, pawn, white)`。

现在让我们考虑走棋cxd4 - 象前面一样，兵移动了，所以我们移除`(C3, pawn, white)`并添加`(D4, pawn, white)`。但是车也被从棋盘上移除了，所以我们还要移除`(D4, rook, black)`。这仍然比从头开始重建输入要少！

### 简单的NNUE网络

我们将使用前一段中的"A"特征集，所以我们有768个输入。为了说明，我们将使用3个线性层，分别是768->8，8->8，8->1。所有层都是线性的，所有隐藏神经元都使用截断ReLU激活函数。下面的图片展示了该架构：

![A[768]->8->8->1 架构图](img/A-768-8-8-1.png)

流程从左到右。第一层是一个具有768个输入的大型全连接层，但是对于每个位置，只有一小部分输入是非零的 - 可以利用稀疏矩阵向量乘法。隐藏层要小得多，始终使用密集矩阵向量乘法进行计算。最后，我们得到1个输出，通常训练成为位置的厘米评估值。

### 累加器

尽管我们观察到从一个位置到另一个位置的输入变化很少，但我们尚未利用这一点。回想一下，线性层只是将一些权重矩阵列相加在一起。我们可以将第一组隐藏神经元作为位置的状态的一部分保留起来，然后根据添加或删除的特征（列）在每个走棋时更新它，而不是为每个位置重新计算第一组隐藏神经元！我们只需要处理两个简单的情况：

1. 特征`i`从输入中删除（1 -> 0） - 从累加器中减去权重矩阵的第`i`列
2. 特征`i`添加到输入中（0 -> 1） - 将权重矩阵的第`i`列添加到累加器中

对于单一走棋，找到哪些"A"特征发生了变化是非常简单的 - 我们知道我们正在移动的棋子，从哪里移动，以及移动到哪里。捕获和升变可以被视为棋子消失或从无处出现。

然而，在使用浮点值时必须小心。反复添加和减去浮点数会导致每次移动累积的误差。需要仔细评估误差是否足够小，以使网络仍能产生良好的结果。值得庆幸的是，最好的实现方法是在撤销移动时不更新累加器。相反，它简单地存储在堆栈上，因此误差受到`O(MAX_DEPTH)`的限制，可以基本忽略。

在使用量化时，这不再是一个问题，但现在有可能累加器溢出。然而，这个问题要小得多，即使没有这些增量更新，这个问题也会出现。量化必须确保没有任何可能的活动特征组合能超过最大值。

总之，NNUE是一种高效的神经网络架构，它利用了输入之间的稀疏性和相邻评估之间的最小变化。通过使用线性层和截断ReLU激活函数，网络能够快速评估棋盘位置。为了进一步提高性能，NNUE网络使用了累加器来在每一步中更新隐藏神经元的状态，从而避免了在每个位置重新计算这些神经元的开销。虽然在处理浮点数时需要小心，但使用量化技术可以有效地解决这个问题。这种架构使得NNUE网络在低延迟CPU推断中表现出色，无需批处理，适用于许多棋盘游戏和其他领域。

### HalfKP

HalfKP是最常见的特征集，其他成功的特征集都是基于它构建的。它恰好处在一个合适的大小，而且平均每步需要更新的特征非常少。每个特征是一个元组`(our_king_square, piece_square, piece_type, piece_color)`，其中`piece_type`不是国王（在HalfKA特征集中，国王被包括在内）。这意味着对于每个国王位置，都有一组特征`P`，即`(piece_square, piece_type, piece_color)`。这使得网络能够更好地理解与国王相关的棋子。特征总数是`64*64*5*2=40960`。（请注意，当前Stockfish实现中有一个来自将棋的剩余部分，还有64个额外的未使用特征，但我们将忽略它们以便于本文档的目的）。特征索引可以计算为
```cpp
p_idx = piece_type * 2 + piece_color
halfkp_idx = piece_square + (p_idx + king_square * 10) * 64
```
需要处理的一个特殊情况是国王移动，因为它与所有特征都有关。所有特征都发生了变化，因此执行累加器刷新。

现在，你可能会问，“但是哪个国王呢？”答案是两个国王都要考虑...原因是，特征集合同时考虑了白棋和黑棋的国王。为了区分两者，可以对特征索引执行某种转换。例如，可以在特征索引的计算中添加一个偏移量。这样，网络可以学会根据不同的国王位置评估局面，并且对于白棋和黑棋的国王，特征索引将是不同的。这种处理方法使得网络能够根据国王的位置来理解棋子之间的关系，从而更准确地评估局面。

总之，HalfKP特征集为神经网络提供了一个有效的方法来表示棋盘上的棋子位置和关系。通过将每个特征定义为一个元组，神经网络可以学会理解棋子与国王之间的关系。这种特征集在许多成功的象棋引擎中被广泛应用，并为实现快速、准确的棋盘评估提供了一个强大的基础。

#### 多视角，多累加器

这时，我们需要分别为双方的特征进行计算。白方将保留自己的累加器，黑方也将保留自己的累加器。因此，现在每个位置状态都有两个累加器。实际上，这意味着最大活动特征数量是简单特征集（只有一个视角）的两倍。更新的次数也将翻倍，但总体而言，这是一个非常好的特征集，是所有优秀特性的基础。这会带来一些问题、选项和选择。让我们逐一讨论它们。

##### 如何组合它们？

1. 由于我们现在有两个累加器，我们需要以某种方式将它们组合成一个传递到网络中的向量。这可以用两种（三种）方法来解决。让我们用`A_w`表示白方的累加器，用`A_b`表示黑方的累加器。
   1. 将 `A_w` 和 `A_b` 连接起来，先放 `A_w`，再放 `A_b`。这是最简单的选择。这样做是可以的，输出总是相对于白方的视角。
   2. 将 `A_w` 和 `A_b` 连接起来，如果轮到白方行动，则先放 `A_w`，否则先放 `A_b`，然后放另一个累加器。这种方法的优点是网络可以学习节奏。现在它知道轮到谁行动了。输出始终相对于要行动的一方的视角。
   3. 采用方法 1 或 2，但是不是连接而是交错。所以是 `A_w[0], A_b[0], A_w[1], A_b[1], ...`。这在某些不常使用整个组合累加器的特殊架构中可能是有优势的，在这种情况下，交错意味着所使用的切片始终包含来自白方和黑方视角的相同数量的输出。例如，当将结构稀疏性应用于第一个隐藏层时，这可能变得有用，该隐藏层最终处理累加器的子集。

##### 使用哪些权重？

1. 那么，我们以相同的方式计算白方和黑方的特征，它们的权重是否相关？可以，但不是必须的。引擎在处理这个问题上有所不同。Stockfish 对于白方和黑方使用相同的权重。例如，Seer 使用不同的权重。
   1. 对两种观点使用相同的权重。这意味着棋盘状态需要以某种方式进行定向。否则，白色国王在 E1 会产生与黑色国王在 E8 不同的特征子集，而白色国王在 G4 会产生与黑色国王在 G4 相同的特征子集。这是不好的。解决方案是镜像黑方的位置并改变棋子的颜色，然后棋子放置到特征映射对两者都是合理的。从白方的角度来看，白国王在 E1 应该与从黑方的角度来看，黑国王在 E8 是相同的。现在你可能会认为翻转是正确的方法，但是虽然国际象棋具有垂直对称性，将棋具有旋转对称性。Stockfish 中 HalfKP 的初始实现使用旋转来改变视角，这对于国际象棋来说可以说是错误的（例如，由于王车易位），但这是过去的遗留问题，希望一旦产生使用镜像而不是翻转的好网络，这个问题就会得到解决。
   2. 对不同的观点使用不同的权重。白国王在 E1 实际上等于黑国王在 E8 吗？其他棋子呢？可以说，一个人作为黑方与作为白方玩游戏的方式不同，因此似乎有理由对这些观点使用不同的特征。这就是某些引擎的做法，这样做没有错。唯一的缺点是尺寸更大，训练时间略长，但除此之外，甚至可能更好！它还完全消除了关于翻转或旋转的讨论，并倾向于更简单、性能更高的实现。

#### HalfKP 示例和网络图

与上面针对 "A" 特征集的示意图类似，这里是具有 HalfKP 特征集的相同网络的示意图，权重组合在一起。改变的是，两个累加器的大小都为 4，因此网络最终为 `HalfKP[40960]->4x2->8->1`

让我们看之前的同一个示例位置：`1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1`。

![](img/board_0.png)

现在我们有两个视角，并将分别列出它们的特征。请记住，特征是 `(our_king_square, piece_square, piece_type, piece_color)`，我们使用翻转来为黑方定向方格，颜色是相反的！（可以将“颜色”视为“我们”或“他们”）

白方视角：`(A1, C3, pawn, white)`，`(A1, D4, rook, black)`

黑方视角：`(B1, C6, pawn, black)`，`(B1, D5, rook, white)`

现在网络图看起来更有趣。

![HalfKP[40960]->4x2->8->1](img/HalfKP-40960-4x2-8-1.png)

## 前向传播实现

在这部分，我们将介绍如何评估网络。输入生成将被省略。请记住，我们现在使用的是浮点数，但稍后会发生变化。

### 示例网络

1. 我们将采用一个更通用的定义网络，其架构为 `FeatureSet[N]->M*2->K->1`。因此，层将是：
   1. `L_0`：线性 `N->M`
   2. `C_0`：大小为 `M*2` 的 Clipped ReLu
   3. `L_1`：线性 `M*2->K`
   4. `C_1`：大小为 `K` 的 Clipped ReLu
   5. `L_2`：线性 `K->1`

### 层参数

线性层有两个参数 - 权重和偏置。我们分别将它们称为 `L_0.weight` 和 `L_0.bias`。层还包含输入和输出的数量，在 `L_0.num_inputs` 和 `L_0.num_outputs` 中分别表示。

这里需要说一下权重矩阵的布局。对于稀疏矩阵乘法，列主（内存中的一列是连续的）布局是有利的，因为我们正在添加列，但对于稠密矩阵乘法，这并不明确，行主布局可能更好。现在我们将坚持列主布局，但在进行量化和优化时，我们可能会重新审视行主布局。这意味着 `L_0.weight` 允许以下形式访问各个元素：`L_0.weight[column][row]`。

代码将是伪 C++。

### 累加器

累加器可以用一个数组表示，该数组与搜索堆栈上的其他位置状态信息一起存储。

```cpp
struct NnueAccumulator {
    // 大小为 N 的两个向量。v[0] 用于白方，v[1] 用于黑方视角。
    float v[2][N];

    // 在后续代码片段中，这将用于使访问更简洁
    float* operator[](Color perspective) {
        return v[perspective];
    }
};
```

累加器可以在评估时懒惰地更新，也可以在每一步中更新。这里并不重要，但它必须*以某种方式*更新。如前所述，有两种情况：

1. 累加器必须从头开始重新计算。
2. 重用前一个累加器，并仅根据更改的特征进行更新。

#### 刷新累加器

```cpp
void refresh_accumulator(
    const LinearLayer&      layer,            // 这将始终是 L_0
    NnueAccumulator&        new_acc,          // 结果的存储空间
    const std::vector<int>& active_features,  // 此位置处于活动状态的特征的索引
    Color                   perspective       // 要刷新的视角
) {
    // 首先我们复制层的偏置，这是我们的起点
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = layer.bias[i];
    }

    // 然后我们只是累积所有活动特征的列。这就是累加器要做的事情！
    for (int a : active_features) {
        for (int i = 0; i < M; ++i) {
            new_acc[perspective][i] += layer.weight[a][i];
        }
    }
}
```

#### 更新累加器

```cpp
void update_accumulator(
    const LinearLayer&      layer,            // 这将始终是 L_0
    NnueAccumulator&        new_acc,          // 最好已经提供了存储新累加器的空间。
                                              // 相关部分将被覆盖
    const NNueAccumulator&  prev_acc,         // 之前的累加器，我们要重用的那个
    const std::vector<int>& removed_features, // 被移除的特征的索引
    const std::vector<int>& added_features,   // 被添加的特征的索引
    Color                   perspective       // 要更新的视角，记住我们有两个，
                                              // 它们有单独的特征列表，甚至可能会发生
                                              // 一个被更新，而另一个需要完全刷新的情况
) {
    // 首先我们复制之前的值，这是我们的起点
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = prev_acc[perspective][i];
    }

    // 然后我们减去被移除特征的权重
    for (int r : removed_features) {
        for (int i = 0; i < M; ++i) {
            // 只需减去 r 列
            new_acc[perspective][i] -= layer.weight[r][i];
        }
    }

    // 对于添加的特征，类似地进行操作，但要添加而不是减去
    for (int a : added_features) {
        for (int i = 0; i < M; ++i) {
            new_acc[perspective][i] += layer.weight[a][i];
        }
    }
}
```

就是这样！很简单，对吧？

### 线性层

这是简单的矩阵-向量乘法，你可能会问这有什么复杂的？现在还没有，但在本文后面部分会变得复杂。现在我们不进行优化，但我们至少会写一个使用权重矩阵具有列主布局的版本。

```cpp
float* linear(
    const LinearLayer& layer,  // 要使用的层。我们有两个：L_1、L_2
    float*             output, // 已分配的结果存储空间
    const float*       input   // 输入，即前一个 CReLu 层的输出
) {
    // 首先将偏置复制到输出。我们将在其上添加列。
    for (int i = 0; i < layer.num_outputs; ++i) {
        output[i] = layer.bias[i];
    }

    // 还记得很久以前那个彩虹般的图吗？这就是它。
    // 我们逐列添加，按输入值缩放。
    for (int i = 0; i < layer.num_inputs; ++i) {
        for (int j = 0; j < layer.num_outputs; ++j) {
            output[j] += input[i] * layer.weight[i][j];
        }
    }

    // 让调用者知道使用的缓冲区在哪里结束。
    return output + layer.num_outputs;
}
```

### ClippedReLu

```cpp
float* crelu(,
    int          size,   // 无需使用任何层结构，我们只需要元素数量
    float*       output, // 已分配的结果存储空间
    const float* input   // 输入，即前一个线性层的输出
) {
    for (int i = 0; i < size; ++i) {
        output[i] = min(max(input[i], 0), 1);
    }

    return output + size;
}
```

### 整合在一起

用粗糙的伪代码。特征收集留给读者作为练习 :P。

```cpp
void Position::do_move(...) {
    ... // 做移动操作的东西

    for (Color perspective : { WHITE, BLACK }) {
        if (needs_refresh[perspective]) {
            refresh_accumulator(
                L_0,
                this->accumulator,
                this->get_active_features(perspective),
                perspective
            );
        } else {
            update_accumulator(
                L_0,
                this->accumulator,
                this->get_previous_position()->accumulator,
                this->get_removed_features(perspective),
                this->get_added_features(perspective),
                perspective
            );
        }
    }
}

float nnue_evaluate(const Position& pos) {
    float buffer[...]; // 为结果分配足够的空间

    // 我们需要先准备输入！我们将先放置走棋方的累加器，然后放置另一个。
    float input[2*M];
    Color stm = pos.side_to_move;
    for (int i = 0; i < M; ++i) {
        input[i] = pos.accumulator[stm][i];
        input[i+M] = pos.accumulator[!stm][i];
    }

    float* curr_output = buffer;
    float* curr_input = input;
    float* next_output;

    // 评估一个层，并向前移动输入和输出。
    // 最后一个输出变成下一个输入。
    next_output = crelu(L_0.num_outputs, curr_output, input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = linear(L_1, curr_output, curr_input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = crelu(L_1.num_outputs, curr_output, input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = linear(L_2, curr_output, curr_input);

    // 完成了。最后一层应该在 *curr_output 下输出 1 个值。
    return *curr_output;
}
```

就是这样！这就是整个网络。你说你不能用它？！哦对了，你没有经过训练的网络，真遗憾。

### 考虑网络大小和成本

选择合适的架构很棘手，因为这是一种强度/性能权衡。大型网络提供更准确的评估，但速度影响可能完全抵消实际游戏中的收益。改变某些部分对性能和实力的影响不同于其他部分。最经过实战检验的架构是 `HalfKP[40960]->256x2->32->32->1`，它似乎提供了接近最优的强度/性能比。

#### 特征集

在选择特征集时，可能会想要涉及到复杂的领域特定知识，但相关的成本使得简单的解决方案更具吸引力。前文详细解释的 HalfKP 极简单、快速且足够好。已经尝试过更复杂的特征集，但它们通常无法抵消性能的影响。HalfKP特征容易计算，而且从一个位置到另一个位置的变化很小。

还需要考虑大小。对于上面介绍的架构，HalfKP在第一层产生约1000万个参数，这相当多。对于某些用途，拥有非常大的特征集可能不是问题，可能有数亿个参数，但对于典型用户来说，这是不方便的。此外，增加特征集大小可能会降低某些实现的训练速度，并肯定需要更多时间来收敛。

#### 第一组隐藏神经元

第一层后的隐藏神经元数量是最关键的参数，但也对速度和大小影响最大。在上述架构中，每个角度的神经元数量为256。与此参数相关的成本有两方面。它增加了更新累加器所需的操作次数。对于优化的实现，必须考虑寄存器数量 - 在 Stockfish 中，超过256个神经元需要在特征索引上进行多次遍历，因为 AVX2  没有足够的寄存器。它还决定了第一个密集线性层的大小，这是目前为止最大的。

#### 进一步的层次

与机器学习中考虑的典型网络不同，这里的大部分知识都存储在第一层，因此在输出附近添加更多小层对精度的提升很小，甚至在采用量化时可能会因错误累积而有害。NNUE 网络保持异常浅层，输出更接近的层的大小可以保持较小以提高性能。

## 使用 Pytorch 训练网络

这将非常简短，因为这毕竟是在 nnue-pytorch 代码库上，所以你可以直接查看代码！我们不会解释 Pytorch 是如何工作的，但是我们会解释一些基本知识，以及为了适应这种非常规用例所需要的特殊之处。

让我们继续使用前向传播实现中的架构。

### 模型规范

Pytorch 内置了线性层的类型，所以定义模型非常简单。

```python
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        self.ft = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, K)

    # 输入是一个完整的批次！
    # `stm` 表示白方是否要走棋。1 = 真，0 = 假。
    def forward(self, white_features, black_features, stm):
        w = self.ft(white_features) # 白方的视角
        b = self.ft(black_features) # 黑方的视角

        # 记住，我们根据谁要走棋来为 2 个视角的累加器排序。
        # 所以我们通过插值在 `stm` 和 `1-stm` 张量之间混合两种可能的排序。
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # 运行线性层并使用 clamp_ 作为 ClippedReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        return self.l2(l2_x)
```

这部分相当简单，而且 Pytorch 显然会自动处理反向传播。真棒！困难的部分可能出人意料地是提供数据。

### 准备输入

这部分有两个主要瓶颈。

1. 解析训练数据集
2. 准备张量输入

#### 解析训练数据集并将它们移至 Python 端

你可能会想用 Python 实现这个功能。它可以工作，但遗憾的是，它的速度要慢好几个数量级。在 nnue-pytorch 中，我们创建了一个 C++ 共享库，它实现了一个非常快速的训练数据解析器，并以可以快速转换为输入张量的形式提供数据。你还可以查看 Seer 的实现，因为它更简单。

对于 C 和 Python 之间的互操作，Ctypes 相当简单，对于这个任务已经足够了。我们只是传递指针而已。只需记住，只有 C 有稳定的 ABI，因此所有从 Python 访问的函数都需要是 `extern "C"`。

数据读取器的架构是在创建时传递一个文件，并生成所请求的工作线程数量，这些线程可以异步地处理数据并准备**整个批次**，然后从 Pytorch 端检索。逐个进行是不行的，需要削减一些角落。你可能会问为什么？Pytorch 可以将多个张量转换为一个批次，那么问题是什么呢？让我们看看...

还记得输入是稀疏的吗？现在假设我们的批量大小是 8192。如果我们发送 8192 个稀疏张量并尝试从它们中形成一个批次，会发生什么？嗯，Pytorch 不喜欢自己这样做，我们需要帮助它。最好的方法是形成一个大型的二维稀疏输入张量，包含整个批次。它有 2 个稀疏维度，索引是 `(position_index, feature_index)`，非常简单，性能优越，无需创建临时张量！如果我们对稀疏张量这样做，那么我们也可以对所有其他张量这样做，因为这更容易。从一开始就形成整个批次的事实也意味着我们可以减少分配的数量，并为批次部分使用更好的内存布局。

因此，我们也不能简单地使用 Pytorch 的 `DataLoader`，而需要将其用作一个简单的包装器。但这种努力是值得的。一个工作线程通常可以在不出问题的情况下饱和甚至是高端 GPU。

#### 实际上发送了什么，如何发送，以及如何将其转换为张量？

所需的最低内容是特征（从两个角度），要移动的方（用于累加器切片顺序）和位置评估（得分）。让我们看看它们是如何表示的。

```cpp
struct SparseBatch {
    SparseBatch(const std::vector<TrainingDataEntry>& entries) {

        // 批次中的位置数量
        size = entries.size();

        // 整个批次中的白/黑活动特征总数。
        num_active_white_features = 0;
        num_active_black_features = 0;

        // 每个位置要移动的方。白色为1，黑色为0。
        // 在前向传播中需要对累加器切片进行排序。
        stm = new float[size];

        // 每个位置的分数。这是我们将教给网络的值。
        score = new float[size];

        // 活动特征的索引。
        // 为什么大小是 * 2？！答案是索引是二维的
        // (position_index, feature_index)。实际上它是一个尺寸
        // (num_active_*_features, 2)的矩阵。我们以行优先方式填充它，
        // 并在 pytorch 端对其进行转置，因为它想要另一种方式
        // 来处理。
        // 重要：我们必须确保索引按升序排列。
        // 首先是第一个位置，然后是第二个，然后是第三个，
        // 依此类推。而对于一个位置的特征，特征索引
        // 也是按升序排列。为什么需要这样做稍后会显现出来。
        white_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];
        black_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];

        fill(entries);
    }

    void fill(const std::vector<TrainingDataEntry>& entries) {
        ...
    }

    int size;
    int num_active_white_features;
    int num_active_black_features;

    float* stm;
    float* score;
    int* white_features_indices;
    int* black_features_indices;

    ~SparseBatch()
    {
        // RAII！或者使用 std::unique_ptr<T[]>，但请记住，只有原始指针应该
        // 通过语言边界传递，因为 std::unique_ptr 没有稳定的 ABI
        delete[] stm;
        delete[] score;
        delete[] white_features_indices;
        delete[] black_features_indices;
    }
};
```

and in python

```python
class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('stm', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('white_features_indices', ctypes.POINTER(ctypes.c_int)),
        ('black_features_indices', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self, device):
        # 这是说明性的。实际上，您可能需要将这些
        # 转移到 GPU。您还可以异步执行，但请记住确保
        # 源在复制完成之前存活足够长的时间。

        # 这是将指针转换为 pytorch 张量的好方法。
        # 需要传递形状，记住我们正在形成整个批次，第一
        # 维度始终是批次大小。
        stm_t = torch.from_numpy(np.ctypeslib.as_array(self.stm, shape=(self.size, 1)))
        score_t = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1)))

        # 正如我们所说，索引需要进行转置，以使位置索引优先
        white_features_indices_t = torch.transpose(torch.from_numpy(np.ctypeslib.as_array(self.white_features_indices, shape=(self.num_active_white_features, 2))), 0, 1).long()
        black_features_indices_t = torch.transpose(torch.from_numpy(np.ctypeslib.as_array(self.black_features_indices, shape=(self.num_active_white_features, 2))), 0, 1).long()

        # 值都是 1，所以我们可以轻松地在原地创建这些张量。
        # 无需通过复制。
        white_features_values_t = torch.ones(self.num_active_white_features)
        black_features_values_t = torch.ones(self.num_active_black_features)

        # 现在魔法来了。我们通过给出非零值的索引（活动特征索引）和这些值（1！）来构造一个稀疏张量。
        # 张量的大小是 batch_size*NUM_FEATURES，这通常会非常大，但由于密度仅为 ~0.1%，所以它占用的空间非常小，而且可以更快地进行前向传播。
        # 为了获得最大性能，我们确实有点作弊。通常，Pytorch 会检查正确性，这是一项昂贵的 O(n) 操作。
        # 通过使用 _sparse_coo_tensor_unsafe，我们避免了这个问题。
        white_features_t = torch._sparse_coo_tensor_unsafe(white_features_indices_t, white_features_values_t, (self.size, NUM_FEATURES))
        black_features_t = torch._sparse_coo_tensor_unsafe(black_features_indices_t, black_features_values_t, (self.size, NUM_FEATURES))

        # 合并是什么？！它确保索引是唯一的且有序的。
        # Now you probably see why we said the inputs must be ordered from the start.
        # This is normally a O(n log n) operation and takes a significant amount of
        # time. But here we **know** that the tensor is already in a coalesced form,
        # therefore we can just tell pytorch that it can use that assumption.
        white_features_t._coalesced_(True)
        black_features_t._coalesced_(True)

        # Now this is what the forward() required!
        return white_features_t, black_features_t, stm_t, score_t

# Let's also tell ctypes how to understand this type.
SparseBatchPtr = ctypes.POINTER(SparseBatch)
```

### 特征因子分解

让我们再次关注特征。我们将更详细地了解 `HalfKP` 特征集。嗯...我们选了 `P`，并将其重复了 64 次，每个方格一次...肯定这 64 个桶之间存在某种关联...我们如何告诉网络它们之间的关系呢？通过引入虚拟特征！

我们有 40960 个 `HalfKP` 特征和 640 个 `P` 特征。它们如何相互映射？确切的计算取决于你的索引方案，但我们可以用简单的术语来描述它。

```
HalfKP` 特征是 `(king_square, piece_square, piece_type, piece_color)
```

`P` 特征是 `(piece_square, piece_type, piece_color)`。

两者之间有 3 个公共部分。因此，对于每个 `P` 特征，都有 64 个相应的 `HalfKP` 特征。我们可以将 40960 个输入扩展到 40960+640，包括 `HalfKP` 和 `P` 特征。现在每个位置对于每个视角最多有 64 个特征（32 个 `HalfKP` 和 32 个 `P`）。数据加载器和前向传播中的其他部分都没有改变，我们只是增加了更多的特征！但是我们不想在实际游戏中使用它们，这样做代价太大，而且有点毫无意义。我们知道哪些特征与其他特征相关，所以在使用网络进行游戏之前，让我们以某种方式将它们合并起来。

#### Virtual feature coalescing

So how can we coalesce them? Let's look how matrix and vector multiplication is done again. Consider the example position from before (`1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1`).

![](img/board_0.png):

Let's focus on the feature `(A1, C3, pawn, white)`. Now, we're also gonna add a `P` feature `(C3, pawn, white)`. What happens when the input goes through the first layer?

```cpp
accumulator += weights[(A1, C3, pawn, white)];
accumulator += weights[(C3, pawn, white)];
```

which is equivalent to

```cpp
accumulator += weights[(A1, C3, pawn, white)] + weights[(C3, pawn, white)];
```

So the relation is very simple. We just need to add the weights of each `P` feature to all the related `HalfKP` feature weights!

#### Other factors

Sometimes it's possible to add even more factors. It should be noted however, that just adding more factors doesn't necessarily improve the training and may even cause it to regress. In general, whether using some factors helps or not depends on the training setup and the net being trained. It's always good to experiment with this stuff. With that said however, we can consider for example the following factors for `HalfKP`.

##### "K" factors

The king position, 64 features. This one requires some careful handling as a single position has this feature multiple times - the number of pieces on the board. This means that the input for this feature is no longer 1, but the number of position on the board instead. This is needed purely because with HalfKP the king feature is not encoded anywhere. HalfKA doesn't need it for example because it specifically has the feature for the king's position. In general, handling this is tricky, it may even require reducing the gradient for these features (otherwise the gradient is `input*weight`, but input is large compared to others).

##### "HalfRelativeKP" factors

In `HalfKP` we use the absolute piece position, but what if we encoded the position as relative to the king? There's 15x15 such relative position possible, and most of them correspond 1:many to some `HalfKP` feature. The HalfRelativeKP feature index could be calculated for example like this:
```cpp
int get_half_relative_kp_index(Color perspective, Square king_sq, Square piece_sq, Piece piece)
{
    const int p_idx = static_cast<int>(piece.type()) * 2 + (piece.color() != perspective);
    const Square oriented_king_sq = orient_flip(perspective, king_sq);
    const Square oriented_piece_sq = orient_flip(perspective, piece_sq);
    // The file/rank difference is always in range -7..7, and we need to map it to 0..15
    const int relative_file = oriented_piece_sq.file() - oriented_king_sq.file() + 7;
    const int relative_rank = oriented_piece_sq.rank() - oriented_king_sq.rank() + 7;
    return (p_idx * 15 * 15) + (relative_file * 15) + relative_rank;
}
```

#### Real effect of the factorizer

While the factorizer helps the net to generalize, it seems to only be relevant in the early stages, that is when the net doesn't really know anything yet and makes the net learn faster. But it quickly becomes unimportant and in later stages of the training can be removed to gain some training speed (after all it can add a lot of active features).

### Loss functions and how to apply them

#### The Goal

Training a network is really just minimizing a loss function, which needs to be smooth and have a minimum at the "optimal" evaluation (the training target). For the purpose of NNUE, this is done by gradient descent through usual machine learning methods (there are also non-gradient methods that are not described here).

#### Converting the evaluation from CP-space to WDL-space

By CP-space we mean the centipawn scale (or engine's internal units). By WDL-space we mean 0=loss, 0.5=draw, 1=win.

It's of course possible to apply the loss function directly on the evaluation value, but this can lead to large gradients (or a lot of hyperparameter tuning), restricts the set of loss functions available, and doesn't allow using results for loss. We will focus on evaluation in WDL-space. But how to convert between these spaces? Usually the evaluation to performance correspondence can be well fitted by a sigmoid. For example in some data generated by Stockfish we have:

![](img/sigmoid_wdl_fit.png)

so in the code we may do the following:
```python
scaling_factor = 410 # this depends on the engine, and maybe even on the data
wdl_space_eval = torch.sigmoid(cp_space_eval / scaling_factor)
```

This transformation also has the nice effect that large evaluations become "closer" together, which aligns well with the real play, where large evaluations don't need to be that precise.

#### Using results along the evaluation

With the values for which we will compute loss being in WDL-space, we may now interpolate them with game results. We will introduce a `lambda_` parameter that governs the interpolation.
```python
# game_result is in WDL-space
wdl_value = lambda_ * wdl_space_eval + (1 - lambda_) * game_result
```

The interpolation can also be applied to the loss.
```python
loss_eval = ... # loss between model eval and position eval
loss_result = ... # loss between model eval and game result
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
```

Which way works better depends on your case :)

#### Mean Squared Error (MSE)

Now we know what we're trying to fit; let's look at how we will fit them.

This is a very simple loss function that just takes a square of the difference between the predicted value and the target. This results in a nice linear gradient.

With interpolation applied before:
```python
scaling = ... # depends on the engine and data. Determines the shape of
              # the sigmoid that transforms the evaluation to WDL space
              # Stockfish uses values around 400
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)
wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result
loss = (wdl_eval_model - wdl_value_target)**2
```

With interpolation applied after:
```python
scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)
loss_eval   = (wdl_eval_model - wdl_eval_target)**2
loss_result = (wdl_eval_model - game_result)**2
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
```

##### loss

![](img/mse_loss.png)
![](img/mse_loss_contour.png)

##### grad

![](img/mse_loss_grad.png)
![](img/mse_loss_grad_contour.png)

#### Cross entropy

This loss function is usually used for continuous classification problems, and our use case could be considered one.

Care must be taken around domain boundaries. Usually a very small value (epsilon) is added such that the values never reach 0 under the logarithm.

With interpolation applied before:
```python
epsilon = 1e-12 # to prevent log(0)
scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)
wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result

# The first term in the loss has 0 gradient, because we always
# differentiate with respect to `wdl_eval_model`, but it makes the loss nice
# in the sense that 0 is the minimum.
loss = (wdl_value_target * log(wdl_value_target + epsilon) + (1 - wdl_value_target) * log(1 - wdl_value_target + epsilon))
      -(wdl_value_target * log(wdl_eval_model   + epsilon) + (1 - wdl_value_target) * log(1 - wdl_eval_model   + epsilon))
```

With interpolation applied after:
```python
epsilon = 1e-12 # to prevent log(0)
scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)

# The first term in the loss has 0 gradient, because we always
# differentiate with respect to `wdl_eval_model`, but it makes the loss nice
# in the sense that 0 is the minimum.
loss_eval   = (wdl_eval_target * log(wdl_eval_target + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_target + epsilon))
             -(wdl_eval_target * log(wdl_eval_model  + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_model  + epsilon))
loss_result = (game_result     * log(wdl_eval_target + epsilon) + (1 - game_result)     * log(1 - wdl_eval_target + epsilon))
             -(game_result     * log(wdl_eval_model  + epsilon) + (1 - game_result)     * log(1 - wdl_eval_model  + epsilon))
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
```

##### loss

![](img/cross_entropy_loss.png)
![](img/cross_entropy_loss_contour.png)

##### grad

![](img/cross_entropy_loss_grad.png)
![](img/cross_entropy_loss_grad_contour.png)

## Quantization

At the start of this document it was briefly mentioned what quantization is and that it will be important. Now it's the time to understand it properly. The goals is that we want to use the smallest possible integers everywhere. Most CPU architectures provide instructions that can work on 8, 16, 32, or even 64 int8 values at a time, and we should take advantage of that. That means we need to use int8 values, with range -128..127, for weights and inputs; or int16, with range -65536..65535, where int8 is not possible.

Coming up with the right quantization scheme is not easy, so first we'll present the one currently used by Stockfish, and then we'll explain how to get there, how to code it, and finally how to optimize it.

### Stockfish quantization scheme

#### Feature Transformer

Let's start with the feature transformer. Recall that its purpose is to accumulate between 0 to 30 (for HalfKP) rows of weights. We want to have int8 values as inputs to the later layers, with the activation range (ClippedReLU) being 0..127, but that means that using int8 integers for the accumulator doesn't provide enough space as the values would go beyond the range of int8 before applying the ClippedReLU... so we use int16 for the accumulator and then convert to int8 when doing the ClippedReLU.

#### Linear layer

We wanted int8 inputs and we can get them without losing too much precision. The nature of matrix-purposed SIMD instructions is that, thankfully, the accumulation happens in int32. So we don't experience the same issue as in the feature transformer where we're manually adding rows, and we can utilize the int8 multiplication with int32 accumulation to the fullest extent, and only later go back to int8 in the ClippedReLU layer.

#### ClippedReLU

Nothing special going on in here. Since the inputs are not being scaled, this is simply the same operation but in a different domain. Instead of clamping to 0..1 we clamp to 0..127. The input type is usually different than the output type as inputs will be either int32 or int16, and the output we want is int8. The values won't change but the conversion needs to be applied.

### The math of quantization and how to make it fit

To quantize the network we need to multiply the weights and biases by some constant to translate them to a different range of values. This poses a problem when confronted with multiplication during network inference - `(a*x) * (a*w) = a*a*x*w`, and we have to sometimes scale back the outputs too. But each layer is still independent so let's go through them one by one again.

#### Feature Transformer

Remember we want our activation range to change from 0..1 to 0..127. Since the feature transformer is a purely additive process,  it's enough that we multiply the weights and biases by 127. Both weights and biases are stored as int16. We could divide the output by some factor `a` to get more precision, in which case the weights and biases would have to be multiplied by `a*127` instead, but in practice it increases the accuracy only by a little bit.

#### Linear layer

To arrive at int8 weights we have to apply some scaling factor. This scaling factor ultimately depends on how much precision needs to be preserved, but cannot be too large because then the weights will be limited in magnitude. For example if we took the scaling factor to be 64 (used in Stockfish), then the maximum weight in the floating point space is `127/64=1.984375`. This is enough to have good nets, but care needs to be taken to clamp the weights during training so that they don't go outside that range. The scaling factor of 64 can also be understood as the smallest weight step that can be represented being `1/64=0.015625`.

A linear layer is just matrix multiplication, so we're multiplying inputs and weights, but now both are scaled relative to the float version. Let's denote the input scaling factor (activation range scaling) as `s_A`, and the weight scaling factor by `s_W`. `x` is the unquantized input, `w` is the unquantized weight, 'b' is the unquantized bias, and `y` is the unquantized output.
So we have:
```
x * w + b = y
((s_A * x) * (s_W * w)) + (b * s_A * s_W) = (y * s_A) * s_W
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) / s_W = (y * s_A)
```
From that we learn that we need to scale the bias by `(s_A * s_W)`, weights by `s_W`, and divide output by `s_W` to get the desired `(y * s_A)`, which is correctly scaled to the activation range.

Now, this applies only when the next layer is the ClippedReLU layer. For the last layer the output range is very different and the quantization will also be different. In Stockfish we want the last layer to output values in range -10000..10000 while still keeping int8 weights. This can be achieved without any additional scaling factors, but it's easiest to do and understand with an additional scaling factor.

We'll introduce a new scaling factor, `s_O`. This scaling factor, unlike others, needs to be applied to the output both during training (for loss calculation against the actual evaluation) and inference. The purpose of it is to scale the float output of the network to match the range of the integer evaluation used by Stockfish. Basically it means that `1` in the float space is equal to `s_O` internal evaluation units. It has an additional advantage that it allows us to have the layer weights be similar in magnitude to the previous layers.

So the math is now:
```
x * w + b = y
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O = ((y * s_A) * s_W) * s_O
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O / s_A / s_W = (y * s_O)
(((s_A * x) * (s_W / s_A * w)) + (b * s_A * s_W / s_A)) * s_O / s_W = (y * s_O)
(((s_A * x) * (s_W * s_O / s_A * w)) + (b * s_W * s_O)) / s_W = (y * s_O)
```
From that we learn that we need to scale the bias by `s_W * s_O`, weights by `s_W * s_O / s_A`, and divide the output by `s_W` to get the desired `(y * s_O)`.

### Implementation

For the unoptimized implementation not much changes. One just has to remember to change the data types to integers with desired size, scale weights on input, and divide the output from linear layers by `s_W`. `s_W` is usually chosen to be a power of two, so that this operation is a simple bitwise right shift, as there are no SIMD division instructions for integers and even if there were it would be slow.

### Optimized implementation

For simplicity we will focus on optimization only for the AVX2 extension of the x86-64 instruction set.

#### Feature Transformer

The benefit from SIMD for the feature transformer is two-fold:

1. multiple additions per instruction can be performed
2. large total register size means we don't need to write to memory as often

Our accumulation structure doesn't change much, we just change float to int16:
```cpp
// We now also make sure that the accumulator structure is aligned to the cache line.
// This is not strictly required by AVX2 instructions but may improve performance.
struct alignas(64) NnueAccumulator {
    // Two vectors of size N. v[0] for white's, and v[1] for black's perspectives.
    int16_t v[2][N];

    // This will be utilised in later code snippets to make the access less verbose
    int16_t* operator[](Color perspective) {
        return v[perspective];
    }
};
```

Now let's look at the refresh function. For simplicity we will assume that there is enough registers so that spills don't happen, but in reality (`M > 256`) it is required to do multiple passes over the active features, each time considering a part of the accumulator only. A single AVX2 register can fit 16 int16 values and there is 16 AVX2 registers (32 since AVX-512).

```cpp
void refresh_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // storage for the result
    const std::vector<int>& active_features,  // the indices of features that are active for this position
    Color                   perspective       // the perspective to refresh
) {
    // The compiler should use one register per value, and hopefully
    // won't spill anything. Always check the assembly generated to be sure!
    constexpr int register_width = 256 / 16;
    static_assert(M % register_width == 0, "We're processing 16 elements at a time");
    constexpr int num_chunks = M / register_width;
    __m128i regs[num_chunks];

    // Load bias to registers and operate on registers only.
    for (int i = 0; i < num_chunks; ++i) {
        regs[i] = _mm256_load_si256(&layer.bias[i * register_width]);
    }

    for (int a : active_features) {
        for (int i = 0; i < num_chunks; ++i) {
            // Now we do 1 memory operation instead of 2 per loop iteration.
            regs[i] = _mm256_add_epi16(regs[i], &layer.weight[a][i * register_width]);
        }
    }

    // Only after all the accumulation is done do the write.
    for (int i = 0; i < num_chunks; ++i) {
        _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]);
    }
}
```

similarily for the update:

```cpp
void update_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // it's nice to have already provided storage for
                                              // the new accumulator. Relevant parts will be overwritten
    const NNueAccumulator&  prev_acc,         // the previous accumulator, the one we're reusing
    const std::vector<int>& removed_features, // the indices of features that were removed
    const std::vector<int>& added_features,   // the indices of features that were added
    Color                   perspective       // the perspective to update, remember we have two,
                                              // they have separate feature lists, and it even may happen
                                              // that one is updated while the other needs a full refresh
) {
    // The compiler should use one register per value, and hopefully
    // won't spill anything. Always check the assembly generated to be sure!
    constexpr int register_width = 256 / 16;
    static_assert(M % register_width == 0, "We're processing 16 elements at a time");
    constexpr int num_chunks = M / register_width;
    __m128i regs[num_chunks];

    // Load the previous values to registers and operate on registers only.
    for (int i = 0; i < num_chunks; ++i) {
        regs[i] = _mm256_load_si256(&prev_acc[perspective][i * register_width]);
    }

    // Then we subtract the weights of the removed features
    for (int r : removed_features) {
        for (int i = 0; i < num_chunks; ++i) {
            regs[i] = _mm256_sub_epi16(regs[i], &layer.weight[r][i * register_width]);
        }
    }

    // Similar for the added features, but add instead of subtracting
    for (int a : added_features) {
        for (int i = 0; i < num_chunks; ++i) {
            regs[i] = _mm256_add_epi16(regs[i], &layer.weight[a][i * register_width]);
        }
    }

    // Only after all the accumulation is done do the write.
    for (int i = 0; i < num_chunks; ++i) {
        _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]);
    }
}
```

#### Linear layer

Matrix multiplication is hard to optimize in general, and there are many approaches depending on the size of the matrices. Since we expect the layers to be small, we will not delve into any fancy blocked algorithms. And just rely on manual unrolling and trying to process multiple values at a time. This is not optimal, but it's simple and very close. We will only describe the case where the number of outputs is divisible by 4. The output layer has 1 output but it's also very small and doesn't require anything clever. We will also require the input size to be a multiple of 32, otherwise adding 0 padding is required.

```cpp
int32_t* linear(
    const LinearLayer& layer,  // the layer to use. We have two: L_1, L_2
    int32_t*           output, // the already allocated storage for the result
    const int8_t*      input   // the input, which is the output of the previous CReLu layer
) {
    constexpr int register_width = 256 / 8;
    assert(layer.num_inputs % register_width == 0, "We're processing 32 elements at a time");
    assert(layer.num_outputs % 4 == 0, "We unroll by 4");
    const int num_in_chunks = layer.num_inputs / register_width;
    const int num_out_chunks = layer.num_outputs / 4;

    for (int i = 0; i < num_out_chunks; ++i) {
        // Prepare weight offsets. One offset for one row of weights.
        // This is a simple index into a 2d array.
        const int offset0 = (i * 4 + 0) * layer.num_inputs;
        const int offset1 = (i * 4 + 1) * layer.num_inputs;
        const int offset2 = (i * 4 + 2) * layer.num_inputs;
        const int offset3 = (i * 4 + 3) * layer.num_inputs;

        // Accumulation starts from 0, we add the bias only at the end.
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
        for (int j = 0; j < num_in_chunks; ++j) {
            // We unroll by 4 so that we can reuse this value, reducing the number of
            // memory operations required.
            const __m256i in = _mm256_load_si256(&input[j * register_width]);

            // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
            // For definition see below.
            m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256(&layer.weights[offset0 + j * register_width]));
            m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256(&layer.weights[offset1 + j * register_width]));
            m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256(&layer.weights[offset2 + j * register_width]));
            m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256(&layer.weights[offset3 + j * register_width]));
        }

        const __m128i bias = _mm256_load_si256(&layer.bias[i * 4]);
        // This function adds horizontally 8 values from each sum together, producing 4 int32 values.
        // For the definition see below.
        __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, bias);
        // Here we account for the weights scaling.
        outval = _mm256_srai_epi32(outval, log2_weight_scale);
        _mm256_store_si256(&output[i * 4], outval);
    }

    return output + layer.num_outputs;
}
```

##### m256_add_dpbusd_epi32

![](img/m256_add_dpbusd_epi32.png)

The output needs to be horizontally accumulated further, but it's faster to do it with 4 sums (sum0, sum1, sum2, sum3) later.

This function can benefit from VNNI extension, here controlled by `USE_VNNI`.

```cpp
void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {
#if defined (USE_VNNI)

    // This does exactly the same thing as explained below but in one instruction.
    acc = _mm256_dpbusd_epi32(acc, a, b);

#else

    // Multiply a * b and accumulate neighbouring outputs into int16 values
    __m256i product0 = _mm256_maddubs_epi16(a, b);

    // Multiply product0 by 1 (idempotent) and accumulate neighbouring outputs into int32 values
    product0 = _mm256_madd_epi16(product0, kOnes256);

    // Add to the main int32 accumulator.
    acc = _mm256_add_epi32(acc, product0);

#endif
};
```

##### m256_haddx4

This function takes 4 \_\_m256i registers containing 8 int32 values each, accumulates them horizontally, and produces one \_\_m128i register containing 3 int32 values, each corresponding to one input sum. In the matrix multiplication above we keep one sum per weight row/input, so in the end we fill the output 4 values at a time.

![](img/m256_haddx4.png)

```cpp
__m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3, __m128i bias) {
    sum0 = _mm256_hadd_epi32(sum0, sum1);
    sum2 = _mm256_hadd_epi32(sum2, sum3);

    sum0 = _mm256_hadd_epi32(sum0, sum2);

    __m128i sum128lo = _mm256_castsi256_si128(sum0);
    __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

    return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
};
```

#### Linear layer with sparse input

In the previous part we described the generic dense matrix multiplication, but let's try to delve a little bit deeper. The case we will be considering here is similar to how our feature transformer operates, but here we always need to perform the full operation instead and the matrices are smaller. But why are we even considering this? Well, it turns out that the feature transformer output, after being passed through ClippedReLU, can have quite significant sparsity. Here is some data presenting the density of the inputs to the first dense fully connected layer, for networks with different feature transformer size:
![](img/fc_input_density.png)
(boxes corresponds to the [25%, 75%] interval, whiskers correspond to the [1%, 99%] interval)

That's already <=15% density for the common sizes, and it's consistent between different networks! However we cannot make it as much faster, there is some cost related to the changed access patterns and more required preprocessing, so whether this approach works for your particular case needs to be thoroughly tested.

Let's see the code that can take advantage of it.

```cpp
int lsb(std::uint32_t v) {
    // returns the least significant set bit in v
    // implementation detail
    // can be implemented for example using compiler intrinsics
    // https://www.chessprogramming.org/BitScan#Leading_Zero_Count
}

// This implementation requires changing the layout and expanding the weights to int16.
// We will transpose the weights as now we'll be going through the columns instead of rows.
void load_weights(
    const LinearLayer& layer,
    const int8_t* data
) {
    static_assert(is_same_v<LinearLayer::WeightType, int16_t>,
        "This approach requires weights to be 16 bit. Otherwise it's hard to widen the multiplication output to 32 bits.");

    for (int i = 0; i < layer.num_outputs; ++i) {
        for (int j = 0; j < layer.num_inputs; ++j) {
            layer.weights[j*layer.num_outputs + i] = data[i*layer.num_inputs + j];
        }
    }

    // For AVX2 we must also swap some lanes in the weights. This is
    // because AVX2 registers functions as two 128 bit ones, and
    // therefore some data is interleaved in the inference process.
    // This makes it so that they end up where we want.
    // Will be more apparent in the visualization.
    // This effectively swaps out the middle 2 64-bit chunks in each 256-bit chunk.
    for (int i = 0; i < layer.num_outputs; ++i) {
        for (int j = 0; j < layer.num_inputs; ++j) {
            const int simd_lane = j % 16;
            const int simd_lane_64 = simd_lane / 4;
            if (simd_lane_64 == 1) {
                swap(
                    layer.weights[i*layer.num_outputs + j + 0],
                    layer.weights[i*layer.num_outputs + j + 4]
                );
            }
        }
    }
}

int32_t* linear_sparse_input(
    const LinearLayer& layer,
    int32_t*           output,
    const int8_t*      input
) {
    static_assert(is_same_v<LinearLayer::WeightType, int16_t>,
        "This approach requires weights to be 16 bit. Otherwise it's hard to widen the multiplication output to 32 bits.");

    constexpr int register_width = 256 / 8;
    constexpr int input_register_width = register_width; // uint8_t
    constexpr int output_register_width = register_width / 4; // int32_t
    constexpr int output_chunk_size = output_register_width * 2; // we will be processing 2 registers at a time
    assert(layer.num_outputs % output_chunk_size == 0, "We're processing 16 output elements at a time");
    assert(layer.num_inputs % input_register_width == 0);

    // We need to find out the indices of the input values that are non-zero
    uint16_t nnz_input_indices[layer.num_inputs];
    int num_nnz_input_indices = 0;

    for (int i = 0; i < layer.num_inputs; i += input_register_width) {
        const __m256i input_chunk = _mm256_load_si256(input + i);
        // Find out where the values are greater than 0 and set the corresponding bits in nnz
        uint32_t nnz =
            _mm256_movemask_epi8(
                _mm256_cmpgt_epi8(input_chunk, _mm256_setzero_si256())
            );

        // Extract the indices of the set bits in nnz
        while (nnz) {
            const int lsb_index = lsb(nnz);
            nnz &= nnz - 1; // reset the least significant set bit in nnz
            nnz_input_indices[num_nnz_input_indices++] = i + lsb_index;
        }
    }

    // First we just copy the biases. Compilers are good at vectorizing this.
    // Could also use memcpy
    for (int i = 0; i < layer.num_outputs; ++i) {
        output[i] = layer.biases[i];
    }

    const int num_chunks = layer.num_outputs / output_chunk_size;
    int i = 0;
    for (; i + 1 < num_nnz_input_indices; i += 2) {
        // We will try to process 2 at a time as much as possible,
        // as we can utilize the available intrinsics better.
        // Will become more apparant on the visualization.
        const int input_id0 = nnz_input_indices[i+0];
        const int input_id1 = nnz_input_indices[i+1];
        const __m256i factor = _mm256_set1_epi32(
            input[input_id0] | (input[input_id1] << 16)
        );

        for (int j = 0; j < num_chunks; ++j) {
            const int output_offset0 = (j*2 + 0)*output_register_width;
            const int output_offset1 = (j*2 + 1)*output_register_width;

            // Weights are packed 2 times as densely as the output.
            const int weight_offset  = (j*1 + 0)*output_register_width;

            // Each chunk requires a load+store.
            // However, if the output is small enough it can be unrolled and
            // all outputs might fit into the registers.
            // Though the compiler probably is not allowed to do it by itself.
            __m256i sum0 = _mm256_load_si256(output + output_offset0);
            __m256i sum1 = _mm256_load_si256(output + output_offset1);

            // Remember, weights are 16 bit here, so one __m256i can hold 16 of them.
            const __m256i col0 = _mm256_load_si256(
                layer.weights + input_id0 * layer.num_outputs + weight_offset
            );
            const __m256i col1 = _mm256_load_si256(
                layer.weights + input_id1 * layer.num_outputs + weight_offset
            );

            // See next below for visualization
            m256_process_chunk(sum0, sum1, col0, col1, factor);

            _mm256_store_si256(output + output_offset0, sum0);
            _mm256_store_si256(output + output_offset1, sum1);
        }
    }

    // Process the remaining single input
    for (; i < num_nnz_input_indices; ++i) {
        const int input_id = nnz_input_indices[i];
        const __m256i factor = _mm256_set1_epi32(input[input_id]);

        for (int j = 0; j < num_chunks; ++j) {
            const int output_offset0 = (j*2 + 0)*output_register_width;
            const int output_offset1 = (j*2 + 1)*output_register_width;

            const int weight_offset  = (j*1 + 0)*output_register_width;

            __m256i sum0 = _mm256_load_si256(output + output_offset0);
            __m256i sum1 = _mm256_load_si256(output + output_offset1);

            const __m256i col0 = _mm256_load_si256(
                layer.weights + input_id * layer.num_outputs + weight_offset
            );

            m256_process_chunk(sum0, sum1, col0, _mm256_setzero_si256(), factor);

            _mm256_store_si256(output + output_offset0, sum0);
            _mm256_store_si256(output + output_offset1, sum1);
        }
    }

    return output + layer.num_outputs;
}
```

##### m256_process_chunk

This function takes int16 weights, a factor being a composition of 2 int8 inputs broadcasted as int32, and produces int32 outputs.

![](img/m256_process_chunk.png)

```cpp
void m256_process_chunk(__m256i& sum0, __m256i& sum1, __m256i col0, __m256i col1, __m256i factor) {
    // We interleave the two columns, because madd adds adjacent values.
    // This way we effectively add the results from both columns.
    sum0 = _mm256_add_epi32(
        sum0, _mm256_madd_epi16(factor, _mm256_unpacklo_epi16(col0, col1))
    );
    sum1 = _mm256_add_epi32(
        sum1, _mm256_madd_epi16(factor, _mm256_unpackhi_epi16(col0, col1))
    );
}
```

#### Linear layer with sparse input and blocked sparse output

Let's go one step further. For now all linear layers had dense outputs, but we can consider a layer where each input is connected only to a subset of outputs. We can consider the weights to be 0 where no connection is present. To make it possible to implement efficiently with vectorization in mind we have to zero out whole blocks of weights. A 16x128 Weight matrix with 2 non-zero 1x16 blocks per input may look like this for example:

![](img/m256_block_sparse_weight_matrix.png)

For AVX2 such blocks must be at least 8 int32s (type of the output values) wide, but we will consider only 16-wide blocks because it's more convenient. With this approach one can have for example a linear layer with 256 outputs, but only 4 (this being constant is quite important for being able to write optimized code) non-zero weight blocks of size 16 per input, effectively having each input only affect 64 outputs.

There is some additional workload in the forward pass to support it, and it doesn't vectorize as nicely as previous cases, but it might still be a win for some architectures.

However, with this approach the training needs to be aware of this and try to create those blocks of 0 weights without harming the network too much. This can be achieved with weight pruning, which will be described later. The inference code will be very similar to the linear layer with sparse inputs case.


```cpp
void load_weights(
    const LinearLayer& layer,
    const int8_t* data
) {
    // This goes the same as in the case with sparse inputs, however
    // the weights matrix is no longer continuous and we need to fill
    // some block indices to know which weights correspond to which ouputs.
    // This can be done either by discovering the zero blocks during loading,
    // or with a different serialized format with the block indices precomputed.
    // We will omit this here and just assume that layer.nnz_block_ids[input_id][4]
    // contains non-zero weight block indices corresponding to each input.
}

int32_t* linear_sparse_input_block_sparse_output(
    const LinearLayer& layer,
    int32_t*           output,
    const int8_t*      input
) {
    static_assert(is_same_v<LinearLayer::WeightType, int16_t>,
        "This approach requires weights to be 16 bit. Otherwise it's hard to widen the multiplication output to 32 bits.");

    constexpr int register_width = 256 / 8;
    constexpr int input_register_width = register_width; // uint8_t
    constexpr int output_register_width = register_width / 4; // int32_t
    constexpr int output_chunk_size = output_register_width * 2; // we will be processing 2 registers at a time
    assert(layer.num_outputs % output_chunk_size == 0, "We're processing 16 output elements at a time");
    assert(layer.num_inputs % input_register_width == 0);

    uint16_t nnz_input_indices[layer.num_inputs];
    int num_nnz_input_indices = 0;

    for (int i = 0; i < layer.num_inputs; i += input_register_width) {
        const __m256i input_chunk = _mm256_load_si256(input + i);
        uint32_t nnz =
            _mm256_movemask_epi8(
                _mm256_cmpgt_epi8(input_chunk, _mm256_setzero_si256())
            );

        while (nnz) {
            const int lsb_index = lsb(nnz);
            nnz &= nnz - 1; // reset the least significant set bit in nnz
            nnz_input_indices[num_nnz_input_indices++] = i + lsb_index;
        }
    }

    for (int i = 0; i < layer.num_outputs; ++i) {
        output[i] = layer.biases[i];
    }

    const int num_chunks = layer.num_outputs / output_chunk_size;
    // There are always tradeoffs. We cannot process two inputs at a time, because
    // they might have different non-zero weight blocks. Makes it visibly slower.
    // There might be some tricks with AVX512, but AVX2 is fairly limited for this use case.
    for (int i = 0; i < num_nnz_input_indices; ++i) {
        const int input_id = nnz_input_indices[i]
        const __m256i factor = _mm256_set1_epi32(input[input_id]);

        // We have hardcoded 4 16-wide non-zero weight blocks per input.
        for (int j = 0; j < 4; ++j) {
            const int block_id = layer.nnz_block_ids[input_id][j];
            const int output_offset0 = (block_id*2 + 0)*output_register_width;
            const int output_offset1 = (block_id*2 + 1)*output_register_width;

            const int weight_offset  = (block_id*1 + 0)*output_register_width;

            __m256i sum0 = _mm256_load_si256(output + output_offset0);
            __m256i sum1 = _mm256_load_si256(output + output_offset1);

            const __m256i col0 = _mm256_load_si256(
                layer.weights + input_id * layer.num_outputs + weight_offset
            );

            m256_process_chunk(sum0, sum1, col0, _mm256_setzero_si256(), factor);

            _mm256_store_si256(output + output_offset0, sum0);
            _mm256_store_si256(output + output_offset1, sum1);
        }
    }

    return output + layer.num_outputs;
}
```

#### ClippedReLU

The clipping is not hard, the more complicated part is conversion. We also need two version, one for int16 -> int8, and one for int32 -> int8.

##### int16 -> int8

![](img/crelu16.png)

```cpp
float* crelu16(,
    int            size,   // no need to have any layer structure, we just need the number of elements
    int8_t*        output, // the already allocated storage for the result
    const int16_t* input   // the input, which is the output of the previous linear layer
) {
    constexpr int in_register_width = 256 / 16;
    constexpr int out_register_width = 256 / 8;
    assert(size % out_register_width == 0, "We're processing 32 elements at a time");
    const int num_out_chunks = size / out_register_width;

    const __m256i zero    = _mm256_setzero_si256();
    const int     control = 0b11011000; // 3, 1, 2, 0; lane 0 is the rightmost one

    for (int i = 0; i < num_out_chunks; ++i) {
        const __m256i in0 = _mm256_load_si256(&input[i * in_register_width * 2 + 0]);
        const __m256i in1 = _mm256_load_si256(&input[i * in_register_width * 2 + 1]);

        const __m256i result =
            // packs changes the order, so we need to fix that with a permute
            _mm256_permute4x64_epi64(
                // clamp from below
                _mm256_max_epi8(
                    // packs saturates to 127, so we only need to clamp from below
                    _mm256_packs_epi16(in0, in1),
                    zero
                ),
                control
            );

        _mm256_store_si256(&output[i * out_register_width], result);
    }

    return output + size;
}
```

##### int32 -> int8

![](img/crelu32.png)

```cpp
float* crelu32(,
    int            size,   // no need to have any layer structure, we just need the number of elements
    int8_t*        output, // the already allocated storage for the result
    const int32_t* input   // the input, which is the output of the previous linear layer
) {
    constexpr int in_register_width = 256 / 32;
    constexpr int out_register_width = 256 / 8;
    assert(size % out_register_width == 0, "We're processing 32 elements at a time");
    const int num_out_chunks = size / out_register_width;

    const __m256i zero    = _mm256_setzero_si256();
    const __m256i control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    for (int i = 0; i < num_out_chunks; ++i) {
        const __m256i in0 =
            _mm256_packs_epi32(
                _mm256_load_si256(&input[i * in_register_width * 4 + 0]),
                _mm256_load_si256(&input[i * in_register_width * 4 + 1])
            );
        const __m256i in1 =
            _mm256_packs_epi32(
                _mm256_load_si256(&input[i * in_register_width * 4 + 2]),
                _mm256_load_si256(&input[i * in_register_width * 4 + 3])
            );

        const __m256i result =
            _mm256_permutevar8x32_epi32(
                _mm256_max_epi8(
                    _mm256_packs_epi16(in0, in1),
                    zero
                ),
                control
            );

        _mm256_store_si256(&output[i * out_register_width], result);
    }

    return output + size;
}
```

### Accounting for quantization in the trainer

Adding (quite aggressive) quantization has reduced the possible range of values for the weights and biases. We can, however, ignore the feature transformer and all biases, as they use large integer types and we don't ever expect to hit the limit. The problematic case are the int8 weights of the linear layer, which for example in Stockfish can only go to about 2 (activation range in 0..1). This is potentially a big problem, as the training can diverge from the quantized representation by more than just rounding. To prevent this from happening, it is necessary to somehow limit the range for these parameters inside the trainer. So far the easiest way of doing it is to modify the optimizer to clamp the values to the available range after each optimization step. These minimum and maximum values can be passed, for example when registering the optimizable parameters in the optimizer.

#### Inside the optimizer

One way to account for this is directly in the optimizer. This is nice because the clipping is applied directly after the step, but requires having access to the optimizer's source. For example:

```python
# The min/max constants are specific for the Stockfish quantization scheme.
train_params = [
    {'params' : [self.ft.weight, self.ft.bias], 'lr' : LR },
    {'params' : [self.l1.weight], 'lr' : LR, 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.l1.bias], 'lr' : LR },
    {'params' : [self.l2.weight], 'lr' : LR, 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.l2.bias], 'lr' : LR },
    {'params' : [self.output.weight], 'lr' : LR, 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
    {'params' : [self.output.bias], 'lr' : LR },
]
optimizer = ranger.Ranger(train_params, betas=(.9, 0.999), eps=1.0e-7)
```

and then in the optimizer:

```python
class Ranger(Optimizer):
    def __init__([...]):
        [...]
        defaults = dict([...]
                        min_weight=None, max_weight=None)

def step(self, closure=None):
    [...]

    for group in self.param_groups:
        for p in group['params']:
            ...
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            if min_weight is not None and max_weight is not None:
                p_data_fp32.clamp_(min_weight, max_weight)
```

#### Outside the optimizer

Alternatively one can do it outside the optimizer for more flexibility:

```python
# The min/max constants are specific for the Stockfish quantization scheme.
self.weight_clipping = [
    {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.output.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
]
```

```python
# and call this in some step function
def _clip_weights(self):
    for group in self.weight_clipping:
        for p in group['params']:
            p_data_fp32 = p.data
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)
```

#### Accounting for virtual layers (factorization)

Sometimes more complex architectures make some layers' parameters be a sum of two layers during training. Just like feature factorization but for whole layers (see for example [this](#multiple-psqt-outputs-and-multiple-subnetworks)). We can account for example like this:

```python
# The min/max constants are specific for the Stockfish quantization scheme.
self.weight_clipping = [
    {'params' : [self.l1.weight], 'min_weight' : -127/64, 'max_weight' : 127/64, 'virtual_params' : self.some_virtual_factor.weight },
    {'params' : [self.l2.weight], 'min_weight' : -127/64, 'max_weight' : 127/64 },
    {'params' : [self.output.weight], 'min_weight' : -127*127/9600, 'max_weight' : 127*127/9600 },
]
```

```python
def _clip_weights(self):
    for group in self.weight_clipping:
        for p in group['params']:
            p_data_fp32 = p.data
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            if 'virtual_params' in group:
                virtual_params = group['virtual_params']
                # The virtual layer is usually N times smaller
                xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                expanded_virtual_layer = virtual_params.repeat(xs, ys)
                min_weight_t = p_data_fp32.new_full(p_data_fp32.shape, min_weight) - expanded_virtual_layer
                max_weight_t = p_data_fp32.new_full(p_data_fp32.shape, max_weight) - expanded_virtual_layer
                p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
            else:
                p_data_fp32.clamp_(min_weight, max_weight)
            p.data.copy_(p_data_fp32)
```

## Optimizing the trainer (CUDA)

### Using custom CUDA kernels

How to run our own kernel? Don't we need a complicated setup with the CUDA compiler and all that? CuPy to the rescue. CuPy is a python library that allows easy creation of CUDA kernels using plain python strings containing the CUDA code. CuPy handles compilation and everything else for us. For example:

```python
import cupy as cp

# Create the kernel
kernel = cp.RawKernel(r'''
void kernel_name(...) {
    // your usual kernel code
}
''', 'kernel_name')

# Optionally compile it, otherwise it will compile on first use
kernel.compile()

# Run the kernel
kernel(
    grid=(batch_size,), # The grid shape
    block=(num_threads,), # The block shape
    args=(...) # The arguments that are passed to the kernel
)
```

PyTorch tensors can be easly passed to the kernel by using `.data_ptr()`, which results the pointer to the tensor. One must however ensure that the memory is contiguous.

### Feature transformer

Up until now we've using pytorch's sparse matrix multiplication for the feature transformer, but their implementation is not great, and we have additional assumptions that we can use.

1. We have an upper bound on the nnz elements for each position.
2. We have large batches

We can therefore replace the feature indices from a 2d tensor of shape `[total_num_active_features, 2]`, which contains the position index and feature index for each value, to a 2d tensor of shape `[batch_size, max_num_features]`, which contains one feature index per one values and the position index is known. We need to somehow handle positions where the number of features is lower than `max_num_features`, so we'll pad the rows with `-1`, these will be omitted by the kernel. This obviously also requires modifying the data loader, but it'll be simpler now.

#### Data loader

```cpp
struct SparseBatch {
    SparseBatch(const std::vector<TrainingDataEntry>& entries) {
        size = entries.size();

        max_active_features = MAX_ACTIVE_FEATURES;

        stm = new float[size];
        score = new float[size];

        // New layout for the indices, now it's [size][MAX_ACTIVE_FEATURES].
        // Also we don't need to sort the indices because the new implementation
        // is fast regardless of the order!
        white_features_indices = new int[size * MAX_ACTIVE_FEATURES];
        black_features_indices = new int[size * MAX_ACTIVE_FEATURES];

        fill(entries);
    }

    void fill(const std::vector<TrainingDataEntry>& entries) {
        ...
    }

    int size;
    int max_active_features;

    float* stm;
    float* score;
    int* white_features_indices;
    int* black_features_indices;

    ~SparseBatch()
    {
        delete[] stm;
        delete[] score;
        delete[] white_features_indices;
        delete[] black_features_indices;
    }
};
```

and in python

```python
class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('stm', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('white_features_indices', ctypes.POINTER(ctypes.c_int)),
        ('black_features_indices', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self, device):
        # This is illustrative. In reality you might need to transfer these
        # to the GPU. You can also do it asynchronously, but remember to make
        # sure the source lives long enough for the copy to finish.

        stm_t = torch.from_numpy(np.ctypeslib.as_array(self.stm, shape=(self.size, 1)))
        score_t = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1)))

        # Now we don't have to bother with the sparse pytorch tensors!
        # And no transpositions required too because we have control over the layout!
        white_features_indices_t = torch.from_numpy(np.ctypeslib.as_array(self.white_features_indices, shape=(self.size, self.max_active_features)))
        black_features_indices_t = torch.from_numpy(np.ctypeslib.as_array(self.black_features_indices, shape=(self.size, self.max_active_features)))

        # The values are all ones, so we can create these tensors in place easly.
        # No need to go through a copy.
        white_features_values_t = torch.ones(self.num_active_white_features)
        black_features_values_t = torch.ones(self.num_active_black_features)

        # No more coalescing! Our implementation will be fast regardless of whether the inputs are sorted or not!
        return white_features_indices_t, white_features_values_t, black_features_indices_t, black_features_values_t, stm_t, score_t

# Let's also tell ctypes how to understand this type.
SparseBatchPtr = ctypes.POINTER(SparseBatch)
```

#### Forward

Now let's try to write a custom CUDA kernel. At this point you should have a good understanding of how the feature transformer works and how to implement it. We will need two kernels, one for forward, and one for backward pass. We'll write these kernels in a generic way that uses values, but for some uses it can of course be assumed that all values are 1. It'll be the easiest to present the kernels with notes in the comments:

```Cuda
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__

/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: feature_indices
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing indices of active features for each position
        in a batch. Feature index of -1 means that the slot is empty
        and the weights will not be accumulated for it. Moreover
        no further indices from this block will be considered.
        The indices form an implicit matrix of shape
        (BATCH_SIZE, NUM_INPUTS), where the first dimension index is
        inferred from the memory location (BATCH_SIZE), and the
        second dimension index is stored in the feature_indices matrix.
        The type for feature indices is int32_t.

    @param: feature_values
        A matrix of shape (BATCH_SIZE, max_active_features)
        containing the values (arity) of the corresponding
        feature index in feature_indices.
        The type for the feature value (arity) is float32.

    @param: weight
        The weight matrix of shape (NUM_INPUTS, output_size).
        Weights must be of type float32.

    @param: bias
        The bias vector of shape (output_size,).
        Bias values must be of type float32.

    @param: output
        An output matrix of shape (BATCH_SIZE, output_size).
        It may not be initialized, bias is always copied
        to the output first.
        Output values must have type float32.

    @const: max_active_features
        The maximum number of features that are active
        (non-zero) for a single position. This value determines
        the shape of the inputs.
        This value is of type uint32_t.

    @const: output_size
        The number of outputs. Must match the shape of weights
        and biases.
        This value is of type uint32.

    @const: output_thread_slice_size
        The number of outputs to process per thread. Must be output_size/num_threads.
        Equivalent to output_size/threadDim.x, but computing it at runtime is wasteful.
        This value is of type uint32.
*/

void feature_transformer_slice_forward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
    const float*   const weight,
    const float*   const bias,
          float*   const output
) {
    // The idea is to process one position per CUDA block, and in each block
    // there will be N threads, each working on some slice of the output.

    // These values are constant to allow more optimization.
    // Since with CuPy we have JIT compilation for free these
    // values can be for example set by string interpolation
    // whenever a specifically parameterized kernel is needed.
    const uint32_t       max_active_features      = ...;
    const uint32_t       output_thread_slice_size = ...;
    const uint32_t       output_size              = ...;

    // We get some memory that is shared between all threads.
    // In theory we don't access it between threads, so this could
    // be local, but arrays defined without __shared__ are
    // placed in the global memory which might be slower, and
    // we'd have to rely on the compiler to optimize it.
    __shared__
          float          shared_output[output_size];

    // 1 block is 1 position
    const uint32_t       block_idx           = blockIdx.x;
    // Each thread processes only a small number of outputs for a position.
    const uint32_t       slice_offset        = threadIdx.x * output_thread_slice_size;

    // Each thread fills only a small number of outputs for a position.
    // Here we calculate the offset into the output [batch_size, output_size] array
    // where we need to put the results from this thread.
          float*   const output_slice        = output + block_idx * output_size + slice_offset;
    // And other similar stuff.
    const float*   const bias_slice          = bias                             + slice_offset;
          float*         shared_output_slice = shared_output                    + slice_offset;

    // When we were using the pytorch's sparse matrices we needed to put 2 indices per value,
    // they were the position index and the feature index. Now we're exploting
    // our first assumption - we have a dense matrix of shape [batch_size, max_active_features],
    // and we only store one index per feature, the position index is known.
    const int32_t* const feature_index_row   = feature_indices + block_idx * max_active_features;
    const float*   const feature_value_row   = feature_values  + block_idx * max_active_features;

    #pragma unroll
    // Copy bias to the "local" memory.
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        shared_output_slice[s] = bias_slice[s];
    }

    // Each thread goes through all active features.
    for (uint32_t k = 0; k < max_active_features; ++k)
    {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        // We made it so that a feature index of -1 stops execution.
        // This condition is the same for all threads, so we can break early
        // and get some additional performance.
        if (feature_index != -1)
        {
            // Compute which weights we need to accumulate.
            const float* const weight_slice = weight + feature_index * output_size + slice_offset;
            #pragma unroll
            for (uint32_t s = 0; s < output_thread_slice_size; ++s)
            {
                // And accumulate the weights to the "local" memory.
                shared_output_slice[s] += weight_slice[s] * feature_value;
            }
        } else break;
    }

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        // Only at the end we put the results back into global memory.
        output_slice[s] = shared_output_slice[s];
    }
}
```

#### Backward

```Cuda
typedef unsigned int uint32_t;
typedef int int32_t;

extern "C" __global__
/*
    @assumptions:
        The blocks must have dimensionality (BATCH_SIZE,)
        The threads must have dimensionality (N,), where
        N * output_thread_slice_size == output_size.

    @param: weight_grad
        The weight gradient matrix of shape (NUM_INPUTS, output_size).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Weights must be of type float32.

    @param: bias_grad
        The bias gradient vector of shape (output_size,).
        The gradient is accumulated, i.e. it must be zero initialized
        on the first call.
        Bias values must be of type float32.

    @param: output_grad
        An output gradient matrix of shape (BATCH_SIZE, output_size).
        Output values must have type float32.
*/
void feature_transformer_slice_backward(
    const int32_t* const feature_indices,
    const float*   const feature_values,
          float*   const weight_grad,
          float*   const bias_grad,
    const float*   const output_grad
) {{
    // The helper indices and pointers we compute are very similar
    // to the forward pass, we're just going to be doing it backwards.
    const uint32_t       max_active_features      = ...;
    const uint32_t       output_thread_slice_size = ...;
    const uint32_t       output_size              = ...;

    // We don't really need to store this in the shared memory, because
    // it's almost surely cached, but since it's free and we do
    // use it many times in this kernel we might as well do it.
    __shared__
          float          shared_output_grad[output_size];

    const uint32_t       block_idx                = blockIdx.x;
    const uint32_t       slice_offset             = threadIdx.x * output_thread_slice_size;

    const float*   const output_grad_slice        = output_grad + block_idx * output_size + slice_offset;
          float*   const bias_grad_slice          = bias_grad                             + slice_offset;
          float*         shared_output_grad_slice = shared_output_grad                    + slice_offset;

    const int32_t* const feature_index_row        = feature_indices + block_idx * max_active_features;
    const float*   const feature_value_row        = feature_values  + block_idx * max_active_features;

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        // Copy the values to "local" memory to hopefully speed up the repeated access.
        shared_output_grad_slice[s] = output_grad_slice[s];
    }

    #pragma unroll
    for (uint32_t s = 0; s < output_thread_slice_size; ++s)
    {
        // x*w+b=y, so the bias gradient is just increased by the output gradient.
        const float sog = shared_output_grad_slice[s];
        // We expect this layer to come before a ClippedReLU so there will be a lot of zeros.
        // Also our kernel is completely memory bound, so we can utilize this to remove
        // redundant additions.
        if (sog != 0.0f)
        {
            // Due to how Nvidia GPUs work, since Kepler architecture, atomic
            // additions execute in specialized units that are closer to global memory.
            // Our access is mostly random, so be benefit here two-fold:
            // 1. atomicAdd executes **faster** than += because it's closer to memory
            // 2. we "rarely" have two atomic accesses to the same memory location.
            // We have to use atomic additions either way, because we're modifying
            // one gradient matrix (instead of multiple outputs as in the forward case),
            // so this is fortunate for us.
            atomicAdd(&bias_grad_slice[s], sog);
        }
    }

    // Same loop as in forward, but we accumulate the gradients now.
    for (uint32_t k = 0; k < max_active_features; ++k)
    {
        const int32_t feature_index = feature_index_row[k];
        const float   feature_value = feature_value_row[k];
        // Exit early after all active indices are processed.
        if (feature_index != -1)
        {
            float* const weight_grad_slice = weight_grad + feature_index * output_size + slice_offset;
            #pragma unroll
            for (int s = 0; s < output_thread_slice_size; ++s)
            {
                const float sog = shared_output_grad_slice[s];
                // Same optimization as in the case of the bias.
                if (sog != 0.0f)
                {
                    // x*w+b=y, so we accumulate output gradient multiplied by x (input).
                    atomicAdd(&weight_grad_slice[s], sog * feature_value);
                }
            }
        } else break;
    }
}
```

#### FeatureTransformerSlice layer

```python
class FeatureTransformerSliceFunction(autograd.Function):

    @staticmethod
    def forward(ctx, feature_indices, feature_values, weight, bias):
        # Save the required stuff for the backward pass.
        ctx.save_for_backward(feature_indices, feature_values, weight, bias)

        # A lot of assertions are needed to ensure the correctness.
        assert len(feature_indices.shape) == 2
        assert len(feature_values.shape) == 2
        assert feature_indices.shape[0] == feature_values.shape[0]
        assert feature_indices.shape[1] == feature_values.shape[1]
        assert feature_indices.dtype == torch.int32
        assert feature_values.dtype == torch.float32

        assert len(weight.shape) == 2
        assert weight.dtype == torch.float32

        assert len(bias.shape) == 1
        assert bias.dtype == torch.float32

        assert feature_indices.is_cuda
        assert feature_values.is_cuda
        assert weight.is_cuda
        assert bias.is_cuda

        assert feature_values.device == feature_indices.device
        assert weight.device == feature_indices.device
        assert bias.device == feature_indices.device

        assert feature_indices.is_contiguous()
        assert feature_values.is_contiguous()
        assert weight.is_contiguous()
        assert bias.is_contiguous()

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        output = torch.empty(batch_size, output_size, dtype=torch.float32, device=device, requires_grad=True)

        # Implementation for make_feature_transformer_slice_forward_kernel not provided. It could
        # for example dynamically create and cache the kernels.
        kernel, num_threads = make_feature_transformer_slice_forward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,), # One position per batch
            block=(num_threads,), # Number of threads per block as "advised" by the function above
            args=( # Pointers to all the tensors, we ensured they are contiguous.
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight.data_ptr(),
                bias.data_ptr(),
                output.data_ptr()
            )
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # We don't handle the gradient for the feature indices and values, so
        # make sure it's not required.
        assert not ctx.needs_input_grad[0]
        assert not ctx.needs_input_grad[1]

        grad_output = grad_output.contiguous()

        # Retrieve the saved tensors.
        feature_indices, feature_values, weight, bias = ctx.saved_tensors

        device = feature_indices.device
        batch_size = feature_indices.shape[0]
        max_active_features = feature_indices.shape[1]
        output_size = weight.shape[1]

        weight_grad = torch.zeros(weight.shape[0], weight.shape[1], dtype=torch.float32, device=device)
        bias_grad = torch.zeros(output_size, dtype=torch.float32, device=device)

        # Similar to the forward case
        kernel, num_threads = make_feature_transformer_slice_backward_kernel(max_active_features, output_size)
        kernel(
            grid=(batch_size,),
            block=(num_threads,),
            args=(
                feature_indices.data_ptr(),
                feature_values.data_ptr(),
                weight_grad.data_ptr(),
                bias_grad.data_ptr(),
                grad_output.data_ptr()
            )
        )

        # The order of returned values here is the same as the order of inputs to the forward pass.
        return None, None, weight_grad, bias_grad

class FeatureTransformerSlice(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FeatureTransformerSlice, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Initialize in the same way nn.Linear would be initialized.
        sigma = math.sqrt(1/num_inputs)
        self.weight = nn.Parameter(torch.rand(num_inputs, num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)
        self.bias = nn.Parameter(torch.rand(num_outputs, dtype=torch.float32) * (2 * sigma) - sigma)

    def forward(self, feature_indices, feature_values):
        # Use our FeatureTransformerSliceFunction for the forward pass.
        # Backward will automatically use the backward function from the FeatureTransformerSliceFunction class
        return FeatureTransformerSliceFunction.apply(feature_indices, feature_values, self.weight, self.bias)
```

#### Results

Using this custom kernel has improved performance by 2 to 4 times on some GPUs! Moreover, the slowdown from using outputs larger than 256 per slice is now much less!

## Architectures and new directions

### Simple HalfKP Stockfish architecture

This is the first architecture used in Stockfish. The only thing that is different is that in this document, we use HalfKP that doesn't have these 64 additional unused features that are a leftover from Shogi. Other than that, there's nothing new, just using the primitives described earlier to put things into perspective.

![](img/HalfKP-40960-256x2-32-32-1.png)

### HalfKAv2 feature set.

HalfKA feature set was briefly mentioned in this document as a brother of HalfKP. It initially had a small drawback that wasted some space. HalfKAv2 is the improved version that uses 8% less space, but otherwise is identical. What's the difference? Let's consider a subset of features for a given our king square `S`. Normally in HalfKA there are 768 possible features, that is `64*12`, as there is 64 squares and 12 pieces (type + color). But we can notice that with the our king square fixed at `S` we know that the opponent's king is not at `S` - our king uses just 1 feature from the 64 given for it, and the other king only uses 63 (minus our king ring, but it doesn't matter) from its 64 given features, and the two sets are disjoint! So we can merge the two pieces "into one", and reduce the number of buckets from 12 into 11, reducing the size by about 8%. However, care must be taken when applying factorization, as this compression needs to be reverted and a whole `A` subset with 768 features must be used. Otherwise it's possible to mix up king positions, as while the compression is valid for a single `64*11` bucket, it doesn't hold when we try to mix the buckets, as it happens when we factorize the features.

### A part of the feature transformer directly forwarded to the output.

Normally the nets have hard time learning high material imbalance, or even representing high evaluations at all. But we can help it with that. We already accumulate some 256 values for each piece on the board, does this ring a bell? What if we added one more and designated it to mean "PSQT"? That's what we will do. We will simply make the feature transformer weight row have 257 values, and use the last one as "PSQT". We can help it during training by initializing it to something that resembles good PSQT values (but remember to scale it according to quantization!). But we have two perspectives? What about that? Right, we do, but we can average them, like `(our - their) / 2` (keeping in mind that their must be negated). Handling it in the trainer is quite easy.

```python
wp = self.ft(w_in)
bp = self.ft(b_in)
w, wpsqt = torch.split(wp, wp.shape[1]-1, dim=1)
b, bpsqt = torch.split(bp, bp.shape[1]-1, dim=1)
[...]
y = self.output(l2_) + (wpsqt - bpsqt) * (us - 0.5)
```

We should also use a feature set that includes king features, as it provides additional PSQT values that may be important. So we will use HalfKAv2.

![](img/HalfKAv2-45056-256x2P1x2-32-32-1.png)

### Multiple PSQT outputs and multiple subnetworks

Until now all networks have been using one PSQT output and one layer stack (that -32-32-1 part in the Stockfish's network; whatever comes after the feature transformer). But what if we could use more? We need to find some easy-to-compute discriminator to choose the outputs/layer stacks by. One such good discriminator is the piece count, as it's cheap to compute, fairly well behaved during the game, and the number of pieces can dramatically change how we look at the position. So let's try 8 buckets for both, based on `(piece_count - 1) / 4`.

![](img/HalfKAv2-45056-256x2P8x2[-32-32-1]x8.png)

But how to implement it in the trainer? "Choosing stuff" is not very GPU friendly, and we're doing batching too, right? It's not indeed, but thankfully the layers are very small, so we can just evaluate all of them and only choose the results! Moreover, multiple `N` linear layers can just be emulated by a single one with `N` times as many outputs. Let's see how it could be implemented in PyTorch:

```python
# Numbers of hidden neurons
L1 = 256
L2 = 32
L3 = 32

class LayerStacks(nn.Module):
    def __init__(self, count):
        super(LayerStacks, self).__init__()

        self.count = count
        # Layers are larger, very good for GPUs
        self.l1 = nn.Linear(2 * L1, L2 * count)
        self.l2 = nn.Linear(L2, L3 * count)
        self.output = nn.Linear(L3, 1 * count)

        # For caching some magic later.
        self.idx_offset = None

        # Don't forget to initialize the layers to your liking.
        # It might be worth it to initialize the layers in each layer
        # stack identically, or introduce a factorizer for the first
        # layer in the layer stacks.

    def forward(self, x, layer_stack_indices):
        # Precompute and cache the offset for gathers
        if self.idx_offset == None or self.idx_offset.shape[0] != x.shape[0]:
            # This is the "magic". There's no easy way to gather just one thing out of
            # many for each position in the batch, but we can interpret the whole batch as
            # N * batch_size outputs and modify the layer_stack_indices to point to
            # `N * i + layer_stack_indices`, where `i` is the position index.
            # Here we precompute the additive part. This part includes just the values `N * i`
            self.idx_offset = torch.arange(0, x.shape[0] * self.count, self.count, device=layer_stack_indices.device)

        # And here we add the current indices to the additive part.
        indices = layer_stack_indices.flatten() + self.idx_offset

        # Evaluate the whole layer
        l1s_ = self.l1(x)
        # View the output as a `N * batch_size` chunks
        # Choose `batch_size` chunks based on the indices we computed before.
        l1c_ = l1s_.view(-1, L2)[indices]
        # We could have applied ClippedReLU earlier, doesn't matter.
        l1y_ = torch.clamp(l1c_, 0.0, 1.0)

        # Same for the second layer.
        l2s_ = self.l2(l1y_)
        l2c_ = l2s_.view(-1, L3)[indices]
        l2y_ = torch.clamp(l2c_, 0.0, 1.0)

        # Same for the third layer, but no clamping since it's the output.
        l3s_ = self.output(l2y_)
        l3y_ = l3s_.view(-1, 1)[indices]

        return l3y_
```

Handling of the PSQT outputs is easier since the is in fact, a simple way of gathering individual values (we couldn't use it above because we were gathering whole rows):

```python
wp = self.input(w_in)
bp = self.input(b_in)
w, wpsqt = torch.split(wp, wp.shape[1]-8, dim=1)
b, bpsqt = torch.split(bp, bp.shape[1]-8, dim=1)
[...]
psqt_indices_unsq = psqt_indices.unsqueeze(dim=1)
wpsqt = wpsqt.gather(1, psqt_indices_unsq)
bpsqt = bpsqt.gather(1, psqt_indices_unsq)
y = self.layer_stacks(l0_, layer_stack_indices) + (wpsqt - bpsqt) * (us - 0.5)
```
