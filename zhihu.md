## TL;DR

-   2024 Meta 发布的 30B 参数量的视频生成模型 Movie Gen 以及 13B 参数量的 Movie Gen Audio 视频配乐模型，支持生成高保真图像、视频和音频，同时也具备编辑和个性化视频的能力。Movie Gen 的技术报告是视频生成领域目前介绍技术细节最详细的文章。

**Paper name**

Movie Gen: A Cast of Media Foundation Models

**Paper Reading Note**

Paper URL: https://ai.meta.com/static-resource/movie-gen-research-paper

Project URL: https://ai.meta.com/research/movie-gen/

Blog URL: https://ai.meta.com/blog/movie-gen-media-foundation-models-generative-ai-video/

Demo video URL: https://www.youtube.com/playlist?list=PL86eLlsPNfyi27GSizYjinpYxp7gEl5K8

---

## Introduction

### 背景

-   视频生成实际上是个非常复杂的任务。比如说人类来想象鸟在海洋中游泳，人类是有惊人的能力来支持快速在大脑中想象出这种画面的。这需要组合和预测运动、场景、物理、几何、音频的真实属性。具备这种生成、组合和预测能力的人工智能系统设计是很有挑战的。

### 本文方案

-   介绍了Movie Gen，一系列媒体生成基础模型。Movie Gen 模型可以本地生成高保真图像、视频和音频，同时也具备编辑和个性化视频的能力

    -   最大的模型结构是 30B 参数 transformer
    -   73k token 的最大上下文长度
    -   生成 16s + 16fps 的视频
    -   生成 48kHz 的音频
    -   多任务支持
        -   文生视频
        -   视频个性化（人脸 condition，保 ID）
        -   视频编辑
        -   视频生成音频
        -   文生音频

-   文生视频

![](https://i-blog.csdnimg.cn/direct/73b90559e0844f83949b55ff4a59987c.png)

-   视频个性化 + 人物一致性

![](https://i-blog.csdnimg.cn/direct/a24ffbaeb406436b9367ab1f90774fce.png)

-   instruction-guided 视频编辑

![](https://i-blog.csdnimg.cn/direct/b1e517ad00a2462c93557bbf963f92a8.png)

-   视频生成音频

![](https://i-blog.csdnimg.cn/direct/11b61922f5444884899ad4084ad1a20a.png)

## Methods

### 整体概览

-   发布了两个基础模型
    -   Movie Gen Video：
        -   30B
        -   text-to-image 和 text-to-video 联合训练
        -   多长宽比(例如 1:1, 9:16, 16:9)、分辨率(768px)、时长(4-16s)支持
        -   预训练：100M video 和 1B image 训练集学习 visual world
        -   SFT：小规模的高质量视频和文本描述

    -   Movie Gen Audio
        -   13B
        -   video- and text-to-audio generation
        -   48kHz
        -   可变长度音频生成，通过音频扩展可以为长达几分钟的视频生成格式连贯的音频
        -   预训练：1M hour 音频。学习物理关联，以及视觉和音频世界之间的心理关联
        -   SFT: (text, audio) and (video, text, audio) 匹配对数据

-   后训练支持
    -   个性化视频生成：
        -   用人照片作为 condition，创建由该人出演的视频
        -   训练数据集：(image, text) 和 video 的匹配数据

    -   视频编辑：
        -   因为大规模有监督的视频编辑数据难以获取。不用有监督数据进行训练

### 图像视频联合生成

-   训练 recipe 如下。使用多阶段训练：
    -   训练 256px 文生图
    -   256-768 px 文生图 + 文生视频
    -   768px 个性化视频 + 文生视频 + 视频编辑

![](https://i-blog.csdnimg.cn/direct/ee6c767d62b847d3ad9bf9e21fba2e76.png)

-   整体模型架构
    -   使用 temporal autoencoder model (TAE) 将图片、视频映射到 latent space
    -   三个 pre-trainded text-encoder 用于提取文本 condition
    -   llama3 结构的 transformer 学习将 gaussian noise 映射到 TAE 的 latent 空间
    -   使用 temporal autoencoder model (TAE) 将 transformer 处理后的 latent 映射回图像、视频空间

![](https://i-blog.csdnimg.cn/direct/119c548f701a4a33996e57260c09254f.png)

#### Temporal Autoencoder (TAE)

-   T3HW 下采样的到 T'CH'W'
    -   每个维度上下采样 8 倍。T 上下采样倍数设置为 8 是比较高的选择，相比于目前开源大部分选择 4，可以方便不用后续的插帧模型

-   从 SD 拿的模型结构，inflate 为 3D 模型。方式是在每个 2D 空间卷积后加一个 temporal 卷积，2D attention 后加 1D temporal attention
    -   temporal conv 使用 symmetrical replicate padding
    -   下采样通过 strided conv （意味着任意长度的视频都可以被 encode，通过丢掉虚假输出帧）

![](https://i-blog.csdnimg.cn/direct/bf48846557b445f0ad4dfd029b26709f.png)

-   上采样通过 nearest 插值 + conv
-   channel 数使用 16，参考 Emu 中的结论，增加 channel 能同时提升重建和生成的效果
-   初始化使用一个预训练的 image encoder，然后 batch 比例 video: image=3: 1 来 interleave 联合训练
-   优化训练目标，降低 'spot' artifact （某些 latent space 中有高范数的点，产生了对应的像素空间中的高亮），猜测是 shortcut learning 导致的，模型在这些高范数的 latent 点中存储关键的全局信息。之前的工作发现去除 group norm 能解决这个问题。本文使用 loss 约束来缓解该问题

![](https://i-blog.csdnimg.cn/direct/0719f1c85af845adb5c07b13b2fbb27f.png)
添加了 outlier penalty loss (OPL) ，惩罚编码远离平均值的潜在值：
![](https://i-blog.csdnimg.cn/direct/f6ce15c97e1b4a4cbe1dbaea0bb2afc8.png)
其他的 loss 就是 autoencoder 的标准损失：(reconstruction, discriminator, and perceptual)

-   使用时间平铺进行有效的推理
    -   由于显存限制，1024px + 256 frame 的直接推理不可行。
    -   我们将输入视频和潜在张量沿时间维度划分为瓦片，编码和/或解码每个瓦片，并将结果拼接在输出处。切片时需要有一定 overlap，然后 merge 不同切片时使用 blending 操作用来缓解 boundary artifacts
        -   tile size 32 帧 (latent 4 帧)
        -   encoder 不 overlap
        -   decoder overlap 16 帧 (latent 2 帧)
        -   线性权重融合

![](https://i-blog.csdnimg.cn/direct/c3986d224e354320aa1686b7949e9ac1.png)

#### 训练损失

-   使用 Flow Matching 框架
    -   Flow Matching通过迭代改变样本来生成目标数据分布的样本，例如从先验分布（如高斯分布）中生成样本。在训练过程中，给定一个视频样本的潜在空间表示 <img src="https://www.zhihu.com/equation?tex=%5Cmathbf%7BX%7D_1" alt="\mathbf{X}_1" class="ee_img tr_noresize" eeimg="1">，我们会选择一个时间步长 $t \in [0, 1]<img src="https://www.zhihu.com/equation?tex=%EF%BC%8C%E5%B9%B6%E4%BB%8E%20" alt="，并从 " class="ee_img tr_noresize" eeimg="1">\mathcal{N}(0,1)<img src="https://www.zhihu.com/equation?tex=%20%E4%B8%AD%E9%87%87%E6%A0%B7%E4%B8%80%E4%B8%AA%E2%80%9C%E5%99%AA%E5%A3%B0%E2%80%9D%E6%A0%B7%E6%9C%AC%20" alt=" 中采样一个“噪声”样本 " class="ee_img tr_noresize" eeimg="1">\mathbf{X}_0<img src="https://www.zhihu.com/equation?tex=%EF%BC%8C%E7%84%B6%E5%90%8E%E4%BD%BF%E7%94%A8%E5%AE%83%E4%BB%AC%E6%9D%A5%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E8%AE%AD%E7%BB%83%E6%A0%B7%E6%9C%AC%20" alt="，然后使用它们来构建一个训练样本 " class="ee_img tr_noresize" eeimg="1">\mathbf{X}_t<img src="https://www.zhihu.com/equation?tex=%E3%80%82%E6%A8%A1%E5%9E%8B%E8%A2%AB%E8%AE%AD%E7%BB%83%E6%9D%A5%E9%A2%84%E6%B5%8B%E9%80%9F%E5%BA%A6%20" alt="。模型被训练来预测速度 " class="ee_img tr_noresize" eeimg="1">\mathbf{V}_t = \frac{d\mathbf{X}_t}{dt}<img src="https://www.zhihu.com/equation?tex=%EF%BC%8C%E8%BF%99%E6%95%99%E4%BC%9A%E6%A8%A1%E5%9E%8B%E5%B0%86%E6%A0%B7%E6%9C%AC%20" alt="，这教会模型将样本 " class="ee_img tr_noresize" eeimg="1">\mathbf{X}_t<img src="https://www.zhihu.com/equation?tex=%20%E5%90%91%E8%A7%86%E9%A2%91%E6%A0%B7%E6%9C%AC%20" alt=" 向视频样本 " class="ee_img tr_noresize" eeimg="1">\mathbf{X}_1<img src="https://www.zhihu.com/equation?tex=%20%E7%9A%84%E6%96%B9%E5%90%91%E2%80%9C%E7%A7%BB%E5%8A%A8%E2%80%9D%E3%80%82%E5%9C%A8%E6%9E%84%E5%BB%BA%20" alt=" 的方向“移动”。在构建 " class="ee_img tr_noresize" eeimg="1">\mathbf{X}_t$ 时，可以使用许多方法，但在本文的工作中，使用了简单的线性插值或最优传输路径，即：

<img src="https://www.zhihu.com/equation?tex=%5Cmathbf%7BX%7D_t%20%3D%20t%20%5Cmathbf%7BX%7D_1%20%2B%20%281%20-%20%281%20-%20%5Csigma_%7B%5Ctext%7Bmin%7D%7D%29t%29%20%5Cmathbf%7BX%7D_0%5C%5C" alt="\mathbf{X}_t = t \mathbf{X}_1 + (1 - (1 - \sigma_{\text{min}})t) \mathbf{X}_0\\" class="ee_img tr_noresize" eeimg="1">
其中 <img src="https://www.zhihu.com/equation?tex=%5Csigma_%7B%5Ctext%7Bmin%7D%7D%20%3D%2010%5E%7B-5%7D" alt="\sigma_{\text{min}} = 10^{-5}" class="ee_img tr_noresize" eeimg="1"> 。因此，真实速度可以表示为：
<img src="https://www.zhihu.com/equation?tex=%5Cmathbf%7BV%7D_t%20%3D%20%5Cfrac%7Bd%5Cmathbf%7BX%7D_t%7D%7Bdt%7D%20%3D%20%5Cmathbf%7BX%7D_1%20-%20%281%20-%20%5Csigma_%7B%5Ctext%7Bmin%7D%7D%29%5Cmathbf%7BX%7D_0" alt="\mathbf{V}_t = \frac{d\mathbf{X}_t}{dt} = \mathbf{X}_1 - (1 - \sigma_{\text{min}})\mathbf{X}_0" class="ee_img tr_noresize" eeimg="1">
表示模型参数为 <img src="https://www.zhihu.com/equation?tex=%5Ctheta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 和文本提示嵌入 <img src="https://www.zhihu.com/equation?tex=%5Cmathbf%7BP%7D" alt="\mathbf{P}" class="ee_img tr_noresize" eeimg="1"> ，预测速度表示为 <img src="https://www.zhihu.com/equation?tex=u%28%5Cmathbf%7BX%7D_t%2C%20%5Cmathbf%7BP%7D%2C%20t%29" alt="u(\mathbf{X}_t, \mathbf{P}, t)" class="ee_img tr_noresize" eeimg="1"> 。模型通过最小化真实速度和模型预测之间的均方误差进行训练：
$$
\mathbb{E}_{t, \mathbf{X}_0, \mathbf{X}_1, \mathbf{P}} \left| u(\mathbf{X}_t, \mathbf{P}, t; \theta) - \mathbf{V}_t \right|^2
$<img src="https://www.zhihu.com/equation?tex=%E4%B8%8E%E4%B9%8B%E5%89%8D%E7%9A%84%E5%B7%A5%E4%BD%9C%E7%B1%BB%E4%BC%BC%EF%BC%8C%E4%BB%8E%E5%AF%B9%E6%95%B0%E6%AD%A3%E6%80%81%E5%88%86%E5%B8%83%E4%B8%AD%E9%87%87%E6%A0%B7%20" alt="与之前的工作类似，从对数正态分布中采样 " class="ee_img tr_noresize" eeimg="1">t$ ，其中底层高斯分布的均值为零，标准差为1。

#### 推理

**推理过程** 在推理阶段，我们首先从标准正态分布 <img src="https://www.zhihu.com/equation?tex=%5Cmathcal%7BN%7D%280%2C1%29" alt="\mathcal{N}(0,1)" class="ee_img tr_noresize" eeimg="1"> 中采样得到 <img src="https://www.zhihu.com/equation?tex=X_0" alt="X_0" class="ee_img tr_noresize" eeimg="1">，然后使用常微分方程（ODE）求解器根据模型估计的 <img src="https://www.zhihu.com/equation?tex=%5Cfrac%7BdX_t%7D%7Bdt%7D" alt="\frac{dX_t}{dt}" class="ee_img tr_noresize" eeimg="1"> 值计算 <img src="https://www.zhihu.com/equation?tex=X_1" alt="X_1" class="ee_img tr_noresize" eeimg="1">。在实际操作中，ODE 求解器的配置存在多个设计选择，例如，一阶或高阶求解器、步长、容差等，这些都会影响估计的 <img src="https://www.zhihu.com/equation?tex=X_1" alt="X_1" class="ee_img tr_noresize" eeimg="1"> 的运行时间和精度。我们使用一个简单的一阶欧拉 ODE 求解器，并根据我们的模型定制了一组独特的离散时间步长 <img src="https://www.zhihu.com/equation?tex=N" alt="N" class="ee_img tr_noresize" eeimg="1">。

**信噪比** 时间步长 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> 控制信噪比（SNR），我们用于构建 <img src="https://www.zhihu.com/equation?tex=X_t" alt="X_t" class="ee_img tr_noresize" eeimg="1"> 的简单插值方案确保了当 <img src="https://www.zhihu.com/equation?tex=t%20%3D%200" alt="t = 0" class="ee_img tr_noresize" eeimg="1"> 时信噪比为零。这确保了在训练过程中，模型接收到的是纯高斯噪声样本，并训练模型预测这些噪声的速度。因此，在推理过程中，当模型在 <img src="https://www.zhihu.com/equation?tex=t%20%3D%200" alt="t = 0" class="ee_img tr_noresize" eeimg="1"> 时接收到纯高斯噪声时，它可以做出合理的预测。

大多数视频生成模型都是使用扩散公式进行训练的。最近的工作表明，选择具有零终端信噪比的正确扩散噪声调度对于视频生成尤为重要。标准的扩散噪声调度不能保证零终端信噪比，因此需要为视频生成目的进行修改。如上所述，我们的流匹配实现自然地保证了零终端信噪比。经验上，我们发现流匹配对噪声调度的确切选择更具鲁棒性，并且优于扩散损失。因此，我们采用流匹配，因为它简单且性能高。

### 模型架构

-   patchify
    -   对于 T3HW 的视频输入，经过 TAE encode 到 TCHW，利用一个 3D convolution layer 进行 patchify 操作，然后 flatten 到一维向量。
        -   patchify 的 kt=1，kh=2 和 kw=2

-   位置编码
    -   分解的可学习位置嵌入，以实现任意大小、长宽比和视频长度支持。每个 THW 的维度上都分别设置一个 [0, maxlen] 的 D 维位置编码
    -   每层都加。与仅将位置嵌入添加到第一层相比，将位置嵌入添加到所有层可以有效地减少失真和变形伪影，尤其是在时间维度上。

-   transformer 结构使用 llama3，另外有三个修改
    -   增加了 cross-attention 用于接入文本 prompt embedding。插在 self-attention 和 ffn 之间。使用了多个有互补能力的 text encoder，embedding 简单 concat 起来
    -   增加了 DiT 中的 adaln，用于编码 time-step t
    -   causul attention -> full attention

![](https://i-blog.csdnimg.cn/direct/f0c702b2abea4294872fa9c411ff00a4.png)

#### 丰富的文本嵌入和视觉文本生成

-   使用了三个 text encoder
    -   UL2。
        -   prompt-level。纯文本训练，提供强大的文本推理能力。

    -   ByT5
        -   character-level。用于编码视觉文本，即文本提示的一部分，可以明确要求在输出中生成字符字符串

    -   Long-prompt MetaCLIP。在 MetaCLIP 基础上用 long text caption finetune 得到，使得支持的长度更长 77 token -> 256 token。
        -   prompt-level。提供有利于跨模态生成的视觉表示对齐的文本表示。

-   融合方式
    -   增加一个 separate linear projection，经过 LN
    -   然后 concat

#### FPS 控制

-   在每个训练 video 后面添加一个包含 fps 的文本，例如 FPS-16。
-   预训练阶段，进行 fps 采样，最小 fps 是 16；微调阶段，采样 fps 固定使用 16 或者 24

### 空间上采样（超分辨率）

-   768px -> 1080p。降低高分辨率生成的总体计算成本。因为基本的文本到视频模型处理的 token 更少。整体流程如下：
    -   低分辨率 video 先空间上采样到目标分辨率，使用 bilinear 插值。
    -   使用 vae 投影到 latent space。使用 frame-wise VAE，为了提升 pixel 的 sharpness
    -   latent space model 对 latent 进行处理。
    -   然后给 vae 解码成高清视频

![](https://i-blog.csdnimg.cn/direct/dcb7774ff55c4725a6e4aac2ed0aee48.png)

-   实现细节
    -   超分模型结构和 text-to-video transformer 模型一致，参数量降低到 7B，使用在 1024px text-to-image 上预训练的模型进行初始化
    -   encoded video 通过和生成输入 (zero-initialized)进行 channel-wise concat 输入超分模型
    -   400K 高清 video 上训练，14 帧的 clip，24 fps
    -   使用 REAL-ESRGAN 里面的 second-order degradation with artifacts 对 TAE 的 latent 进行增广
    -   20 推理 step 就足够了
    -   这种设计可以用于各种倍率的超分，不够本文训练的是 2x 空间超分
    -   类似 TAE tiling，这里使用 sliding window (window size 14) 进行 latent 超分，overlap 4 latent
        -   使用 MultiDiffusion，traning-free 的优化来降低融合 boundry 上的不连续性

### Model scaling 和训练效率

-   6144 H100
-   RoCE 通信
-   full attention，暂时没有探索 GQA 之类的
-   768px 256 frame 的 token 数是 (768/16)**2*(256/8)=73728

#### 模型并行

-   3D parallelism，在三个模型层面进行 scaling
    -   模型参数
    -   输入 token
    -   数据集

-   并行包括以下
    -   **Fully Sharded Data Parallel (FSDP)**：在 data-parallel gpu 上 shard 模型、优化器和梯度。每个训练步骤中都同步 gathering 和 scattering 参数、梯度
    -   **Tensor-parallelism (TP)**：将线性层的权重沿列或行分割，导致每个参与分割的GPU执行的计算量（FLOPs）减少 tp 大小，对于列并行分割生成的激活减少 tp 大小，对于行并行分割消耗的激活也减少 tp 大小。进行这种分割的代价是在前向（行并行）和后向（列并行）传递中增加了全归约通信开销。
    -   **Sequence-parallelism (SP)**：基于张量并行（TP），进一步允许在序列维度上对输入进行切片，用于那些复制的层，并且这些层中的每个序列元素可以独立处理。这样的层，例如 LayerNorm，否则将进行重复计算，并在 TP 组内生成相同（因此是复制的）激活。
    -   **Context-parallelism (CP)**：允许在序列依赖的softmax-attention操作中，对序列维度进行部分切片。CP 利用了这样一个见解：对于任意给定的（源（上下文 context）、目标（查询 query））序列对，softmax-attention 只依赖于 context 序列，而不是 query 序列。因此，在自注意力（self-attention）的情况下，输入的源序列和目标序列相同，CP 允许只对 K 和 V projections 进行 all-gather 操作（而不是 Q、K 和 V）来完成前向传播，并在反向传播中对它们的相关梯度进行 reduce-scatter。此外，由于 Q 与 K、V projections 行为的分离，CP 的性能不仅取决于 context 长度，还取决于上下文维度的大小。其结果是，CP 在不同任务中的扩展性能和开销特性有所不同，例如 Movie Gen Video 和最新的 LLM（如使用 GQA 技术的 LLaMa3），它们生成的 K、V 张量更小（例如，LLaMa3 70B 的 K、V 张量小 8 倍）。

### 预训练

#### 预训练数据

-   图-文匹配数据使用和 Emu 类似的数据准备方案。这里只介绍图-视频匹配数据的准备

-   预训练数据

    -   4s-16s
    -   single-shot, non-trivial motion

-   数据清洗流程

    -   视觉过滤
        -   移除最小长宽小于 720px 的视频
        -   过滤 aspect ratio 实现： 60% 横向 + 40% 竖向。横向的时长更长，美学更好，运动更稳定
        -   video OCR 过滤文本很多的视频
        -   ffmpeg 做 scene boundary detection
        -   移除视频的开头的几秒，可以避免一些不稳定的相机运动和过渡

    -   运动过滤
        -   过滤低运动视频
            -   移除完全不动的视频
            -   VMAF motion scores and motion vectors
            -   移除 PPT 之类的

    -   内容过滤
        -   copy-detection embedding 去除重复的视频
        -   基于 video-text embedding 进行重采样，将相似的聚类并调整采样概率

    -   caption
        -   LLaMa3-Video。70% 8B captions 和 30% 70B captions
        -   训练了 camera motion classifier，预测 16 个相机运动，类似 zoom-out, pan-left。将高置信度的相机运动放在文本 caption 前面。

    -   分阶段数据准备
        -   第一阶段是最小分辨率 720px 的视频
        -   第二阶段是最小分辨率 768px 的视频
        -   第三阶段是增强的高分辨率视频
        -   60% 包含人

    -   不同长度和尺寸的 bucket：
        -   使用 5 个不同 aspect ratio 的 bucket
        -   5 个不同长度的 bucket（4s-16s）

![](https://i-blog.csdnimg.cn/direct/1e8898cf0c7f43deb94d7cb8fff9c642.png)

### 训练

-   先训练 text-to-image (T2I)，然后 joint 训练 text-to-image 和 text-to-video (T2V)。这样训练比之前 joint 训练精度好，包括 T2I 和 T2V 任务上都会好
    -   T2I： 使用 256px 训练，方便用大 batchsize，可以过更多数据
    -   256px T2I/V：double spatial positional embedding 从而实现可变 aspect ratio，从 T2I 来初始化。增加 temporal position embedding layer，支持最大 32 latent
        -   4 epoch，中间 2x global size 能让 valid loss 大幅降低

    -   768 px T2I/V：3x spatial positional embedding
        -   10k iteration 后 valid loss 平了，然后降低 lr 又能继续降低 valid loss。后面就是 valid loss 不降低了就降低 lr

![](https://i-blog.csdnimg.cn/direct/b0d03e3ef04c475b86172b8f271caab4.png)

### 微调

#### 微调数据

1.  自动化挑选 candidate 视频：
    -   使用严格的美学、运动、场景切换等指标
    -   使用目标检测模型（支持 2w 类别）过滤小目标视频

1.  平衡视频中的概念
    -   video-text joint embedding model 来提取视频的特征，进行 video knn 来平衡分布

1.  手动识别电影视频
    -   标注员标注，挑选高质量视频（角度照明，鲜艳的不过饱和的，不混乱，non-trivial motion，无相机抖动，无覆盖文本）
    -   手动切分视频到合适的时长，将视频中最引入瞩目的切割出来

1.  人工修复 caption
    -   手工修改 LLaMa3-Video 生成的 caption

-   微调数据集的时长在 10.6s 到 16s 之间。50% 的视频都是 16s 长

#### 微调技巧

-   small batch size 和 64 nodes 来训练
-   cosine lr
-   16s 视频训练 16fps，其他的训练 24fps
-   模型平均
    -   不同的微调数据，超参数，预训练 ckpt 都会影响模型性能，包括运动、一致性和相机运动。使用模型平均技巧来综合各个模型的优势。这个操作类似 llama3

### 推理

-   推理参数
    -   classifier-free guidance scale： 7.5
    -   linear-quadratic sampler
    -   50 steps

-   推理 prompt 改写
    -   LLaMa3 进行改写
        -   替换复杂的词为通用和直接的表达，增强可理解性
        -   对运动细节的过度精细描述可能会导致生成视频中的伪影

-   70B 蒸馏一个 8B 模型来加速。蒸馏方式是
    -   先训练 70B llama3 模型(in-context learning 构造训练样例)，然后基于这个模型生成一些数据，人工挑选高质量数据给 8B 模型训练

-   使用 linear-quadratic strategy 来进行推理加速。因为发现降低太多 inference step 对于 video 的影响比 image 大。这种设计基于的观察是第一个推理步骤在设置视频的场景和运动方面很重要。

### 评测

-   提出了 Movie Gen Video Bench。使用人工评估来评估生成视频在各个评估轴上的质量。通过成对 A/B 测试来实现。
-   评估维度
    -   文本对齐
    -   视觉质量
    -   真实性和美学

-   发现 FVD/IS 等指标和人类评估不符合

### Video Personalization （个性化视频）

-   Personalized Text-to-Video (PT2V)
    -   人脸（masked face image）基于可学习的 Long-prompt MetaCLIP 提取特征加到 text prompt 中，经过 cross attention 融入模型
    -   freeze 黄色模块，绿色模块可训练

![](https://i-blog.csdnimg.cn/direct/440033e49f304d379d4bb057919e932b.png)

-   数据准备
    -   1fps 抽帧，利用人脸检测器挑选只有一个人的视频。相邻帧之间的 ArcFace cosine similarity score 大于 0.5
    -   需要也用上来源于不同视频的相同 id 的配对数据，避免模型学到 copy-paste shortcut

-   训练阶段如下

![](https://i-blog.csdnimg.cn/direct/3de0bbeec1a446ad8ea4484bc1843a23.png)

### Instruction-Guided Precise Video Editing （视频编辑）

-   介绍了 Movie Gen Edit，基于文本引导的视频编辑模型，在没有任何监督视频编辑数据的情况下训练
-   模型结构调整
    -   增加额外的 channel 作为 input video condition
    -   遵循Emu Edit，增加了对特定编辑任务（例如，添加对象、更改背景等）的条件化支持。具体来说，为每个任务有一个学习的任务嵌入向量。对于给定的任务，模型对相应的任务嵌入应用线性变换，产生四个嵌入，这些嵌入被连接到文本编码器的隐藏表示。还对任务嵌入应用第二次线性变换，并将得到的向量添加到时间步嵌入

-   分三阶段训练
    -   第一阶段：单帧视频编辑。利用图像编辑数据集，将图像编辑视为单帧视频编辑。和 text-to-video 一起训练，这里 condition video 用纯黑视频代替
    -   第二阶段：多帧视频编辑。从第一阶段训练的模型能够精确编辑图像，并从文本生成高质量视频。然而，在视频编辑任务中，它会产生非常模糊的编辑视频。假设这些伪影是由于第一阶段训练和视频编辑之间的训练测试差异造成的。识别出的最大差异是模型在第一阶段训练期间没有在多帧视频输入上进行条件化。通过以下两个任务数据进行训练	![](https://i-blog.csdnimg.cn/direct/1ab7b35a0dca4213ae12c4757c5b5462.png)
    -   第三阶段：通过反向翻译进行视频编辑。这样目标数据是真实数据会对模型学习更友好，数据类似如下：

![](https://i-blog.csdnimg.cn/direct/5a24a140110340e588c0c9ccac0ab48f.png)

![](https://i-blog.csdnimg.cn/direct/a1ab8232821a49019f8e4bdd4a62215b.png)

### Movie Gen Audio

-   给视频剪辑和短片生成配乐，这些视频的长度可能从几秒钟到几分钟不等。在这项工作中所考虑的配乐包括环境声、音效（Foley）和器乐音乐，但不包括语音或带有人声的音乐。环境声应该与视觉环境相匹配，音效应该在时间上与动作对齐，并且与视觉对象相符合，音乐应该表达视频的情绪和情感，与音效和环境声恰当融合，并与场景对齐，就像观看电影时人们所期望的那样。
    -   推理阶段是分 chunk 来做

![](https://i-blog.csdnimg.cn/direct/f0b26aea963e471b830a8a01a8964cda.png)

-   模型架构。构建了一个基于 flow-matching 和 DiT 的单一模型，该模型既可以根据视频生成音频，也可以在视频的部分音频已生成的情况下进行音频扩展。
    -   通过训练模型执行掩蔽音频预测来实现音频扩展，即模型根据整个视频及其周围的音频来预测音频目标
    -   支持文本提示来控制音乐风格

![](https://i-blog.csdnimg.cn/direct/e6a4671ac1d54e148fdfd1f261b04992.png)

## Experiments

### 和之前工作对比

-   Movie Gen Video Bench 中的 prompt 生成视频和之前工作对比，non-cherry picked 结果进行对比。不过和 sora 比是只能用 sora 公开的视频比较，这里就 5 选 1 进行对比。报告了我们模型的净胜率，这个比率可能在[-100, 100]的范围内。
    -   与Runway Gen3、LumaLabs和OpenAI Sora相比，我们发现电影生成视频在所有质量分解轴上要么表现更好，要么持平
    -   与Kling1.5相比，我们发现 modie gen 在帧一致性上显著获胜（13.5%），但在运动完整性上输了（-10.04%）。我们注意到，这种大的运动完整性与差的帧一致性表明Kling1.5有时会生成带有扭曲的不自然大运动。运动完整性只评估视频中运动的大小，不考虑扭曲、快速运动或不自然

![](https://i-blog.csdnimg.cn/direct/14a5d1c4bb2f4e0ba3d8399941b3012d.png)

### 定性结果

![](https://i-blog.csdnimg.cn/direct/6de891b6a27c4af7a9e9d5d9c219bf80.png)

-   moviegen 高质量，具有自然、逼真的运动，并与文本提示对齐

![](https://i-blog.csdnimg.cn/direct/847068077daa426fa64347116848b402.png)

-   moviegen 生成具有真实运动的自然观看视频，即使对于 out-of-training 提示也是如此。如图所示，对于这样的提示，OpenAI Sora 倾向于生成不太逼真的视频（例如，第二行中的卡通袋鼠）

![](https://i-blog.csdnimg.cn/direct/aaf479b60ffc46369c11ef0afb81618c.png)

### validation loss

-   validation loss 和人类评估的一致性很高

![](https://i-blog.csdnimg.cn/direct/081b41f266754d8c86abf39dab06b84c.png)

### finetune 影响

-   Visual quality 和 Text-alignment 上有提升

![](https://i-blog.csdnimg.cn/direct/1abaed72824f43c3a3f6f9b6c6b2030c.png)

### 消融实验

-   消融实验配置：用了更小的 5B 模型做消融实验，评测数据集也是一个子集 (Movie Gen Video Bench-Mini)

    -   352×192 分辨率、4-8 秒长的视频
    -   16×24×44 的潜在表示
    -   更小的数据集，包含2100万个视频，用LLaMa3-Video 8B进行标注，这些视频具有恒定的风景长宽比，用于视频训练

-   Flow Matching vs diffusion

    -   参考 emu video，diffusion 使用 v-pred 和 zero terminal-SNR formulation。flow matching 更好

-   video caption

    -   对比 image caption，video caption 在文本对齐上更好。image caption 是抽取三帧，第一帧，中间帧和最后一帧，然后改写三个图像描述为视频描述。

-   模型架构

    -   llama3 的架构比 dit 更好，模型架构区别是 ![](https://i-blog.csdnimg.cn/direct/9796197f72e34927ade7b5bd8c56645d.png)

![](https://i-blog.csdnimg.cn/direct/def15545742244339068a77c6c29e3ae.png)

### TAE 重建结果

-   定量精度。指标是在训练集上评测得到的，测试了 200 samples，2s/4s/6s/8s 时长数据。视频的测试是取所有 video frames 的指标平均。比较了一个不执行任何时间压缩的基线逐帧自编码器。我们观察到 TAE 在实现与逐帧编码器相当的性能的同时，实现了8倍更高的时间压缩。TAE 的表现超过了逐帧模型，这种改进可以归因于潜在通道大小的增加（8 对 16）

![](https://i-blog.csdnimg.cn/direct/6e0f2ddd15664febb97427a5f48bfbe8.png)

-   重建效果。TAE能够在保留视觉细节的同时重建视频帧。在图像和视频帧中的高频空间细节以及视频中的快速运动方面，TAE的重建质量会降低。当视频中同时存在高频空间细节和大幅度运动时，可能会导致细节丢失，正如图16中的示例所示，重建中的细节被平滑处理了

![](https://i-blog.csdnimg.cn/direct/08f5d1481a6c4c22a668d7805b3b36a3.png)

#### TAE 消融实验

-   消融实验的 baseline setting 是 4x 压缩率和 8-channel latent space
-   2.5D与3D注意力和卷积的比较。
    -   比较了在TAE中使用2.5D（即2D空间注意力/卷积后跟1D时间注意力/卷积）与使用3D时空注意力/卷积的效果
    -   观察到3D时空注意力带来了略微更好的重建指标。然而，我们发现这种改进并不足以证明与2.5D模型相比，一个完全3D模型相关的更大的内存和计算成本是合理的。因此，我们在我们的TAE中使用2.5D。

![](https://i-blog.csdnimg.cn/direct/17d2899fdb09407cb76d4e66c18855bb.png)

-   异常值惩罚损失的影响，研究了添加异常值惩罚损失（OPL）的效果。加这种损失可以去除生成和重建视频中的伪影，如图5所示，并提高重建性能。我们首先训练一个没有OPL的基线模型50K次迭代。然后，我们用OPL对这个模型进行微调10K次迭代，并将其与没有OPL微调20K次迭代的基线进行比较。总结在表12中的结果表明，OPL微调对于图像和视频的重建都有改善。

![](https://i-blog.csdnimg.cn/direct/06bcbd4ee10f4e8da119e418e328a8b3.png)

### spatial upsampler (超分辨率)

-   对比 200 px and 400 px crops，有视觉增强的作用

![](https://i-blog.csdnimg.cn/direct/86c27017ee8c41279bf6156c268c05f3.png)

### 文生图

-   在 joint 训练的基础上额外训练一个文生图模型。替换 TAE 为一个 image autoencoder
    -   post training 使用 1000 张内部的高质量数据

-   比当前 sota 模型都好

![](https://i-blog.csdnimg.cn/direct/fb664a0104f146ccb459eb48fc938469.png)

-   不过看样例美学等也是一般的水平

![](https://i-blog.csdnimg.cn/direct/a0c00ee06d8248749c94944ac218a42b.png)

### Personalized Movie Gen Video (PT2V)

-   强于当前 SOTA (ID-Animator)

![](https://i-blog.csdnimg.cn/direct/c3305958827d44a381a92cab83094be1.png)

-   可视化

![](https://i-blog.csdnimg.cn/direct/b5d9a911c654434db212fbbbb55d5a43.png)
![](https://i-blog.csdnimg.cn/direct/b1bafd97693f482cb161f763c9948a88.png)

### 视频编辑评估

-   人工评估和 ViCLIP 指标都是最好的

![](https://i-blog.csdnimg.cn/direct/135b16ac53864548917d0cb12eeedb34.png)

-   基于多任务的全参数 finetune 比使用 adapter 方式更好

![](https://i-blog.csdnimg.cn/direct/d42735ba441841e3a97b82cb3a098eed.png)

## Thoughts

-   目前视频生成领域最详尽的技术报告，没有之一
-   TAE 用 tiling 方式引入的 blending 操作可能会有 artifacts，看起来有优化空间
-   部分 [demo 视频](https://ai.meta.com/research/movie-gen/) 看起来细节丰富度一般，可能是低分辨率推理加上 2x 上采样超分带来的问题。不过物理规律方面看起来还挺强的。
-   期待未来能像 llama 系列一样开源



Reference:

