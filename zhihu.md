## TL;DR

-   2024 年微软发表的数字人工作 VASA-1，支持基于单人头像照片和语音音频输入，来实时生成数字人视频。支持在线生成高达 40 FPS 的 512×512 分辨率视频，目前为数不多能做到实时推理且效果很好的 talking face 工作。

**Paper name**

VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time

**Paper Reading Note**

Paper URL: https://arxiv.org/abs/2404.10667

Project URL: https://www.microsoft.com/en-us/research/project/vasa-1/

---

## Introduction

### 背景

-   人脸不仅仅是一个面孔，而是一个动态的画布，每一个微妙的动作和表情都能表达情感、传达未言之语，并促进共情联系。
-   人工智能生成的会说话的面孔的出现，为我们打开了一扇窗，让我们看到了一个技术增强人与人之间、人与人工智能互动丰富性的未来。这样的技术有望丰富数字通信，为沟通障碍者增加可及性，通过互动式人工智能辅导转变教育方法，并在医疗保健中提供治疗支持和社交互动。
-   当前的技术主要关系唇部动作，同时生成效率较慢，难以实时处理

### 本文方案

-   VASA：根据单个静态图像和语音音频剪辑生成具有吸引力的视觉情感技能（VAS）的逼真会说话的面孔
    -   生成与音频完美同步的唇部动作
    -   产生一系列面部细微表情和自然头部动作
    -   支持在线生成高达40 FPS的512×512视频，几乎没有启动延迟

![](https://i-blog.csdnimg.cn/direct/8770acb2215f4a76b8fb871030e6574b.png)

## Methods

-   采用单张人脸图像、可选的控制信号和语音音频片段来生成逼真的说话人脸视频。不是直接生成视频帧，而是在潜在空间中根据音频和其他信号生成整体面部动态和头部运动。
    -   首先构建面部潜在空间并训练面部编码器和解码器。我们设计了一个富有表现力和解耦的面部潜在学习框架，并在真实人脸视频上进行训练。
    -   然后我们训练一个简单但功能强大的扩散 transformer（Diffusion Transformer），在测试时根据音频和其他条件对运动分布进行建模并生成运动潜在编码

### 富有表现力和解耦的面部潜在空间构建

-   给定一个未标记的说话人脸视频集，旨在**为人脸建立一个具有高度解耦性和表现力的潜在空间**。
    -   解耦使得能够在大量视频上对人头和整体面部行为进行有效的生成建模，而不受主体身份的影响。它还实现了输出的解耦因素控制，这在许多应用中都是有用的。
    -   另一方面，面部外观和动态运动的表现力确保解码器能够输出具有丰富面部细节的高质量视频，而潜在生成器能够捕捉细微的面部动态。

-   参考 Megaportraits 等工作中的 3D 辅助面部重演框架 (face reenactment framework) 构建模型。与 2D 特征图相比，3D 外观特征体能更好地表征 3D 中的外观细节。显式的 3D 特征扭曲 (warping) 在对 3D 头部和面部运动建模方面也很强大。
    -   具体来说，我们将面部图像分解为规范的 3D 外观体积 <img src="https://www.zhihu.com/equation?tex=V_%7Bapp%7D" alt="V_{app}" class="ee_img tr_noresize" eeimg="1">、身份代码 <img src="https://www.zhihu.com/equation?tex=z_%7Bid%7D" alt="z_{id}" class="ee_img tr_noresize" eeimg="1">、3D 头部姿势 <img src="https://www.zhihu.com/equation?tex=z_%7Bpose%7D" alt="z_{pose}" class="ee_img tr_noresize" eeimg="1"> 和面部动态代码 <img src="https://www.zhihu.com/equation?tex=z_%7Bdyn%7D" alt="z_{dyn}" class="ee_img tr_noresize" eeimg="1">。除了 <img src="https://www.zhihu.com/equation?tex=V_%7Bapp%7D" alt="V_{app}" class="ee_img tr_noresize" eeimg="1"> 外，其余都是通过独立编码器从面部图像中提取的，而 <img src="https://www.zhihu.com/equation?tex=V_%7Bapp%7D" alt="V_{app}" class="ee_img tr_noresize" eeimg="1"> 是通过先提取带姿势的 3D 体积，然后进行刚性和非刚性 3D 扭曲到规范体积来构建的，这种方法参考了 Megaportraits。单个解码器 <img src="https://www.zhihu.com/equation?tex=D" alt="D" class="ee_img tr_noresize" eeimg="1"> 将这些潜在变量作为输入并重建人脸图像，其中，首先对 <img src="https://www.zhihu.com/equation?tex=V_%7Bapp%7D" alt="V_{app}" class="ee_img tr_noresize" eeimg="1"> 应用类似的逆方向扭曲场以获得带姿势的外观体积。读者可以参考文献 Megaportraits 以了解该架构的更多细节。
    -   为了学习解耦的潜在空间，核心思想是通过在视频中的不同图像之间交换潜在变量来构建图像重建损失。我们的基本损失函数改编自 Megaportraits，但我们发现原始损失在面部动态和头部姿势之间的解耦较差，同时身份与运动之间的解耦也不完善。因此，我们引入了几种额外的损失来实现我们的目标。
        -   受文献 Dpe 启发，我们添加了一种**成对的头部姿势和面部动态传递损失**，以改善它们的解耦。设 <img src="https://www.zhihu.com/equation?tex=I_i" alt="I_i" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=I_j" alt="I_j" class="ee_img tr_noresize" eeimg="1"> 为从同一视频中随机采样的两个帧。我们使用编码器提取它们的潜在变量，将 <img src="https://www.zhihu.com/equation?tex=I_i" alt="I_i" class="ee_img tr_noresize" eeimg="1"> 的头部姿势传递给 <img src="https://www.zhihu.com/equation?tex=I_j" alt="I_j" class="ee_img tr_noresize" eeimg="1"> 得到 <img src="https://www.zhihu.com/equation?tex=%5Chat%7BI%7D_%7Bj%2Cz_%7Bpose%7D%5Ei%7D%20%3D%20D%28V_%7Bapp%7D%5Ej%2C%20z_%7Bid%7D%5Ej%2C%20z_%7Bpose%7D%5Ei%2C%20z_%7Bdyn%7D%5Ej%29" alt="\hat{I}_{j,z_{pose}^i} = D(V_{app}^j, z_{id}^j, z_{pose}^i, z_{dyn}^j)" class="ee_img tr_noresize" eeimg="1">，并将 <img src="https://www.zhihu.com/equation?tex=I_j" alt="I_j" class="ee_img tr_noresize" eeimg="1"> 的面部运动传递给 <img src="https://www.zhihu.com/equation?tex=I_i" alt="I_i" class="ee_img tr_noresize" eeimg="1"> 得到 <img src="https://www.zhihu.com/equation?tex=%5Chat%7BI%7D_%7Bi%2Cz_%7Bdyn%7D%5Ej%7D%20%3D%20D%28V_%7Bapp%7D%5Ei%2C%20z_%7Bid%7D%5Ei%2C%20z_%7Bpose%7D%5Ei%2C%20z_%7Bdyn%7D%5Ej%29" alt="\hat{I}_{i,z_{dyn}^j} = D(V_{app}^i, z_{id}^i, z_{pose}^i, z_{dyn}^j)" class="ee_img tr_noresize" eeimg="1">。然后最小化 <img src="https://www.zhihu.com/equation?tex=%5Chat%7BI%7D_%7Bj%2Cz_%7Bpose%7D%5Ei%7D" alt="\hat{I}_{j,z_{pose}^i}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=%5Chat%7BI%7D_%7Bi%2Cz_%7Bdyn%7D%5Ej%7D" alt="\hat{I}_{i,z_{dyn}^j}" class="ee_img tr_noresize" eeimg="1"> 之间的一致性损失 <img src="https://www.zhihu.com/equation?tex=l_%7Bconsist%7D" alt="l_{consist}" class="ee_img tr_noresize" eeimg="1">。
        -   为了强化身份与运动之间的解耦，我们为跨身份姿势和面部运动传递结果添加了**面部身份相似性损失** $l_{cross_id}<img src="https://www.zhihu.com/equation?tex=%E3%80%82%E8%AE%BE%20" alt="。设 " class="ee_img tr_noresize" eeimg="1">I_s<img src="https://www.zhihu.com/equation?tex=%20%E5%92%8C%20" alt=" 和 " class="ee_img tr_noresize" eeimg="1">I_d<img src="https://www.zhihu.com/equation?tex=%20%E4%B8%BA%E4%B8%A4%E4%B8%AA%E4%B8%8D%E5%90%8C%E4%B8%BB%E4%BD%93%E7%9A%84%E8%A7%86%E9%A2%91%E5%B8%A7%EF%BC%8C%E5%8F%AF%E4%BB%A5%E5%B0%86%20" alt=" 为两个不同主体的视频帧，可以将 " class="ee_img tr_noresize" eeimg="1">I_d<img src="https://www.zhihu.com/equation?tex=%20%E7%9A%84%E8%BF%90%E5%8A%A8%E4%BC%A0%E9%80%92%E5%88%B0%20" alt=" 的运动传递到 " class="ee_img tr_noresize" eeimg="1">I_s<img src="https://www.zhihu.com/equation?tex=%20%E4%B8%8A%EF%BC%8C%E5%BE%97%E5%88%B0%20" alt=" 上，得到 " class="ee_img tr_noresize" eeimg="1">\hat{I}_{s,z_{pose}^d,z_{dyn}^d} = D(V_{app}^s, z_{id}^s, z_{pose}^d, z_{dyn}^d)<img src="https://www.zhihu.com/equation?tex=%E3%80%82%E7%84%B6%E5%90%8E%E5%AF%B9%E4%BB%8E%20" alt="。然后对从 " class="ee_img tr_noresize" eeimg="1">I_s<img src="https://www.zhihu.com/equation?tex=%20%E5%92%8C%20" alt=" 和 " class="ee_img tr_noresize" eeimg="1">\hat{I}_{s,z_{pose}^d,z_{dyn}^d}$ 提取的深度面部身份特征 (arcface) 应用余弦相似性损失。正如我们在实验中展示的那样，我们的新损失函数设计对于实现有效的因素解耦以及促进高质量、逼真的说话人脸生成至关重要。

### 3.2 整体面部动态生成与扩散变换器

-   在构建了人脸潜在空间并训练了编码器后，我们可以从真实的说话人脸视频中提取面部动态和头部动作，并训练生成模型。关键在于我们考虑的是与身份无关的整体面部动态生成（HFDG），其中学习到的潜在代码表示所有面部运动，例如唇部动作、（非唇部的）表情、眼睛的注视和眨眼。这不同于现有方法，它们通常为不同因素应用独立模型，使用交错的回归和生成形式 (比如 Audio2head、Sadtalker 等)。此外，以往方法往往在有限的身份数据上进行训练 (比如 Sadtalker、Codetalker 等)，无法涵盖不同人类的广泛运动模式，特别是在存在丰富运动潜在空间的情况下。
-   在本研究中，我们利用扩散模型进行音频条件的 HFDG，并在大量多身份的说话人脸视频上进行训练。我们采用变换器架构 (参考 DiT、Diffposetalk) 来完成序列生成任务。图 2 展示了我们的HFDG框架概览。

![](https://i-blog.csdnimg.cn/direct/fe64d117f0f142a9a023c9e663e893e8.png)

-   形式上，从视频片段中提取的运动序列定义为 $X = \{[z_{pose}^i , z_{dyn}^i ]\}, i = 1, . . . , W<img src="https://www.zhihu.com/equation?tex=%E3%80%82%E7%BB%99%E5%AE%9A%E5%85%B6%E5%AF%B9%E5%BA%94%E7%9A%84%E9%9F%B3%E9%A2%91%E7%89%87%E6%AE%B5%20" alt="。给定其对应的音频片段 " class="ee_img tr_noresize" eeimg="1">a<img src="https://www.zhihu.com/equation?tex=%EF%BC%8C%E6%88%91%E4%BB%AC%E4%BD%BF%E7%94%A8%E9%A2%84%E8%AE%AD%E7%BB%83%E7%9A%84%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E5%99%A8%20Wav2Vec2%20%E6%8F%90%E5%8F%96%E5%90%8C%E6%AD%A5%E7%9A%84%E9%9F%B3%E9%A2%91%E7%89%B9%E5%BE%81%20" alt="，我们使用预训练的特征提取器 Wav2Vec2 提取同步的音频特征 " class="ee_img tr_noresize" eeimg="1">A = \{f_{audio}^i\}$。

#### 扩散模型公式

扩散模型定义了两个马尔科夫链，正向链逐渐向目标数据添加高斯噪声，反向链则逐步从噪声中恢复原始信号。基于去噪得分匹配目标，我们定义简化的损失函数为：

$$
\mathbb{E}_{t \sim U[1, T], X_0, C \sim q(X_0, C)}(|X_0 - H(X_t, t, C)|^2),
$$

其中 <img src="https://www.zhihu.com/equation?tex=t" alt="t" class="ee_img tr_noresize" eeimg="1"> 表示时间步，<img src="https://www.zhihu.com/equation?tex=X_0%20%3D%20X" alt="X_0 = X" class="ee_img tr_noresize" eeimg="1"> 为原始运动潜在序列，<img src="https://www.zhihu.com/equation?tex=X_t" alt="X_t" class="ee_img tr_noresize" eeimg="1"> 是扩散前向过程 <img src="https://www.zhihu.com/equation?tex=q%28X_t%20%7C%20X_%7Bt-1%7D%29%20%3D%20N%28X_t%3B%20%5Csqrt%7B1%20-%20%5Cbeta_t%7D%20X_%7Bt-1%7D%2C%20%5Cbeta_t%20I%29" alt="q(X_t | X_{t-1}) = N(X_t; \sqrt{1 - \beta_t} X_{t-1}, \beta_t I)" class="ee_img tr_noresize" eeimg="1"> 生成的噪声输入。<img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> 是我们的 transformer 网络，它直接预测原始信号而非噪声。<img src="https://www.zhihu.com/equation?tex=C" alt="C" class="ee_img tr_noresize" eeimg="1"> 是条件信号，接下来将进行描述。

#### 条件信号

-   在我们的音频驱动的运动生成任务中，主要的条件信号是音频特征序列 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1">。我们还加入了若干额外的信号，不仅让生成建模更加易处理，还增强了生成的可控性。具体来说，我们考虑了主要的注视方向 <img src="https://www.zhihu.com/equation?tex=g" alt="g" class="ee_img tr_noresize" eeimg="1">、头部与摄像机的距离 <img src="https://www.zhihu.com/equation?tex=d" alt="d" class="ee_img tr_noresize" eeimg="1"> 以及情感偏移 <img src="https://www.zhihu.com/equation?tex=e" alt="e" class="ee_img tr_noresize" eeimg="1">。

    -   主要注视方向 <img src="https://www.zhihu.com/equation?tex=g%20%3D%20%28%CE%B8%2C%20%CF%86%29" alt="g = (θ, φ)" class="ee_img tr_noresize" eeimg="1"> 由球坐标中的一个向量定义，用来指定生成的说话人脸的注视方向。我们通过在每帧中使用文献 L2cs-net 的方法，并辅以简单的基于直方图的聚类算法，提取训练视频片段中的 <img src="https://www.zhihu.com/equation?tex=g" alt="g" class="ee_img tr_noresize" eeimg="1">。
    -   头部距离 <img src="https://www.zhihu.com/equation?tex=d" alt="d" class="ee_img tr_noresize" eeimg="1"> 是一个归一化的标量，控制人脸与虚拟摄像机之间的距离，从而影响生成视频中人脸的大小。我们通过 3D 人脸重建的方法为训练视频获取这一尺度标签。
    -   情感偏移 <img src="https://www.zhihu.com/equation?tex=e" alt="e" class="ee_img tr_noresize" eeimg="1"> 用于调节说话人脸上显示的情绪。注意，情绪通常内在地与音频相关，且能从音频中大致推断出来；因此，<img src="https://www.zhihu.com/equation?tex=e" alt="e" class="ee_img tr_noresize" eeimg="1"> 仅作为一个全局偏移，用于在需要时增强或适度地调整情绪，而非在推理过程中完全改变情绪或产生与输入音频不一致的情绪。在实际操作中，我们使用文献 Hsemotion 提取的平均情绪系数作为情绪信号。

-   为了实现相邻窗口之间的无缝过渡，我们将前一个窗口中音频特征和生成的运动的最后 <img src="https://www.zhihu.com/equation?tex=K" alt="K" class="ee_img tr_noresize" eeimg="1"> 帧作为当前窗口的条件。总结而言，我们的输入条件可以表示为 <img src="https://www.zhihu.com/equation?tex=C%20%3D%20%5BX_%7B%5Ctext%7Bpre%7D%7D%2C%20A_%7B%5Ctext%7Bpre%7D%7D%3B%20A%2C%20g%2C%20d%2C%20e%5D" alt="C = [X_{\text{pre}}, A_{\text{pre}}; A, g, d, e]" class="ee_img tr_noresize" eeimg="1">。所有条件在时间维度上与噪声连接后作为 transformer 的输入。

#### 无分类器引导 (CFG)

在训练阶段，我们随机去除每个输入条件。在推理阶段，我们应用：

<img src="https://www.zhihu.com/equation?tex=%5Chat%7BX%7D_0%20%3D%20%5Cleft%281%20%2B%20%5Csum_%7Bc%20%5Cin%20C%7D%20%5Clambda_c%5Cright%29%20%5Ccdot%20H%28X_t%2C%20t%2C%20C%29%20-%20%5Csum_%7Bc%20%5Cin%20C%7D%20%5Clambda_c%20%5Ccdot%20H%28X_t%2C%20t%2C%20C%20%7C_%7Bc%20%3D%20%E2%88%85%7D%29%5C%5C" alt="\hat{X}_0 = \left(1 + \sum_{c \in C} \lambda_c\right) \cdot H(X_t, t, C) - \sum_{c \in C} \lambda_c \cdot H(X_t, t, C |_{c = ∅})\\" class="ee_img tr_noresize" eeimg="1">

其中，<img src="https://www.zhihu.com/equation?tex=%5Clambda_c" alt="\lambda_c" class="ee_img tr_noresize" eeimg="1"> 是条件 <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> 的CFG比例。<img src="https://www.zhihu.com/equation?tex=C%20%7C%20_%7Bc%20%3D%20%E2%88%85%7D" alt="C | _{c = ∅}" class="ee_img tr_noresize" eeimg="1"> 表示将条件 <img src="https://www.zhihu.com/equation?tex=c" alt="c" class="ee_img tr_noresize" eeimg="1"> 替换为空集。在训练中，除了 <img src="https://www.zhihu.com/equation?tex=X_%7B%5Ctext%7Bpre%7D%7D" alt="X_{\text{pre}}" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=A_%7B%5Ctext%7Bpre%7D%7D" alt="A_{\text{pre}}" class="ee_img tr_noresize" eeimg="1"> 使用 0.5 的丢弃概率外，其他每个条件的丢弃概率为 0.1。这样可以确保模型能很好地处理没有前置音频和运动的第一个窗口（即设置为空）。我们还随机去除 <img src="https://www.zhihu.com/equation?tex=A" alt="A" class="ee_img tr_noresize" eeimg="1"> 的最后几帧，以确保在音频序列短于窗口长度时也能生成稳健的运动。

### 3.3 说话人脸视频生成

在推理时，给定任意人脸图像和音频片段，我们首先使用训练好的面部编码器提取 3D 外观体积 <img src="https://www.zhihu.com/equation?tex=V_%7Bapp%7D" alt="V_{app}" class="ee_img tr_noresize" eeimg="1"> 和身份代码 <img src="https://www.zhihu.com/equation?tex=z_%7Bid%7D" alt="z_{id}" class="ee_img tr_noresize" eeimg="1">。接着，提取音频特征，将其分割成长度为 <img src="https://www.zhihu.com/equation?tex=W" alt="W" class="ee_img tr_noresize" eeimg="1"> 的片段，并使用训练好的扩散变换器 <img src="https://www.zhihu.com/equation?tex=H" alt="H" class="ee_img tr_noresize" eeimg="1"> 以滑动窗口的方式逐段生成头部和面部运动序列 $\{X = \{[z_{pose}^i , z_{dyn}^i ]\}\}$。最终的视频可以使用我们训练好的解码器生成。

## Experiments

### 实验配置

-   模型
    -   对于运动潜在生成，我们使用一个 8 层的 transformer 编码器，嵌入维度为512，头数为 8，作为我们的扩散网络。

-   数据
    -   对于人脸潜在空间的学习：我们使用公开数据集VoxCeleb2，该数据集包含大约 6000 名主体的说话人脸视频。我们重新处理该数据集，并通过 Blind image quality assessment (BIQA)  的方法丢弃包含多个人物的片段以及质量较低的片段
    -   对于运动潜在生成：该模型在 VoxCeleb2 和我们收集的另一个高分辨率对话视频数据集上进行训练，后者包含约 3500 名主体。

-   推理配置：
    -   在我们的默认设置中，模型使用面向前方的主要注视条件，所有训练视频的平均头部距离，以及空的情感偏移条件。
    -   CFG 参数设置为 <img src="https://www.zhihu.com/equation?tex=%5Clambda_A%20%3D%200.5" alt="\lambda_A = 0.5" class="ee_img tr_noresize" eeimg="1"> 和 <img src="https://www.zhihu.com/equation?tex=%5Clambda_g%20%3D%201.0" alt="\lambda_g = 1.0" class="ee_img tr_noresize" eeimg="1">，采样步数为50。

-   训练
    -   我们的人脸潜在模型在 4 个 NVIDIA RTX A6000 GPU 工作站上大约需要 7 天的训练时间，而扩散变换器大约需要 3 天。用于训练的总数据量约为 50 万段，每段持续 2 到 10 秒。我们的 3D 辅助人脸潜在模型和扩散变换器模型的参数量分别约为 200M 和 29M。

### 评估基准

-   我们使用两个数据集来评估我们的方法。
    -   第一个是 VoxCeleb2 的一个子集。我们从 VoxCeleb2 的测试集随机选择了46个主体，并为每个主体随机抽取 10 个视频片段，总计 460 个片段。这些视频片段时长约为 5 到 15 秒（80% 小于 10 秒），大部分内容为访谈和新闻报道。
    -   为了进一步评估我们的方法在长时语音生成和更广泛声音变化下的表现，我们另外收集了 17 个人的 32 个一分钟视频片段。这些视频主要来自在线辅导课程和教育讲座，其讲话风格比 VoxCeleb2 更为多样化。我们将此数据集称为  OneMin-32。

### 推理速度

在离线批处理模式下，我们的方法可生成 512×512 大小的视频帧，速度为45fps；在在线流媒体模式下，支持速度最高可达 40fps，前置延迟仅为 170ms，评估是在配备单个 NVIDIA RTX 4090 GPU 的桌面 PC 上进行的。

### 定量评估

#### 评测指标

我们使用以下指标对生成的**唇部运动**、**头部姿态**和**整体视频质量**进行定量评估，包括一种类似于 CLIP 训练的新数据驱动的音频-姿态同步指标：

-   **音频-唇部同步**。我们使用一个预训练的音频-唇部同步网络，即 SyncNet，来评估输入音频与生成唇部运动在视频中的对齐情况。具体来说，我们分别计算置信度分数和特征距离，分别表示为 SC 和 SD。通常来说，较高的 SC 和较低的 SD 表明更好的音频-唇部同步质量。
-   **音频-姿态对齐**。衡量生成的头部姿态与输入音频的对齐并不容易，目前没有成熟的指标。一些最近的研究使用了 Beat Align Score 来评估音频-姿态对齐。然而，由于“节拍”在自然语音和人类头部运动中的概念并不清晰，这一指标并不理想。在本研究中，我们引入了一种新的数据驱动指标，称为对比音频和姿态预训练 (CAPP) 分数。受到 CLIP 的启发，我们联合训练了一个姿态序列编码器和一个音频序列编码器，并预测输入的姿态序列和音频是否配对。音频编码器从预训练的 Wav2Vec2 网络初始化，而姿态编码器是一个随机初始化的 6 层 Transformer 网络。输入窗口大小为 3 秒。我们的 CAPP 模型在 2K 小时的真实音频和姿态序列上训练，表现出评估音频输入与生成姿态同步程度的强大能力。
-   **姿态变化强度**。我们进一步定义了一个姿态变化强度分数 ∆P，它是相邻帧之间的姿态角度差异的平均值。∆P 在所有生成的帧上进行平均，提供了该方法生成的总体头部运动强度的指示。
-   **视频质量**。根据以前的视频生成工作，我们使用弗雷谢视频距离 (FVD) 来评估生成的视频质量。我们使用 25 帧的连续序列，分辨率为 224×224 来计算 FVD 指标。

#### 对比结果

-   我们将我们的方法与现有的三个基于音频的说话人脸生成方法进行比较：MakeItTalk，Audio2Head 和 SadTalker。
    -   没有在 VoxCeleb2 上评估 FVD，因为其视频质量参差不齐且往往较低。
    -   在两个基准上，我们的方法在所有评估指标上均表现最佳。就音频-唇部同步分数（SC 和 SD）而言，我们的方法远超其他方法。需要注意的是，我们的方法比真实视频得分更高，这归因于音频 CFG 的效果。
    -   我们生成的姿态与音频的对齐效果更好，尤其是在 OneMin-32 基准上，这在 CAPP 分数上得到了反映。
    -   根据 ∆P，头部运动也显示出最高的强度，尽管与真实视频的强度仍有差距。
    -   我们的 FVD 分数显著低于其他方法，表明我们生成结果的视频质量和真实性更高。

![](https://i-blog.csdnimg.cn/direct/132c91b3e5f24b1791e8fbfbadab1b90.png)

### 定性评估

**视觉结果**。图 1 展示了我们方法的一些具有代表性的基于音频驱动的说话人脸生成结果。通过视觉检查，我们的方法能够生成高质量的视频帧，并展现出生动的面部情感。此外，还能生成类似人类的对话行为，包括在讲话和思考过程中偶尔的眼神转移，以及自然和多变的眨眼节奏等细微之处。

**生成的可控性**。图 3 显示了我们在不同控制信号下的生成结果，包括主要眼神方向、头部距离和情绪偏移。我们的模型能够很好地解读这些信号，并生成紧密符合这些指定参数的说话人脸结果。

![](https://i-blog.csdnimg.cn/direct/18509e2e310b418f971fd7a228dcc3a1.png)

**面部潜在因素的解耦**。图 A.1 显示了在不同主体上应用相同的运动潜在序列时，我们的方法有效地保持了独特的面部运动和独特的面部身份。这表明了我们方法在身份和运动解耦方面的有效性。图 A.2 进一步展示了头部姿态和面部的有效解耦。
![](https://i-blog.csdnimg.cn/direct/93eecf4925114fcbb054cf6af9e113db.png)
![](https://i-blog.csdnimg.cn/direct/57b85639ff6e4cdba25fab89e8e2e11b.png)

## Thoughts

-   训练和推理资源消耗都不大，几十 M 的 diffusion 模型支持实时生成，官网上的效果也不错，有比较大的应用前景，可惜没有开源。
-   比较好奇其中解耦设计带来的具体好处，消融实验暂时没有提到



Reference:

