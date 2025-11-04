### Segment Anything

1. Abstract

   1. We introduce the Segment Anything (SA) project: a new task, model, and dataset for image segmentation. Using our efficient model in a data collection loop, we built the largest segmentation dataset to date (by far), with over 1 billion masks on 11M licensed and privacy respecting images. **The model is designed and trained to be promptable, so it can transfer zero-shot to new image distributions and tasks**. We evaluate its capabilities on numerous tasks and find that its zero-shot performance is impressive – often competitive with or even superior to prior fully supervised results. We are releasing the Segment Anything Model (SAM) and corresponding dataset (SA-1B) of 1B masks and 11M images at https://segment-anything.com to foster research into foundation models for computer vision.
   2. 强调零样本学习的优势，并且图像分割的操作是可以被Prompt所驱动的。

2. Introduction

   1. 对齐来自网络的成对文本和图像。例如 CLIP 和 ALIGN 使用对比学习来训练对齐两种模态的文本和图像编码器。
   2. 1. What task will enable zero-shot generalization? 
      2. What is the corresponding model architecture? 
      3. What data can power this task and model?、

3. Segment Anything Task

   1. Task：点、框、掩码、“everything”...
   2. Pre-train：为每个训练样本模拟一系列提示（例如，点、框、掩码），并将模型的掩码预测与基本事实进行比较。
   3. Zero-shot transfer：可以通过向我们的模型提供检测器的框输出作为提示来解决猫实例分割。In general, a wide array of practical segmentation tasks can be cast as prompting.
   4. 分割是一个广泛的领域：有交互式分割[57,109]、边缘检测[3]、超级像素化[85]、对象建议生成[2]、前景分割[94]、语义分割[90]、实例分割[66]、全景分割[59]等

4. SAM Model Structure

   ![imchatu 23](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\SAM Model Structure.png)

   1. **Image Encoder：**图像编码器。在可扩展性和强大的预训练方法的激励下，我们使用了**MAE**[【47】](https://arxiv.org/abs/2111.06377)预训练的**视觉转换器(ViT)**[【33】](https://arxiv.org/abs/2010.11929)，该转换器至少适用于处理高分辨率输入[【62】](https://arxiv.org/abs/2203.16527)。图像编码器在每个图像运行一次，可以在提示模型之前应用。
   2. **Prompt Encoder：**提示编码器。我们考虑两组提示：稀疏（点、框、文本）和密集（掩码）。我们用**位置编码**[【95】](https://arxiv.org/abs/2006.10739)表示点和框，并用**CLIP** [【82】](https://arxiv.org/abs/2103.00020)的现成文本编码器对每种提示类型的学习嵌入和自由格式文本进行求和。密集提示（即掩码）使用卷积嵌入，并与图像嵌入按元素求和。
   3. **Mask decoder**：掩码解码器有效地将图像嵌入、提示嵌入和输出令牌映射到掩码。该设计受到[【14】](https://arxiv.org/abs/2005.12872),[【20】](https://arxiv.org/abs/2107.06278)的启发，采用了对**Transformer解码器**块[【103】](https://arxiv.org/abs/1706.03762)的修改，后跟一个**动态掩码预测头**。我们修改后的解码器块在两个方向（提示到图像嵌入，反之亦然）使用提示自注意力和交叉注意力来更新所有嵌入。运行两个块后，我们对图像嵌入进行上采样，MLP 将输出标记映射到动态线性分类器，然后计算
   4. **Resolving ambiguity**：解决歧义。对于一个输出，如果给出模棱两可的提示，模型将对多个有效掩码进行平均。为了解决这个问题，我们修改了模型以预测单个提示的多个输出掩码（见图 3）。我们发现 3 个掩码输出足以解决大多数常见情况（嵌套掩码通常最多三个深度：整体、部分和子部分）。在训练期间，我们仅在面罩上背支撑最小损失 [15， 45， 64]。为了对掩码进行排名，该模型预测每个掩码的置信度分数（即估计的 IoU）。 
   5. **Efficiency**：整体模型设计很大程度上是由效率驱动的。给定预先计算的图像嵌入，提示编码器和掩码解码器在 CPU 上的 Web 浏览器中运行，运行时间约为 50 毫秒。这种运行时性能支持我们的模型的无缝、实时交互式提示。 
   6. **Losses and training**：我们使用[14]中使用的Focal Loss[65]和Dice Loss[73]的线性组合来监督掩模预测。我们使用几何提示的混合来训练可提示的分割任务（有关文本提示，请参阅 §7.5）。按照[92,37]，我们通过在每个掩码中随机采样11轮提示来模拟交互式设置，使SAM能够无缝集成到我们的数据引擎中。

