# 基于多模态融合的脑疾病智能诊断方法
> 此项目为本科毕业设计，上传至Github以存档，如有问题欢迎交流！

## 运行环境（AutoDL云服务器）

14核 Intel(R) Xeon(R) Gold 6330 CPU

24GB RTX3090

PyTorch 1.10.0

Monai（实现数据增强、数据集读取）

Visual Studio Code

## 1.数据集

### 数据集概况

项目数据集来自ANDI（Alzheimer's Disease Neuroimaging Initiative）数据库。该数据库是由美国国立卫生研究院于2003年资助创建的[14]，公开了诸如MRI、PET以及相关诊断信息，以供研究人员进行阿尔兹海默症的诊断研究。有关最新信息，请参阅http://www.adni-info.org。

实验数据集的2个分类分别为AD（Alzheimer’s Disease）和CN（Cognitively Normal），本文从数据库中选取每个主题的MRI-T1和FDG-PET图像。通过筛选得到238个主题的多模态图像，其中，2个类别所对应的主题数分别为108名AD、130名CN。数据集各类别的信息统计见下表。

|                         |     AD     |     CN     |
| :---------------------: | :--------: | :--------: |
|       主题人数/人       |    108     |    130     |
| 年龄（平均年龄±标准差） | 75.16±6.27 | 75.72±6.49 |
|      性别（男/女）      |   61/47    |   64/66    |

### 数据集预处理

数据集需经过AC-PC校正、偏置场校正以及颅骨剥离，以上操作均使用基于Matlab开发的软件SPM12（http://www.fil.ion.ucl.ac.uk/spm/）以及其CAT12工具箱（http://www.neuro.uni-jena.de/cat/）完成。

## 2.Model

模型基于ResNet-18，将原有2D卷积核替换为3D卷积核，将MRI和PET输入并行的3D ResNet-18网络进行特征提取，将提取出的特征Concat实现融合，最后放入全连接层实现分类，模型结构如下：

![image-20220603152948401](/statistics/model.png)

## 3.实验结果

<img src="/statistics/Accuracy.png" alt="image-20220603153058188" style="zoom:30%;" />

<img src="/statistics/Loss.png" alt="image-20220603153131521" style="zoom:30%;" />

<img src="/statistics/ROC_AUC_Curve.png" alt="image-20220603154919072" style="zoom:30%;" />

|  AD:CN  |            |            |            |            |
| :-----: | :--------: | :--------: | :--------: | :--------: |
|         |    ACC     |    AUC     |  F1-Score  |   Recall   |
|   MRI   |   0.9125   |   0.9323   |   0.9123   |   0.9129   |
|   PET   |   0.8750   |   0.8708   |   0.8738   |   0.8729   |
| MRI+PET | **0.9444** | **0.9463** | **0.9442** | **0.9463** |

可见，使用MRI+PET融合特征作为输入的准确率比使用MRI提取特征分类的准确率提高了3.21%，比使用PET提取特征分类的准确率提高了6.94%。
