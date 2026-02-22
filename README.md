# MovieLens-LightGCN-FuzhouU

**我完成的第一篇学术论文开源项目**

[![GitHub stars](https://img.shields.io/github/stars/dutyman-sy/MovieLens-LightGCN-FuzhouU)](https://github.com/dutyman-sy/MovieLens-LightGCN-FuzhouU/stargazers)
[![GitHub license](https://img.shields.io/github/license/dutyman-sy/MovieLens-LightGCN-FuzhouU)](https://github.com/dutyman-sy/MovieLens-LightGCN-FuzhouU/blob/main/LICENSE)

项目简介
本项目是福州大学学生王尚宇独立完成的科研实践成果，聚焦“小样本电影评分预测”场景，提出融合**静态注意力权重分配**与**权重稀疏化**的轻量化LightGCN模型，解决传统模型参数冗余、部署成本高的问题。

核心创新点
1. 无参静态注意力：基于节点特征余弦相似度，零额外参数，提升用户-电影差异化关联捕捉能力；
2. 轻量化：L1正则化+硬阈值剪枝，零权重比例达97.66%，存储大小从108KB压缩至3.2KB；
3. CPU部署适配：无需GPU，硬件成本降低90%，适配校园影视平台等边缘场景。

关键实验参数
| 模型                | MAE    | RMSE   | 存储大小 | 推理速度(ms/条) |
|---------------------|--------|--------|----------|----------------|
| LightGCN（基线）    | 0.8372 | 1.0463 | 108KB    | 0.0384         |
| 本文模型（Att+WS）  | 0.8276 | 1.0279 | 3.2KB    | 0.0421         |
| 性能提升            | ↓1.15% | ↓1.76% | 压缩97%  | -              |


联系方式
作者：[王尚宇]（福州大学 计算机与大数据学院 大一）
GitHub：[dutyman-sy]
邮箱：[102502135@fzu.edu.cn]
备注：本项目为大一独立完成，欢迎老师、学长学姐指导交流！
开源说明
代码完全开源，可自由复用、修改（注明出处即可）；
论文为科研实践成果，仅用于学术交流，禁止商用；
如果对你有帮助，欢迎点击右上角「Star」支持一下！

### 跑通实验过程
### 安装依赖（终端运行）
```bash
pip install -r requirements.txt

下载地址：MovieLens 100K
操作：下载 ml-100k.zip，解压后文件夹命名为 ml-100k，与代码放在同一目录。

# 第一步：数据预处理（生成preprocessed_data.pt）
python data_preprocess.py

# 第二步：模型训练+测试（输出最终结果）
python model_core.py

MovieLens-LightGCN-FuzhouU/
├── data_preprocess.py    # 数据清洗、二分图构建、数据划分
├── model_core.py         # 模型定义、训练、测试、对比实验
├── requirements.txt      # 依赖库列表（一键安装）
├── paper.pdf             # 论文全文
└── quick_start.ipynb     # 新手教程（含运行）
