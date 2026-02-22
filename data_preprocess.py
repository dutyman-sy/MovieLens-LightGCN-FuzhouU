# -*- coding: utf-8 -*-
"""
对应论文4.2节 实验数据集与预处理
功能：生成用户-电影二分图、节点特征、训练/验证/测试数据
代码来源声明：
- 核心逻辑为作者原创，基于PyTorch官方数据处理模板改编
- 适配MovieLens 100K数据集，确保可复现性
"""
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader

# 配置参数（与论文4.1.2节严格对齐）
class DataConfig:
    DATA_PATH = "ml-100k"  # 数据集文件夹路径
    SEED = 42  # 随机种子
    TRAIN_RATIO = 0.8  # 训练集比例
    VAL_RATIO = 0.1  # 验证集比例
    IN_FEAT = 20  # 节点特征维度（论文3.3.1节）

cfg = DataConfig()
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

def load_and_preprocess_data():
    """加载并预处理数据，返回训练/验证/测试加载器、节点特征、邻接矩阵等"""
    # 1. 加载评分数据（MovieLens 100K）
    u_data = pd.read_csv(
        f"{cfg.DATA_PATH}/u.data",
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )

    # 2. 数据清洗：剔除冷门用户/电影（评分记录<5条）
    user_rat_count = u_data['user_id'].value_counts()
    valid_users = user_rat_count[user_rat_count >= 5].index
    movie_rat_count = u_data['movie_id'].value_counts()
    valid_movies = movie_rat_count[movie_rat_count >= 5].index
    u_data = u_data[
        (u_data['user_id'].isin(valid_users)) &
        (u_data['movie_id'].isin(valid_movies))
    ]

    # 3. 重新编码用户/电影ID（连续索引，适配模型输入）
    u_data['user_id'] = u_data['user_id'].astype('category').cat.codes.astype(np.int64)
    u_data['movie_id'] = u_data['movie_id'].astype('category').cat.codes.astype(np.int64)
    num_users = u_data['user_id'].nunique()
    num_movies = u_data['movie_id'].nunique()
    num_nodes = num_users + num_movies  # 总节点数（用户+电影）

    # 4. 生成节点特征（对应论文3.3.1节）
    # 4.1 用户特征：观影频次、平均评分（归一化至[0,1]）
    user_feat = []
    for user in range(num_users):
        user_data = u_data[u_data['user_id'] == user]
        freq = len(user_data) / u_data['user_id'].value_counts().max()  # 频次归一化
        avg_rating = user_data['rating'].mean() / 5.0  # 评分归一化
        user_feat.append([freq, avg_rating])
    user_feat = np.array(user_feat, dtype=np.float32)

    # 4.2 电影特征：类型编码（独热）、平均评分（归一化至[0,1]）
    movie_feat = []
    movie_info = pd.read_csv(
        f"{cfg.DATA_PATH}/u.item",
        sep='|',
        names=['movie_id', 'title', 'release_date'] + [f'genre_{i}' for i in range(19)],
        encoding='latin-1',
        usecols=['movie_id'] + [f'genre_{i}' for i in range(19)]
    )
    movie_info['movie_id'] = movie_info['movie_id'].astype('category').cat.codes.astype(np.int64)
    for movie in range(num_movies):
        # 电影类型独热编码（19维）
        genre = movie_info[movie_info['movie_id'] == movie][[f'genre_{i}' for i in range(19)]].values[0]
        # 电影平均评分归一化
        movie_data = u_data[u_data['movie_id'] == movie]
        avg_rating = movie_data['rating'].mean() / 5.0
        # 拼接特征并补齐至20维
        feat = np.hstack([genre, avg_rating]).astype(np.float32)
        if len(feat) < cfg.IN_FEAT:
            feat = np.pad(feat, (0, cfg.IN_FEAT - len(feat)), 'constant')
        movie_feat.append(feat[:cfg.IN_FEAT])
    movie_feat = np.array(movie_feat, dtype=np.float32)

    # 4.3 合并用户+电影特征
    node_feat = np.vstack([user_feat, movie_feat])

    # 5. 构建邻接矩阵（对应论文2.1节，用户-电影二分图）
    u_ids = u_data['user_id'].values
    m_ids = u_data['movie_id'].values + num_users  # 电影节点ID偏移（避免与用户ID冲突）
    ratings = (u_data['rating'].values / 5.0).astype(np.float32)  # 评分归一化至[0.2,1.0]
    adj = coo_matrix(
        (ratings, (u_ids, m_ids)),
        shape=(num_nodes, num_nodes)
    ).toarray()
    adj = adj + adj.T  # 邻接矩阵对称化（无向图）

    # 6. 划分训练集/验证集/测试集（8:1:1）
    u_data_shuffle = u_data.sample(frac=1, random_state=cfg.SEED)
    train_size = int(cfg.TRAIN_RATIO * len(u_data_shuffle))
    val_size = int(cfg.VAL_RATIO * len(u_data_shuffle))
    train_data = u_data_shuffle.iloc[:train_size]
    val_data = u_data_shuffle.iloc[train_size:train_size+val_size]
    test_data = u_data_shuffle.iloc[train_size+val_size:]

    # 7. 转换为Tensor格式
    def to_tensor(data):
        user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
        movie_ids = torch.tensor(data['movie_id'].values + num_users, dtype=torch.long)
        ratings = torch.tensor(data['rating'].values, dtype=torch.float32)
        return user_ids, movie_ids, ratings

    train_users, train_movies, train_ratings = to_tensor(train_data)
    val_users, val_movies, val_ratings = to_tensor(val_data)
    test_users, test_movies, test_ratings = to_tensor(test_data)

    # 8. 构建DataLoader（批次大小64，与论文4.1.2节一致）
    train_dataset = TensorDataset(train_users, train_movies, train_ratings)
    val_dataset = TensorDataset(val_users, val_movies, val_ratings)
    test_dataset = TensorDataset(test_users, test_movies, test_ratings)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"预处理完成：用户{num_users}个 + 电影{num_movies}部")
    print(f"训练集{len(train_data)}条 | 验证集{len(val_data)}条 | 测试集{len(test_data)}条")

    return {
        "node_feat": node_feat,  # 节点特征矩阵 (N+M, 20)
        "adj": adj,  # 邻接矩阵 (N+M, N+M)
        "loaders": (train_loader, val_loader, test_loader),  # 数据加载器
        "num_users": num_users,  # 用户数量
        "num_movies": num_movies  # 电影数量
    }

if __name__ == "__main__":
    # 执行预处理并保存数据（后续训练直接加载，无需重复预处理）
    processed_data = load_and_preprocess_data()
    torch.save(processed_data, "preprocessed_data.pt")
    print("预处理数据已保存至：preprocessed_data.pt")