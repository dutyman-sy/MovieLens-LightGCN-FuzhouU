# -*- coding: utf-8 -*-
"""
对应论文3章 模型设计与改进、4章 实验验证与分析
包含：本文模型定义、训练/测试、对比模型、稀疏阈值实验
代码来源声明：
- 核心创新模块（注意力+权重稀疏化）为作者原创
- 基础训练框架基于PyTorch官方教程改编，已适配本文实验需求
"""
from data_preprocess import load_and_preprocess_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

# 全局配置（与论文3.3.3、4.1.2节严格对齐）
cfg = {
    "DEVICE": torch.device('cpu'),  # CPU部署（适配边缘设备，论文1.1.1节）
    "IN_FEAT": 20,  # 输入特征维度
    "HIDDEN_FEAT": 64,  # 隐藏层维度
    "OUT_FEAT": 64,  # 输出特征维度
    "L1_LAMBDA": 2e-2,  # L1正则化系数（论文3.3.3节）
    "LR": 2e-4,  # 学习率（论文4.1.2节）
    "BATCH_SIZE": 64,  # 批次大小
    "EPOCHS": 50,  # 最大训练轮次
    "EARLY_STOP_PATIENCE": 5,  # 早停耐心值（论文4.1.2节）
    "SPARSITY_THRESHOLD": 5e-4,  # 稀疏化阈值（论文3.3.3节）
    "SEED": 42  # 随机种子
}
torch.manual_seed(cfg["SEED"])
np.random.seed(cfg["SEED"])

# ==================== 本文所提模型（核心创新：注意力+权重稀疏化）====================
class LightGCN_Sparse_Attention(nn.Module):
    def __init__(self, adj_matrix, node_features):
        super().__init__()
        # 轻量化GCN层（无偏置，减少参数量，论文3.3.3节）
        self.gcn1 = nn.Linear(cfg["IN_FEAT"], cfg["HIDDEN_FEAT"], bias=False)
        self.gcn2 = nn.Linear(cfg["HIDDEN_FEAT"], cfg["OUT_FEAT"], bias=False)
        # 超参数绑定
        self.l1_lambda = cfg["L1_LAMBDA"]
        self.sparsity_threshold = cfg["SPARSITY_THRESHOLD"]
        # 构建注意力加权邻接矩阵（核心创新模块1，论文3.3.2节）
        self.attention_adj = self._build_attention_adj(adj_matrix, node_features).to(cfg["DEVICE"])
        # 权重初始化（Xavier均匀分布，提升训练稳定性）
        nn.init.xavier_uniform_(self.gcn1.weight)
        nn.init.xavier_uniform_(self.gcn2.weight)

    def _build_attention_adj(self, adj_matrix, node_features):
        """构建注意力加权邻接矩阵（论文3.3.2节核心公式）"""
        adj = torch.tensor(adj_matrix, dtype=torch.float32)
        node_feat = torch.tensor(node_features, dtype=torch.float32)

        # 1. 计算节点特征余弦相似度（衡量用户-电影关联强度）
        similarity = F.cosine_similarity(node_feat.unsqueeze(1), node_feat.unsqueeze(0), dim=2)
        # 2. 过滤无效边（仅保留原始邻接矩阵中的非零边）
        adj_mask = (adj > 0).float()
        similarity = similarity * adj_mask
        # 3. Softmax归一化：每行权重和为1
        attention_weights = F.softmax(similarity, dim=1)
        # 4. 构建加权邻接矩阵 + 自环（保留自身特征） + GCN归一化
        adj_att = adj * attention_weights
        adj_att_with_self = adj_att + torch.eye(adj.shape[0])  # 加入自环
        degree = torch.sum(adj_att_with_self, dim=1)  # 度向量
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # 处理孤立节点
        degree_mat = torch.diag(degree_inv_sqrt)
        return degree_mat @ adj_att_with_self @ degree_mat  # 归一化邻接矩阵

    def forward(self, input_emb):
        """前向传播（论文3.3.4节）+ 权重稀疏化（核心创新模块2）"""
        # 两层LightGCN特征聚合
        hidden_emb = self.gcn1(self.attention_adj @ input_emb)
        output_emb = self.gcn2(self.attention_adj @ hidden_emb)
        # 硬阈值稀疏化：剪去绝对值小于阈值的权重（训练时实时剪枝）
        with torch.no_grad():
            for param in self.parameters():
                param.data = torch.where(
                    torch.abs(param.data) < self.sparsity_threshold,
                    torch.zeros_like(param.data),
                    param.data
                )
        return output_emb

    def get_l1_loss(self):
        """计算L1正则化损失（论文3.3.3节公式）"""
        return self.l1_lambda * sum(torch.norm(param, p=1) for param in self.parameters())

    def predict_rating(self, user_embeds, movie_embeds):
        """评分预测（论文3.3.4节公式，归一化至1-5分）"""
        cos_sim = F.cosine_similarity(user_embeds, movie_embeds, dim=1)
        return torch.clamp(1 + 4 * cos_sim, 1.0, 5.0)  # 相似度映射到1-5分

    def count_parameters(self):
        """统计可训练参数量（单位：K）"""
        return sum(param.numel() for param in self.parameters() if param.requires_grad) // 1000

    def calculate_sparsity_ratio(self):
        """计算权重稀疏比例（论文4.4.2节）"""
        total_params = 0
        zero_params = 0
        for param in self.parameters():
            total_params += param.numel()
            zero_params += (torch.abs(param.data) < self.sparsity_threshold).sum().item()
        return (zero_params / total_params) * 100  # 返回百分比

# ==================== 对比模型（论文4.3.2节）====================
class MF(nn.Module):
    """矩阵分解（传统基线模型）"""
    def __init__(self, num_users, num_movies, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.movie_embed = nn.Embedding(num_movies, embed_dim)

    def forward(self, user_ids, movie_ids):
        """评分预测：用户嵌入 × 电影嵌入（点积）"""
        user_emb = self.user_embed(user_ids)
        movie_emb = self.movie_embed(movie_ids)
        return torch.clamp((user_emb * movie_emb).sum(dim=1), 1.0, 5.0)

    def count_parameters(self):
        """统计参数量（单位：K）"""
        return (self.user_embed.num_embeddings + self.movie_embed.num_embeddings) * 64 // 1000

class BasicGCN(nn.Module):
    """基础GCN（无注意力+无稀疏化，基线模型）"""
    def __init__(self, adj_matrix, node_features):
        super().__init__()
        self.gcn1 = nn.Linear(cfg["IN_FEAT"], cfg["HIDDEN_FEAT"])
        self.gcn2 = nn.Linear(cfg["HIDDEN_FEAT"], cfg["OUT_FEAT"])
        self.adj = self._normalize_adj(adj_matrix).to(cfg["DEVICE"])

    def _normalize_adj(self, adj):
        """邻接矩阵归一化（传统GCN方式）"""
        adj = torch.tensor(adj, dtype=torch.float32) + torch.eye(adj.shape[0])
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return torch.diag(degree_inv_sqrt) @ adj @ torch.diag(degree_inv_sqrt)

    def forward(self, input_emb):
        """前向传播：含ReLU激活函数"""
        hidden_emb = F.relu(self.gcn1(self.adj @ input_emb))
        output_emb = self.gcn2(hidden_emb)
        return output_emb

    def predict_rating(self, user_embeds, movie_embeds):
        """评分预测（与本文模型一致）"""
        cos_sim = F.cosine_similarity(user_embeds, movie_embeds, dim=1)
        return torch.clamp(1 + 4 * cos_sim, 1.0, 5.0)

    def count_parameters(self):
        """统计参数量（单位：K）"""
        return sum(param.numel() for param in self.parameters() if param.requires_grad) // 1000

class LightGCN_Base(nn.Module):
    """LightGCN基线（无注意力+无稀疏化，论文4.3.2节）"""
    def __init__(self, adj_matrix, node_features):
        super().__init__()
        self.gcn1 = nn.Linear(cfg["IN_FEAT"], cfg["HIDDEN_FEAT"], bias=False)
        self.gcn2 = nn.Linear(cfg["HIDDEN_FEAT"], cfg["OUT_FEAT"], bias=False)
        self.adj = self._normalize_adj(adj_matrix).to(cfg["DEVICE"])

    def _normalize_adj(self, adj):
        """LightGCN邻接矩阵归一化（无偏置、无激活）"""
        adj = torch.tensor(adj, dtype=torch.float32) + torch.eye(adj.shape[0])
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        return torch.diag(degree_inv_sqrt) @ adj @ torch.diag(degree_inv_sqrt)

    def forward(self, input_emb):
        """前向传播：仅保留特征聚合，无激活函数"""
        hidden_emb = self.gcn1(self.adj @ input_emb)
        output_emb = self.gcn2(self.adj @ hidden_emb)
        return output_emb

    def predict_rating(self, user_embeds, movie_embeds):
        """评分预测（与本文模型一致）"""
        cos_sim = F.cosine_similarity(user_embeds, movie_embeds, dim=1)
        return torch.clamp(1 + 4 * cos_sim, 1.0, 5.0)

    def count_parameters(self):
        """统计参数量（单位：K）"""
        return sum(param.numel() for param in self.parameters() if param.requires_grad) // 1000

# ==================== 训练与测试工具函数 ====================
def train_one_batch(model, batch_data, node_feat_tensor, optimizer, criterion):
    """训练单个批次（提取子函数，简化代码）"""
    user_ids, movie_ids, ratings = batch_data
    user_ids = user_ids.to(cfg["DEVICE"])
    movie_ids = movie_ids.to(cfg["DEVICE"])
    ratings = ratings.to(cfg["DEVICE"])

    optimizer.zero_grad()
    if isinstance(model, MF):
        # MF模型直接输入用户/电影ID
        pred_ratings = model(user_ids, movie_ids - node_feat_tensor.shape[0] + model.user_embed.num_embeddings)
        loss = criterion(pred_ratings, ratings)
    else:
        # GCN类模型先获取节点嵌入，再预测评分
        node_embeds = model(node_feat_tensor)
        user_embeds = node_embeds[user_ids]
        movie_embeds = node_embeds[movie_ids]
        pred_ratings = model.predict_rating(user_embeds, movie_embeds)
        # 仅本文模型添加L1正则化损失
        l1_loss = model.get_l1_loss() if hasattr(model, 'get_l1_loss') else 0.0
        loss = criterion(pred_ratings, ratings) + l1_loss

    loss.backward()
    optimizer.step()
    return loss.item() * len(ratings)

def train_model(model, model_name, data):
    """完整训练流程（含早停机制，论文3.4节）"""
    train_loader, val_loader, test_loader = data["loaders"]
    node_feat_tensor = torch.tensor(data["node_feat"], dtype=torch.float32).to(cfg["DEVICE"])
    optimizer = optim.Adam(model.parameters(), lr=cfg["LR"])
    criterion = nn.MSELoss()  # 回归任务损失函数

    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0

    print(f"\n=== 开始训练 {model_name} ===")
    for epoch in range(1, cfg["EPOCHS"] + 1):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            total_train_loss += train_one_batch(model, batch, node_feat_tensor, optimizer, criterion)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        total_val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                user_ids, movie_ids, ratings = batch
                user_ids = user_ids.to(cfg["DEVICE"])
                movie_ids = movie_ids.to(cfg["DEVICE"])
                ratings = ratings.to(cfg["DEVICE"])

                if isinstance(model, MF):
                    pred_ratings = model(user_ids, movie_ids - node_feat_tensor.shape[0] + model.user_embed.num_embeddings)
                else:
                    node_embeds = model(node_feat_tensor)
                    user_embeds = node_embeds[user_ids]
                    movie_embeds = node_embeds[movie_ids]
                    pred_ratings = model.predict_rating(user_embeds, movie_embeds)

                total_val_mae += torch.abs(pred_ratings - ratings).sum().item()
        avg_val_mae = total_val_mae / len(val_loader.dataset)

        # 打印日志
        print(f"Epoch {epoch:3d} | 训练损失: {avg_train_loss:.4f} | 验证MAE: {avg_val_mae:.4f}")

        # 早停机制
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["EARLY_STOP_PATIENCE"]:
                print(f"早停触发于Epoch {epoch}（连续{cfg['EARLY_STOP_PATIENCE']}轮验证集无提升）")
                break

    # 加载最优模型并测试
    model.load_state_dict(best_model_state)
    test_results = test_model(model, model_name, test_loader, node_feat_tensor, data)
    return test_results

def test_model(model, model_name, test_loader, node_feat_tensor, data):
    """测试模型：返回MAE、RMSE、推理速度、参数量等指标（论文4.3.3节）"""
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            user_ids, movie_ids, ratings = batch
            user_ids = user_ids.to(cfg["DEVICE"])
            movie_ids = movie_ids.to(cfg["DEVICE"])
            ratings = ratings.to(cfg["DEVICE"])

            if isinstance(model, MF):
                pred_ratings = model(user_ids, movie_ids - node_feat_tensor.shape[0] + model.user_embed.num_embeddings)
            else:
                node_embeds = model(node_feat_tensor)
                user_embeds = node_embeds[user_ids]
                movie_embeds = node_embeds[movie_ids]
                pred_ratings = model.predict_rating(user_embeds, movie_embeds)

            # 累计误差
            total_mae += torch.abs(pred_ratings - ratings).sum().item()
            total_rmse += torch.pow(pred_ratings - ratings, 2).sum().item()

    # 计算最终指标
    n_samples = len(test_loader.dataset)
    avg_mae = round(total_mae / n_samples, 4)
    avg_rmse = round(np.sqrt(total_rmse / n_samples), 4)
    infer_speed = round((time.time() - start_time) / n_samples * 1000, 4)  # ms/条
    param_size = model.count_parameters()

    # 构建结果字典
    result_dict = {
        "模型": model_name,
        "MAE": avg_mae,
        "RMSE": avg_rmse,
        "参数规模(K)": param_size,
        "推理速度(ms/条)": infer_speed
    }

    # 新增：本文模型添加稀疏比例指标
    if hasattr(model, 'calculate_sparsity_ratio'):
        result_dict["零权重比例(%)"] = round(model.calculate_sparsity_ratio(), 2)

    # 打印测试结果
    print(f"\n=== {model_name} 测试结果 ===")
    for key, value in result_dict.items():
        print(f"{key}: {value}")
    print("="*50)

    return result_dict

def sparse_threshold_experiment(data):
    """稀疏阈值敏感性实验（论文4.4.2节）"""
    print("\n" + "="*60)
    print("开始执行：稀疏阈值超参数敏感性实验")
    print("="*60)

    thresholds = [1e-4, 2e-4, 5e-4, 8e-4, 1e-3]  # 论文表2阈值范围
    results = []
    node_feat_tensor = torch.tensor(data["node_feat"], dtype=torch.float32).to(cfg["DEVICE"])

    for threshold in thresholds:
        # 更新全局稀疏化阈值
        cfg["SPARSITY_THRESHOLD"] = threshold
        # 初始化模型
        model = LightGCN_Sparse_Attention(data["adj"], data["node_feat"]).to(cfg["DEVICE"])
        model_name = f"阈值_{threshold}"
        # 训练并测试
        test_results = train_model(model, model_name, data)
        # 记录结果
        results.append({
            "稀疏阈值": threshold,
            "MAE": test_results["MAE"],
            "RMSE": test_results["RMSE"],
            "推理速度(ms/条)": test_results["推理速度(ms/条)"],
            "零权重比例(%)": test_results["零权重比例(%)"]
        })

    # 恢复默认阈值
    cfg["SPARSITY_THRESHOLD"] = 5e-4

    # 保存结果到CSV（论文表2数据来源）
    result_df = pd.DataFrame(results)
    result_df.to_csv("稀疏阈值实验结果.csv", index=False, encoding="utf-8-sig")
    print("\n稀疏阈值实验结果已保存至：稀疏阈值实验结果.csv")
    print("\n实验结果汇总：")
    print(result_df.to_string(index=False))

    return result_df

# ==================== 主程序入口（一键运行所有实验）====================
if __name__ == "__main__":
    # 1. 加载预处理数据（优先加载缓存，无缓存则自动预处理）
    print("=== 加载实验数据 ===")
    try:
        data = torch.load("preprocessed_data.pt")
        print("成功加载缓存数据：preprocessed_data.pt")
    except FileNotFoundError:
        print("未找到缓存数据，开始预处理...")
        data = load_and_preprocess_data()
        torch.save(data, "preprocessed_data.pt")
        print("预处理完成并缓存数据")

    # 2. 初始化所有模型（论文4.3.2节对比模型列表）
    models = [
        ("MF（矩阵分解）", MF(data["num_users"], data["num_movies"])),
        ("BasicGCN（基础GCN）", BasicGCN(data["adj"], data["node_feat"])),
        ("LightGCN_Base（LightGCN基线）", LightGCN_Base(data["adj"], data["node_feat"])),
        ("本文模型(Att+WS)", LightGCN_Sparse_Attention(data["adj"], data["node_feat"]))
    ]

    # 3. 执行模型对比实验（论文4.4.1节）
    print("\n" + "="*80)
    print("开始执行：所有模型对比实验")
    print("="*80)
    all_compare_results = []
    for model_name, model in models:
        model = model.to(cfg["DEVICE"])
        test_res = train_model(model, model_name, data)
        all_compare_results.append(test_res)

    # 保存对比结果（论文表1数据来源）
    compare_df = pd.DataFrame(all_compare_results)
    compare_df.to_csv("模型对比结果.csv", index=False, encoding="utf-8-sig")
    print("\n=== 所有模型对比结果汇总 ===")
    print(compare_df.to_string(index=False))
    print("\n对比结果已保存至：模型对比结果.csv")

    # 4. 执行稀疏阈值敏感性实验（论文4.4.2节）
    sparse_threshold_experiment(data)

    print("\n" + "="*80)
    print("所有实验执行完成！")
    print("生成文件：模型对比结果.csv | 稀疏阈值实验结果.csv | preprocessed_data.pt")
    print("="*80)