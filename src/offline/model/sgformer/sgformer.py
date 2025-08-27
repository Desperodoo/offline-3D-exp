import random
import torch
import torch.nn as nn
from model.sgformer.modules import GCN, SGFormer, SingleHeadAttention, Decoder
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def prepare_batch_data(node_inputs, adj_list):
    """
    准备批量图数据，将节点特征和邻接列表转换为PyG格式
    
    Args:
        node_inputs (torch.Tensor): 节点特征 [batch_size, num_nodes, node_dim]
        adj_list (torch.Tensor): 邻接列表 [batch_size, num_nodes, max_neighbors]
    
    Returns:
        Data: PyG数据对象，包含节点特征和边索引
    """
    batch_size, num_nodes, node_dim = node_inputs.size()
    num_total_nodes = batch_size * num_nodes

    # 展平节点特征: [batch_size * num_nodes, node_dim]
    combined_node_features = node_inputs.view(-1, node_dim)

    # 基于邻接列表构建边索引
    # 找到所有有效邻居 (adj_list >= 0 表示有效邻居)
    batch_idx, row, neigh_idx = torch.where(adj_list >= 0)
    col = adj_list[batch_idx, row, neigh_idx]
    
    # 转换为全局节点索引
    row = row + batch_idx * num_nodes  # 源节点全局索引
    col = col + batch_idx * num_nodes  # 目标节点全局索引

    # 构建稀疏边索引
    sparse_edge_index = SparseTensor(
        row=row, col=col, 
        sparse_sizes=(num_total_nodes, num_total_nodes)
    )
    data = Data(x=combined_node_features, edge_index=sparse_edge_index)

    return data


def split_batch_output(output, batch_size, num_nodes_per_graph):
    """
    将展平的输出重新组织为批量格式
    
    Args:
        output (torch.Tensor): 展平输出 [batch_size * num_nodes, feature_dim]
        batch_size (int): 批量大小
        num_nodes_per_graph (int): 每个图的节点数
    
    Returns:
        torch.Tensor: 批量格式输出 [batch_size, num_nodes, feature_dim]
    """
    return output.view(batch_size, num_nodes_per_graph, -1)


class PolicyNet(nn.Module):
    """
    基于图神经网络的策略网络
    
    使用SGFormer编码器处理图结构，生成动作概率分布
    """
    def __init__(self, node_dim, embedding_dim):
        super(PolicyNet, self).__init__()

        # 图编码器
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        gcn = GCN(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=4,
        )
        self.encoder = SGFormer(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=1,
            num_heads=4,
            use_graph=True,
            gnn=gcn,
        )

        # 状态解码器
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # 指针网络 - 用于选择动作
        self.pointer = SingleHeadAttention(embedding_dim)

        # 参数
        self.embedding_dim = embedding_dim

    def encode_graph(self, node_inputs, adj_list):
        """
        编码图结构
        
        Args:
            node_inputs (torch.Tensor): 节点特征 [batch_size, num_nodes, node_dim]
            adj_list (torch.Tensor): 邻接列表 [batch_size, num_nodes, max_neighbors]
        
        Returns:
            torch.Tensor: 增强节点特征 [batch_size, num_nodes, embedding_dim]
        """
        # 初始节点嵌入: [batch_size, num_nodes, embedding_dim]        
        node_feature = self.initial_embedding(node_inputs)

        # 准备PyG格式数据
        data = prepare_batch_data(node_feature, adj_list)
        data.x = data.x.to(node_inputs.device)
        data.edge_index = data.edge_index.to(node_inputs.device)

        # SGFormer编码: [batch_size * num_nodes, embedding_dim]
        output = self.encoder(data)

        # 重新组织为批量格式: [batch_size, num_nodes, embedding_dim]
        batch_size, num_nodes, _ = node_inputs.size()
        enhanced_node_feature = split_batch_output(output, batch_size, num_nodes)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        """
        解码当前状态
        
        Args:
            enhanced_node_feature (torch.Tensor): 增强节点特征 [batch_size, num_nodes, embedding_dim]
            current_index (torch.Tensor): 当前节点索引 [batch_size, 1, 1]
            node_padding_mask (torch.Tensor): 节点填充掩码 [batch_size, 1, num_nodes]
        
        Returns:
            tuple: (当前节点特征 [batch_size, 1, embedding_dim], 
                   全局特征 [batch_size, 1, embedding_dim])
        """
        # 提取当前节点特征: [batch_size, 1, embedding_dim]
        current_node_feature = torch.gather(
            enhanced_node_feature, 
            1,
            current_index.repeat(1, 1, self.embedding_dim)
        )
            
        # 使用Decoder生成全局特征: [batch_size, 1, embedding_dim]
        global_feature, _ = self.decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask
        )
        
        return current_node_feature, global_feature

    def output_policy(self, current_node_feature, global_feature,
                      enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index=None):
        """
        生成策略输出 (动作logits)
        
        Args:
            current_node_feature (torch.Tensor): 当前节点特征 [batch_size, 1, embedding_dim]
            global_feature (torch.Tensor): 全局特征 [batch_size, 1, embedding_dim]
            enhanced_node_feature (torch.Tensor): 增强节点特征 [batch_size, num_nodes, embedding_dim]
            viewpoints (torch.Tensor): 候选动作节点索引 [batch_size, num_viewpoints, 1]
            viewpoint_padding_mask (torch.Tensor): 动作填充掩码 [batch_size, 1, num_viewpoints]
            current_index (torch.Tensor, optional): 当前节点索引 (未使用)
        
        Returns:
            torch.Tensor: 动作logits [batch_size, num_viewpoints]
        """
        # 组合当前状态特征: [batch_size, 1, embedding_dim]
        current_state_feature = self.current_embedding(
            torch.cat((global_feature, current_node_feature), dim=-1)
        )

        # 提取候选动作节点特征: [batch_size, num_viewpoints, embedding_dim]
        viewpoint_feature = torch.gather(
            enhanced_node_feature, 1,
            viewpoints.repeat(1, 1, self.embedding_dim).long()
        )
        
        # 使用指针网络计算注意力分数: [batch_size, 1, num_viewpoints]
        logits = self.pointer(current_state_feature, viewpoint_feature, viewpoint_padding_mask)
        
        # 压缩维度: [batch_size, num_viewpoints]
        logits = logits.squeeze(1)

        return logits
    
    def forward(self, observation):
        """
        标准前向传播，返回动作logits
        
        Args:
            observation (tuple): 观察数据，包含:
                - node_inputs (torch.Tensor): 节点特征 [batch_size, NODE_PADDING_SIZE, feature_dim]
                - node_padding_mask (torch.Tensor): 节点填充掩码 [batch_size, 1, NODE_PADDING_SIZE]
                - current_index (torch.Tensor): 当前节点索引 [batch_size, 1, 1]
                - viewpoints (torch.Tensor): 候选动作节点索引 [batch_size, VIEWPOINT_PADDING_SIZE, 1]
                - viewpoint_padding_mask (torch.Tensor): 动作填充掩码 [batch_size, 1, VIEWPOINT_PADDING_SIZE]
                - adj_list (torch.Tensor): 邻接列表 [batch_size, NODE_PADDING_SIZE, K_SIZE]
        
        Returns:
            torch.Tensor: 动作logits [batch_size, VIEWPOINT_PADDING_SIZE]
        """
        # 解析观察数据
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        
        # 完整的前向传播流程
        # 1. 图编码: [batch_size, NODE_PADDING_SIZE, embedding_dim]
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        
        # 2. 状态解码: [batch_size, 1, embedding_dim], [batch_size, 1, embedding_dim]
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)

        # 3. 生成动作logits: [batch_size, VIEWPOINT_PADDING_SIZE]
        logits = self.output_policy(current_node_feature, global_feature,
                                enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index)
        return logits

    def forward_ppo(self, observation, action=None):
        """
        PPO算法的前向传播
        
        Args:
            observation (tuple): 观察数据 (详细维度见forward方法)
            action (torch.Tensor, optional): 指定动作 [batch_size]
        
        Returns:
            如果action为None: (action [batch_size], log_prob [batch_size])
            否则: (log_prob [batch_size], entropy [batch_size])
        """
        # 获取动作logits
        logits = self.forward(observation)
        
        # 创建概率分布
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            # 采样动作
            action = probs.sample()
            logp = probs.log_prob(action)
            return action, logp
        else:
            # 计算给定动作的log概率和熵
            return probs.log_prob(action), probs.entropy()
    
    def forward_ppo_incrementally(self, observation, action=None, percentage=0.0):
        """
        渐进式PPO前向传播 (支持部分随机探索)
        
        Args:
            observation (tuple): 观察数据 (详细维度见forward方法)
            action (torch.Tensor, optional): 指定动作 [batch_size]
            percentage (float): 使用策略采样的概率，1-percentage为随机采样概率
        
        Returns:
            同forward_ppo
        """
        # 获取动作logits
        logits = self.forward(observation)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            if random.random() < percentage:
                # 使用策略采样
                action = probs.sample()
            else:
                # 随机采样有效动作
                # 解析viewpoint_padding_mask：0表示有效动作，1表示填充
                _, _, _, _, viewpoint_padding_mask, _ = observation
                valid_indices = torch.where(viewpoint_padding_mask[0, 0] == 0)[0]
                if valid_indices.size(0) > 0:
                    random_idx = torch.randint(0, valid_indices.size(0), (1,), device=logits.device)
                    action = valid_indices[random_idx]
                else:
                    # 备用方案：选择第一个动作
                    action = torch.zeros(1, dtype=torch.long, device=logits.device)
            
            logp = probs.log_prob(action)
            return action, logp
        else:
            return probs.log_prob(action), probs.entropy()
    
    def predict(self, observation):
        """
        确定性预测，选择最高概率的动作
        
        Args:
            observation (tuple): 观察数据 (详细维度见forward方法)
        
        Returns:
            torch.Tensor: 确定性动作 [batch_size]
        """
        # 获取动作logits并选择最高概率的动作
        logits = self.forward(observation)
        deterministic_action = torch.argmax(logits, dim=-1)
        return deterministic_action


class QNet(nn.Module):
    """
    基于图神经网络的Q值网络
    
    计算状态-动作对的Q值，用于critic训练
    """
    def __init__(self, node_dim, embedding_dim):
        super(QNet, self).__init__()

        # 图编码器 (与PolicyNet相同)
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        gcn = GCN(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=4,
        )
        self.encoder = SGFormer(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=1,
            num_heads=4,
            use_graph=True,
            gnn=gcn,
        )
        
        # 状态解码器
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)

        # Q值输出层
        self.q_values_layer = nn.Linear(embedding_dim * 3, 1)

    def encode_graph(self, node_inputs, adj_list):
        """
        编码图结构 (与PolicyNet相同)
        
        Args:
            node_inputs (torch.Tensor): 节点特征 [batch_size, num_nodes, node_dim]
            adj_list (torch.Tensor): 邻接列表 [batch_size, num_nodes, max_neighbors]
        
        Returns:
            torch.Tensor: 增强节点特征 [batch_size, num_nodes, embedding_dim]
        """
        # 初始节点嵌入
        node_feature = self.initial_embedding(node_inputs)
        
        # 准备PyG格式数据并编码
        data = prepare_batch_data(node_feature, adj_list)
        data.x = data.x.to(node_inputs.device)
        data.edge_index = data.edge_index.to(node_inputs.device)

        # SGFormer编码
        output = self.encoder(data)
        
        # 重新组织为批量格式
        batch_size, num_nodes, _ = node_inputs.size()
        enhanced_node_feature = split_batch_output(output, batch_size, num_nodes)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        """
        解码当前状态 (与PolicyNet相同)
        
        Args:
            enhanced_node_feature (torch.Tensor): 增强节点特征 [batch_size, num_nodes, embedding_dim]
            current_index (torch.Tensor): 当前节点索引 [batch_size, 1, 1]
            node_padding_mask (torch.Tensor): 节点填充掩码 [batch_size, 1, num_nodes]
        
        Returns:
            tuple: (当前节点特征 [batch_size, 1, embedding_dim], 
                   全局特征 [batch_size, 1, embedding_dim])
        """
        embedding_dim = enhanced_node_feature.size()[2]
        
        # 提取当前节点特征
        current_node_feature = torch.gather(
            enhanced_node_feature, 1, 
            current_index.repeat(1, 1, embedding_dim)
        )
        
        # 生成全局特征
        global_feature, _ = self.decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask
        )

        return current_node_feature, global_feature
    
    def output_q2(self, current_node_feature, global_feature, enhanced_node_feature, viewpoints):
        """
        计算所有候选动作的Q值
        
        Args:
            current_node_feature (torch.Tensor): 当前节点特征 [batch_size, 1, embedding_dim]
            global_feature (torch.Tensor): 全局特征 [batch_size, 1, embedding_dim]
            enhanced_node_feature (torch.Tensor): 增强节点特征 [batch_size, num_nodes, embedding_dim]
            viewpoints (torch.Tensor): 候选动作节点索引 [batch_size, num_viewpoints, 1]
        
        Returns:
            torch.Tensor: Q值 [batch_size, num_viewpoints, 1]
        """
        embedding_dim = enhanced_node_feature.size()[2]
        num_viewpoints = viewpoints.size()[1]
        
        # 组合当前状态特征: [batch_size, 1, embedding_dim * 2]
        current_state_feature = torch.cat((global_feature, current_node_feature), dim=-1)
        
        # 提取候选动作节点特征: [batch_size, num_viewpoints, embedding_dim]
        viewpoint_feature = torch.gather(
            enhanced_node_feature, 1, 
            viewpoints.repeat(1, 1, embedding_dim).long()
        )
        
        # 组合状态-动作特征: [batch_size, num_viewpoints, embedding_dim * 3]
        action_features = torch.cat(
            (current_state_feature.repeat(1, num_viewpoints, 1), viewpoint_feature), 
            dim=-1
        )
        
        # 计算Q值: [batch_size, num_viewpoints, 1]
        q_values = self.q_values_layer(action_features)
        
        return q_values

    def forward(self, observation):
        """
        前向传播，计算所有动作的Q值
        
        Args:
            observation (tuple): 观察数据，包含:
                - node_inputs (torch.Tensor): 节点特征 [batch_size, NODE_PADDING_SIZE, feature_dim]
                - node_padding_mask (torch.Tensor): 节点填充掩码 [batch_size, 1, NODE_PADDING_SIZE]
                - current_index (torch.Tensor): 当前节点索引 [batch_size, 1, 1]
                - viewpoints (torch.Tensor): 候选动作节点索引 [batch_size, VIEWPOINT_PADDING_SIZE, 1]
                - viewpoint_padding_mask (torch.Tensor): 动作填充掩码 [batch_size, 1, VIEWPOINT_PADDING_SIZE]
                - adj_list (torch.Tensor): 邻接列表 [batch_size, NODE_PADDING_SIZE, K_SIZE]
        
        Returns:
            torch.Tensor: Q值 [batch_size, VIEWPOINT_PADDING_SIZE, 1]
        """
        # 解析观察数据
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        
        # 完整的前向传播流程
        # 1. 图编码: [batch_size, NODE_PADDING_SIZE, embedding_dim]
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        
        # 2. 状态解码: [batch_size, 1, embedding_dim], [batch_size, 1, embedding_dim]
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask
        )

        # 3. 计算Q值: [batch_size, VIEWPOINT_PADDING_SIZE, 1]
        q_values = self.output_q2(
            current_node_feature, global_feature, enhanced_node_feature, viewpoints
        )
        
        return q_values


# 值网络实现 - 专门为IQL设计
class ValueNet(nn.Module):
    """
    IQL算法专用的状态值网络
    
    提供与QNet相同的接口，但只输出状态值而不是状态-动作值
    """
    def __init__(self, node_dim, embedding_dim):
        super(ValueNet, self).__init__()
        
        # 图编码器 (与QNet相同)
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        gcn = GCN(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=4,
        )
        self.encoder = SGFormer(
            in_channels=embedding_dim,
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=1,
            num_heads=4,
            use_graph=True,
            gnn=gcn,
        )
        
        # 状态解码器
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        
        # 值函数输出层
        self.value_head = nn.Linear(embedding_dim * 3, 1)
    
    def encode_graph(self, node_inputs, adj_list=None):
        """
        编码图结构 (与QNet相同)
        
        Args:
            node_inputs (torch.Tensor): 节点特征 [batch_size, num_nodes, node_dim]
            adj_list (torch.Tensor): 邻接列表 [batch_size, num_nodes, max_neighbors]
        
        Returns:
            torch.Tensor: 增强节点特征 [batch_size, num_nodes, embedding_dim]
        """
        node_feature = self.initial_embedding(node_inputs)
        data = prepare_batch_data(node_feature, adj_list)

        data.x = data.x.to(node_inputs.device)
        data.edge_index = data.edge_index.to(node_inputs.device)

        output = self.encoder(data)
        batch_size, num_nodes, _ = node_inputs.size()
        enhanced_node_feature = split_batch_output(output, batch_size, num_nodes)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        """
        解码当前状态 (与QNet相同)
        
        Args:
            enhanced_node_feature (torch.Tensor): 增强节点特征 [batch_size, num_nodes, embedding_dim]
            current_index (torch.Tensor): 当前节点索引 [batch_size, 1, 1]
            node_padding_mask (torch.Tensor): 节点填充掩码 [batch_size, 1, num_nodes]
        
        Returns:
            tuple: (当前节点特征 [batch_size, 1, embedding_dim], 
                   全局特征 [batch_size, 1, embedding_dim])
        """
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(
            enhanced_node_feature, 
            1,
            current_index.repeat(1, 1, embedding_dim)
        )
        global_feature, _ = self.decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask
        )

        return current_node_feature, global_feature

    def forward(self, observation):
        """
        计算状态值
        
        Args:
            observation (tuple): 观察数据，包含:
                - node_inputs (torch.Tensor): 节点特征 [batch_size, NODE_PADDING_SIZE, feature_dim]
                - node_padding_mask (torch.Tensor): 节点填充掩码 [batch_size, 1, NODE_PADDING_SIZE]
                - current_index (torch.Tensor): 当前节点索引 [batch_size, 1, 1]
                - viewpoints (torch.Tensor): 候选动作 [batch_size, VIEWPOINT_PADDING_SIZE, 1] (未使用)
                - viewpoint_padding_mask (torch.Tensor): 动作填充掩码 [batch_size, 1, VIEWPOINT_PADDING_SIZE] (未使用)
                - adj_list (torch.Tensor): 邻接列表 [batch_size, NODE_PADDING_SIZE, K_SIZE]
            
        Returns:
            torch.Tensor: 状态值 [batch_size]
        """        
        # 解析观察数据
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        
        # 1. 图编码: [batch_size, NODE_PADDING_SIZE, embedding_dim]
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        
        # 2. 状态解码: [batch_size, 1, embedding_dim], [batch_size, 1, embedding_dim]
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask
        )
        
        # 3. 特征提取和组合
        # 压缩维度: [batch_size, embedding_dim]
        current_node_feature = current_node_feature.squeeze(1)
        global_feature = global_feature.squeeze(1)
        
        # 计算图级特征: [batch_size, embedding_dim]
        graph_feature = torch.mean(enhanced_node_feature, dim=1, keepdim=False)
        
        # 组合所有特征: [batch_size, embedding_dim * 3]
        combined_features = torch.cat(
            (current_node_feature, global_feature, graph_feature), dim=-1
        )
        
        # 4. 输出状态值: [batch_size]
        value = self.value_head(combined_features).squeeze(-1)
        
        return value
