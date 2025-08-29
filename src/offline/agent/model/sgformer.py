import random
import torch
import torch.nn as nn
from .modules import GCN, SGFormer, SingleHeadAttention, Decoder
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def prepare_batch_data(node_inputs, adj_list):
    batch_size, num_nodes, node_dim = node_inputs.size()
    num_total_nodes = batch_size * num_nodes

    combined_node_features = node_inputs.view(-1, node_dim)

    # 基于 adj_list 构建边索引
    batch_idx, row, neigh_idx = torch.where(adj_list >= 0)
    col = adj_list[batch_idx, row, neigh_idx]
    row = row + batch_idx * num_nodes
    col = col + batch_idx * num_nodes

    sparse_edge_index = SparseTensor(row=row, col=col, sparse_sizes=(num_total_nodes, num_total_nodes))
    data = Data(x=combined_node_features, edge_index=sparse_edge_index)

    return data


def split_batch_output(output, batch_size, num_nodes_per_graph):
    return output.view(batch_size, num_nodes_per_graph, -1)


class PolicyNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(PolicyNet, self).__init__()

        # graph encoder
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

        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # pointer
        self.pointer = SingleHeadAttention(embedding_dim)

        # parameters
        self.embedding_dim = embedding_dim

    def encode_graph(self, node_inputs, adj_list):
        node_feature = self.initial_embedding(node_inputs)
        data = prepare_batch_data(node_feature, adj_list)
        
        data.x = data.x.to(node_inputs.device)
        data.edge_index = data.edge_index.to(node_inputs.device)


        output = self.encoder(data)  # Shape: (batch_size * num_nodes, enhanced_dim)
        # 可视化注意力图

        batch_size, num_nodes, _ = node_inputs.size()
        enhanced_node_feature = split_batch_output(
            output, batch_size, num_nodes
        )  # Shape: (batch_size, num_nodes, enhanced_dim)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        # 提取当前节点特征作为query和enhanced node features作为key/value
        current_node_feature = torch.gather(
            enhanced_node_feature, 
            1,
            current_index.repeat(1, 1, self.embedding_dim))
            
        global_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)
        return current_node_feature, global_feature

    def output_policy2(self, current_node_feature, global_feature,
                      enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index=None):
        current_state_feature = self.current_embedding(torch.cat((global_feature,
                                                                current_node_feature), dim=-1))

        viewpoint_feature = torch.gather(enhanced_node_feature, 1,
                                           viewpoints.repeat(1, 1, self.embedding_dim).long())
        # 当前状态特征作为 query 和 viewpoint features 作为 key/value
        logp = self.pointer(current_state_feature, viewpoint_feature, viewpoint_padding_mask)
        logp = logp.squeeze(1)

        return logp  # batch_size, n_viewpoints
    
    def forward_ppo(self, observation, action=None):
        """返回动作和对数概率，用于PPO等算法"""
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)

        logits = self.output_policy2(current_node_feature, global_feature,
                                enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()
            logp = probs.log_prob(action)
            return action, logp
        else:
            return probs.log_prob(action), probs.entropy()
    
    def forward_ppo_incrementally(self, observation, action=None, percentage=0.0):
        """返回动作和对数概率，用于PPO等算法"""
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)

        logits = self.output_policy2(current_node_feature, global_feature,
                                enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            if random.random() < percentage:
                action = probs.sample()  # action shape: torch.Size([1])
            else:
                # Get indices of valid actions (where mask is 1)
                valid_indices = torch.where(viewpoint_padding_mask[0, 0] == 0)[0]
                # Randomly select one valid action
                if valid_indices.size(0) > 0:
                    random_idx = torch.randint(0, valid_indices.size(0), (1,), device=logits.device)
                    action = valid_indices[random_idx]
                else:
                    # Fallback if no valid actions found
                    action = torch.zeros(1, dtype=torch.long, device=logits.device)
            logp = probs.log_prob(action)
            return action, logp
        else:
            return probs.log_prob(action), probs.entropy()
        
    def forward_ppo_randomly(self, observation, action=None, percentage=0.0):
        """返回动作和对数概率，用于PPO等算法"""
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)

        logits = self.output_policy2(current_node_feature, global_feature,
                                enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            if random.random() < percentage:
                action = probs.sample()  # action shape: torch.Size([1])
            else:
                # Get indices of valid actions (where mask is 1)
                valid_indices = torch.where(viewpoint_padding_mask[0, 0] == 0)[0]
                # Randomly select one valid action
                if valid_indices.size(0) > 0:
                    random_idx = torch.randint(0, valid_indices.size(0), (1,), device=logits.device)
                    action = valid_indices[random_idx]
                else:
                    # Fallback if no valid actions found
                    action = torch.zeros(1, dtype=torch.long, device=logits.device)
            logp = probs.log_prob(action)
            return action, logp
        else:
            return probs.log_prob(action), probs.entropy()
    
    def forward(self, observation):
        """直接返回原始logits，用于IQL等算法"""
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)

        logits = self.output_policy2(current_node_feature, global_feature,
                                enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index)
        return logits
    
    def predict(self, observation):
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        current_node_feature, global_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)
        logits = self.output_policy2(current_node_feature, global_feature,
                                enhanced_node_feature, viewpoints, viewpoint_padding_mask, current_index)
        deterministic_action = torch.argmax(logits, dim=-1)
        return deterministic_action


class QNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(QNet, self).__init__()

        # graph encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        # ----------- SGFormer ----------- #
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
        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)

        self.q_values_layer = nn.Linear(embedding_dim * 3, 1)

    def encode_graph(self, node_inputs, adj_list):
        # print(f"[encode_graph] node_inputs shape: {node_inputs.shape}")
        # print(f"[encode_graph] adj_list shape: {adj_list.shape}")
        
        node_feature = self.initial_embedding(node_inputs)
        # print(f"[encode_graph] node_feature after initial_embedding: {node_feature.shape}")
        
        data = prepare_batch_data(node_feature, adj_list)
        # print(f"[encode_graph] data.x shape: {data.x.shape}")
        # print(f"[encode_graph] data.edge_index shape: {data.edge_index.sizes()}")

        data.x = data.x.to(node_inputs.device)
        data.edge_index = data.edge_index.to(node_inputs.device)

        output = self.encoder(data)  # Shape: (batch_size * num_nodes, enhanced_dim)
        # print(f"[encode_graph] encoder output shape: {output.shape}")
        
        batch_size, num_nodes, _ = node_inputs.size()
        enhanced_node_feature = split_batch_output(
            output, batch_size, num_nodes
        )  # Shape: (batch_size, num_nodes, enhanced_dim)
        # print(f"[encode_graph] enhanced_node_feature shape: {enhanced_node_feature.shape}")

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        # print(f"[decode_state] enhanced_node_feature shape: {enhanced_node_feature.shape}")
        # print(f"[decode_state] current_index shape: {current_index.shape}")
        # print(f"[decode_state] node_padding_mask shape: {node_padding_mask.shape}")
        
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))
        # print(f"[decode_state] current_node_feature shape: {current_node_feature.shape}")
        
        global_feature, _ = self.decoder(
            current_node_feature,
            enhanced_node_feature,
            node_padding_mask
        )
        # print(f"[decode_state] global_feature shape: {global_feature.shape}")

        return current_node_feature, global_feature  # batch_size, 1, embedding_dim
    
    def output_q2(self, current_node_feature, global_feature, enhanced_node_feature, viewpoints):
        # print(f"[output_q2] current_node_feature shape: {current_node_feature.shape}")
        # print(f"[output_q2] global_feature shape: {global_feature.shape}")
        # print(f"[output_q2] enhanced_node_feature shape: {enhanced_node_feature.shape}")
        # print(f"[output_q2] viewpoints shape: {viewpoints.shape}")
        
        embedding_dim = enhanced_node_feature.size()[2]
        k_size = viewpoints.size()[1]
        current_state_feature = torch.cat((global_feature, current_node_feature), dim=-1)
        # print(f"[output_q2] current_state_feature shape: {current_state_feature.shape}")
        
        viewpoint_feature = torch.gather(enhanced_node_feature, 1, viewpoints.repeat(1, 1, embedding_dim).long())
        # print(f"[output_q2] viewpoint_feature shape: {viewpoint_feature.shape}")
        
        action_features = torch.cat((current_state_feature.repeat(1, k_size, 1), viewpoint_feature), dim=-1)
        # print(f"[output_q2] action_features shape: {action_features.shape}")
        
        q_values = self.q_values_layer(action_features) # batch_size, n_viewpoints, 1
        # print(f"[output_q2] q_values shape: {q_values.shape}")
        
        return q_values

    def forward(self, observation):
        # print(f"[forward] Starting forward pass")
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        # print(f"[forward] node_inputs shape: {node_inputs.shape}")
        # print(f"[forward] node_padding_mask shape: {node_padding_mask.shape}")
        # print(f"[forward] current_index shape: {current_index.shape}")
        # print(f"[forward] viewpoints shape: {viewpoints.shape}")
        # print(f"[forward] viewpoint_padding_mask shape: {viewpoint_padding_mask.shape}")
        # print(f"[forward] adj_list shape: {adj_list.shape}")
        
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)
        current_node_feature, global_feature = self.decode_state(enhanced_node_feature, current_index, node_padding_mask)
        # print(f"[forward] squeezed current_node_feature shape: {current_node_feature.shape}")
        # print(f"[forward] squeezed global_feature shape: {global_feature.shape}")

        q_values = self.output_q2(current_node_feature, global_feature, enhanced_node_feature, viewpoints)
        # print(f"[forward] Final q_values shape: {q_values.shape}")
        
        return q_values  # batch_size, n_viewpoints, 1


# 值网络实现 - 专门为IQL设计
class ValueNet(nn.Module):
    """
    IQL算法专用的值网络
    
    提供与QNet相同的接口，但只输出状态值而不是状态-动作值
    """
    def __init__(self, node_dim, embedding_dim):
        super(ValueNet, self).__init__()
        # 复用QNet的大部分结构，但最终只输出一个值        
        # graph encoder - 与QNet相同
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
        
        # decoder - 与QNet相同 
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=4, n_layer=1)
        
        # 输出单个值的值函数
        self.value_head = nn.Linear(embedding_dim * 3, 1)
    
    def encode_graph(self, node_inputs, adj_list=None):
        node_feature = self.initial_embedding(node_inputs)
        data = prepare_batch_data(node_feature, adj_list)

        data.x = data.x.to(node_inputs.device)
        data.edge_index = data.edge_index.to(node_inputs.device)

        output = self.encoder(data)  # Shape: (batch_size * num_nodes, enhanced_dim)
        batch_size, num_nodes, _ = node_inputs.size()
        enhanced_node_feature = split_batch_output(
            output, batch_size, num_nodes
        )  # Shape: (batch_size, num_nodes, enhanced_dim)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(
            enhanced_node_feature, 
            1,
            current_index.repeat(1, 1, embedding_dim))
        global_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask, )

        return current_node_feature, global_feature  # batch_size, 1, embedding_dim    

    def forward(self, observation):
        """
        计算状态值
        
        参数:
            observation: 图结构观察数据的列表
            
        返回:
            value: 状态值
        """        
        node_inputs, node_padding_mask, current_index, viewpoints, viewpoint_padding_mask, adj_list = observation
        
        # # 编码图结构
        # node_feature = self.initial_embedding(node_inputs)
        # data = prepare_batch_data(node_feature, adj_list)
        
        # data.x = data.x.to(node_inputs.device)
        # data.edge_index = data.edge_index.to(node_inputs.device)
        
        # output = self.encoder(data)
        # batch_size, num_nodes, _ = node_inputs.size()
        # enhanced_node_feature = split_batch_output(output, batch_size, num_nodes)
        
        enhanced_node_feature = self.encode_graph(node_inputs, adj_list)

        
        # # 解码当前状态
        # embedding_dim = enhanced_node_feature.size()[2]
        # current_node_feature = torch.gather(
        #     enhanced_node_feature, 
        #     1,
        #     current_index.repeat(1, 1, embedding_dim))
        # global_feature, _ = self.decoder(
        #     current_node_feature,
        #     enhanced_node_feature,
        #     node_padding_mask, 
        #     adj_list=adj_list
        # )
        
        current_node_feature, global_feature = self.decode_state(enhanced_node_feature, current_index, node_padding_mask)
        current_node_feature = current_node_feature.squeeze(1)
        global_feature = global_feature.squeeze(1)
        
        # 组合特征并输出状态值
        graph_feature = torch.mean(enhanced_node_feature, dim=1, keepdim=False)  # batch_size, embedding_dim 
        combined_features = torch.cat((current_node_feature, global_feature, graph_feature), dim=-1)
        value = self.value_head(combined_features).squeeze(-1)
        
        # 确保输出是 [B, 1] 而不是 [B]
        # return value.unsqueeze(1) 
        
        return value  # 确保这个输出是 [B, 1] 或 [B]