"""
In this file, we design the SAC agent using a GNN encoder that aims to 
learn the optimal policy for reweighting decoding graph edges in a two
pass MWPM decoder.
"""

import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

##########################
# 1. GRAPH REPLAY BUFFER #
##########################

class GraphReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        data = Data(
            x=torch.tensor(obs["node_features"], dtype=torch.float32),
            edge_index=torch.tensor(obs["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(obs["edge_attr"], dtype=torch.float32),
            action_mask=torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(-1),
            action=torch.tensor(action, dtype=torch.float32).unsqueeze(-1),
            reward=torch.tensor([reward], dtype=torch.float32),
            done=torch.tensor([done], dtype=torch.float32)
        )
        
        if next_obs is not None:
            data.next_x = torch.tensor(next_obs["node_features"], dtype=torch.float32)
            data.next_edge_index = torch.tensor(next_obs["edge_index"], dtype=torch.long)
            data.next_edge_attr = torch.tensor(next_obs["edge_attr"], dtype=torch.float32)
            data.next_action_mask = torch.tensor(next_obs["action_mask"], dtype=torch.float32).unsqueeze(-1)
        else:
            data.next_x = torch.zeros_like(data.x)
            data.next_edge_index = torch.zeros_like(data.edge_index)
            data.next_edge_attr = torch.zeros_like(data.edge_attr)
            data.next_action_mask = torch.zeros_like(data.action_mask)
        
        self.buffer.append(data)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return Batch.from_data_list(samples)

    def __len__(self):
        return len(self.buffer)


##################################
# 2. GNN ACTOR & CRITIC NETWORKS #
##################################

class GNNActor(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, action_mask, evaluate=False):
        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
        
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        
        mu = self.mu_head(x)
        
        # If testing, act deterministically (no exploration noise)
        if evaluate:
            action = torch.tanh(mu) * action_mask
            return action, None
            
        log_std = torch.clamp(self.log_std_head(x), -20, 2)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        
        action = torch.tanh(z) * action_mask
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True) 
        
        return action, log_prob


class GNNCritic(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_dim + 1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, action, batch_index):
        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
        
        xu = torch.cat([x, action], dim=-1)
        
        h = F.relu(self.conv1(xu, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        
        pooled = global_mean_pool(h, batch_index)
        q = self.q_head(pooled)
        return q


####################################
# 3. SOFT ACTOR-CRITIC (SAC) AGENT #
####################################

class SACAgent:
    def __init__(self, node_dim, hidden_dim, lr=3e-4, gamma=0.0, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha 
        
        self.actor = GNNActor(node_dim, hidden_dim).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        
        self.critic1 = GNNCritic(node_dim, hidden_dim).to(self.device)
        self.critic2 = GNNCritic(node_dim, hidden_dim).to(self.device)
        self.critic_optimizer = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        
        self.target_critic1 = GNNCritic(node_dim, hidden_dim).to(self.device)
        self.target_critic2 = GNNCritic(node_dim, hidden_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def select_action(self, obs, evaluate=False):
        """Used during environment interaction. evaluate=True disables Gaussian noise."""
        with torch.no_grad():
            x = torch.tensor(obs["node_features"], dtype=torch.float32).to(self.device)
            edge_index = torch.tensor(obs["edge_index"], dtype=torch.long).to(self.device)
            edge_attr = torch.tensor(obs["edge_attr"], dtype=torch.float32).to(self.device)
            mask = torch.tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(-1).to(self.device)
            
            action, _ = self.actor(x, edge_index, edge_attr, mask, evaluate=evaluate)
            
            return action.cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return 0.0, 0.0 
            
        batch = replay_buffer.sample(batch_size).to(self.device)
        
        x, edge_index, edge_attr, action_mask = batch.x, batch.edge_index, batch.edge_attr, batch.action_mask
        action = batch.action
        
        reward = batch.reward.view(-1, 1)
        done = batch.done.view(-1, 1)
        
        next_x, next_edge_index, next_edge_attr, next_mask = batch.next_x, batch.next_edge_index, batch.next_edge_attr, batch.next_action_mask

        # CRITIC UPDATE
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_x, next_edge_index, next_edge_attr, next_mask)
            next_log_prob_pooled = global_mean_pool(next_log_prob, batch.batch)
            
            target_q1 = self.target_critic1(next_x, next_edge_index, next_edge_attr, next_action, batch.batch)
            target_q2 = self.target_critic2(next_x, next_edge_index, next_edge_attr, next_action, batch.batch)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob_pooled
            
            y = reward + (1 - done) * self.gamma * target_q
            
        current_q1 = self.critic1(x, edge_index, edge_attr, action, batch.batch)
        current_q2 = self.critic2(x, edge_index, edge_attr, action, batch.batch)
        
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ACTOR UPDATE
        new_action, log_prob = self.actor(x, edge_index, edge_attr, action_mask)
        log_prob_pooled = global_mean_pool(log_prob, batch.batch)
        
        q1_new = self.critic1(x, edge_index, edge_attr, new_action, batch.batch)
        q2_new = self.critic2(x, edge_index, edge_attr, new_action, batch.batch)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_prob_pooled - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # TARGET SOFT UPDATE
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return critic_loss.item(), actor_loss.item()

    def save_models(self, path="models/sac_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
        }, path)
        print(f"[*] Models successfully saved to {path}")

    def load_models(self, path="models/sac_model.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.target_critic1.load_state_dict(checkpoint['target_critic1'])
            self.target_critic2.load_state_dict(checkpoint['target_critic2'])
            print(f"[*] Models successfully loaded from {path}")
        else:
            print(f"[!] Warning: No model found at {path}. Proceeding with random initialization.")