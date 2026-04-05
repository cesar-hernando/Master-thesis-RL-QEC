"""
In this file, we design the SAC agent using a GNN encoder that aims to 
learn the optimal policy for reweighting decoding graph edges in a two
pass MWPM decoder.
"""

import os
import random
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

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
        #self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, x, edge_index, edge_attr, action_mask, evaluate=False):
        # Use edge_attr as edge weights if they exist, otherwise treat as unweighted
        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
        
        # Capture the neighbor context
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))

        # Mask out inactive edges (where action_mask == 0) 
        active_mask = action_mask.squeeze(-1).bool()

        # Compute action and the log probability for active edges only
        h_active = h[active_mask]
        mu = self.mu_head(h_active)
        full_action = torch.zeros((x.size(0), 1), device=x.device)
        full_log_prob = torch.zeros((x.size(0), 1), device=x.device)
        
        # If testing, act deterministically (no exploration noise)
        if evaluate:
            full_action[active_mask] = torch.tanh(mu)
            return full_action, None

        # Add Gaussian noise for exploration during training    
        log_std = torch.clamp(self.log_std_head(h_active), -20, 2)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        
        # Apply tanh squashing and re-scale to action bounds (if any)
        action = torch.tanh(z) * action_mask
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Place the computed actions and log probabilities back into the full tensor
        full_action[active_mask] = action
        full_log_prob[active_mask] = log_prob
        
        return full_action, full_log_prob


class GNNCritic(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(node_dim + 1, hidden_dim)
        #self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, action, batch_index):
        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
        
        # Combine state and action
        xu = torch.cat([x, action], dim=-1)
        
        # Extract graph context
        h = F.relu(self.conv1(xu, edge_index, edge_weight=edge_weight))
        
        # Global Pool and Predict Q
        pooled = global_mean_pool(h, batch_index)
        q = self.q_head(pooled)
        return q


####################################
# 3. SOFT ACTOR-CRITIC (SAC) AGENT #
####################################

class SACAgent:
    def __init__(self, node_dim, hidden_dim, lr=1e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha 
        
        self.actor = GNNActor(node_dim, hidden_dim).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        
        self.critic1 = GNNCritic(node_dim, hidden_dim).to(self.device)
        self.critic2 = GNNCritic(node_dim, hidden_dim).to(self.device)
        self.critic_optimizer = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

        # Target entropy is -1.0 because we use global_mean_pool (average over edges)
        self.target_entropy = -1.0 
        
        # Initialize log_alpha as a learnable PyTorch tensor
        self.log_alpha = torch.tensor([np.log(alpha)], dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)
        
        # Conditionally initialize target networks ONLY if gamma > 0
        if self.gamma > 0.0:
            self.target_critic1 = GNNCritic(node_dim, hidden_dim).to(self.device)
            self.target_critic2 = GNNCritic(node_dim, hidden_dim).to(self.device)
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        else:
            self.target_critic1 = None
            self.target_critic2 = None

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
        
        # CRITIC UPDATE
        if self.gamma > 0.0:
            # Full RL Mode: Compute target Q-values with future states
            next_x, next_edge_index, next_edge_attr, next_mask = batch.next_x, batch.next_edge_index, batch.next_edge_attr, batch.next_action_mask

            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_x, next_edge_index, next_edge_attr, next_mask)
                # Calculate average entropy for the NEXT state
                next_log_prob_sum = global_add_pool(next_log_prob, batch.batch)
                next_active_count = global_add_pool(next_mask, batch.batch)
                next_log_prob_avg = next_log_prob_sum / torch.clamp(next_active_count, min=1.0)
                
                target_q1 = self.target_critic1(next_x, next_edge_index, next_edge_attr, next_action, batch.batch)
                target_q2 = self.target_critic2(next_x, next_edge_index, next_edge_attr, next_action, batch.batch)
                
                # Use tuned alpha and the average entropy ---
                current_alpha = self.log_alpha.exp().detach()
                target_q = torch.min(target_q1, target_q2) - current_alpha * next_log_prob_avg
                
                y = reward + (1 - done) * self.gamma * target_q
        else:
            # Contextual Bandit Mode: Fast path, target is just the immediate reward
            y = reward
            
        current_q1 = self.critic1(x, edge_index, edge_attr, action, batch.batch)
        current_q2 = self.critic2(x, edge_index, edge_attr, action, batch.batch)
        
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ACTOR UPDATE
        alpha = self.log_alpha.exp().detach()
        new_action, log_prob = self.actor(x, edge_index, edge_attr, action_mask)

        # MASKED MEAN POOLING 
        # 1. Sum the log probabilities per graph in the batch
        log_prob_sum = global_add_pool(log_prob, batch.batch)
        
        # 2. Count how many active edges (mask == 1) exist per graph in the batch
        active_nodes_per_graph = global_add_pool(action_mask, batch.batch)
        
        # 3. Prevent division by zero (if a graph has no active edges, divide by 1)
        safe_divisor = torch.clamp(active_nodes_per_graph, min=1.0)
        
        # 4. Calculate the true average entropy of ONLY the active edges
        log_prob_pooled = log_prob_sum / safe_divisor
        
        q1_new = self.critic1(x, edge_index, edge_attr, new_action, batch.batch)
        q2_new = self.critic2(x, edge_index, edge_attr, new_action, batch.batch)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha * log_prob_pooled - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ALPHA AUTOTUNER UPDATE
        # Alpha tries to push log_prob_pooled to exactly match the target_entropy
        alpha_loss = (-self.log_alpha.exp() * (log_prob_pooled.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            # np.log(0.005) ≈ -5.29, np.log(0.2) ≈ -1.61
            self.log_alpha.clamp_(np.log(0.00005), np.log(0.2))

        # TARGET SOFT UPDATE
        if self.gamma > 0.0:
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        return critic_loss.item(), actor_loss.item()

    def save_models(self, path="models/sac_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict()
        }
        # Only save target networks if they exist
        if self.gamma > 0.0:
            save_dict['target_critic1'] = self.target_critic1.state_dict()
            save_dict['target_critic2'] = self.target_critic2.state_dict()
            
        torch.save(save_dict, path)
        print(f"[*] Models successfully saved to {path}")

    def load_models(self, path="models/sac_model.pth"):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic1.load_state_dict(checkpoint['critic1'])
            self.critic2.load_state_dict(checkpoint['critic2'])

            # Safely load target networks if both the agent and the checkpoint support them
            if self.gamma > 0.0 and 'target_critic1' in checkpoint:
                self.target_critic1.load_state_dict(checkpoint['target_critic1'])
                self.target_critic2.load_state_dict(checkpoint['target_critic2'])

            print(f"[*] Models successfully loaded from {path}")
        else:
            print(f"[!] Warning: No model found at {path}. Proceeding with random initialization.")