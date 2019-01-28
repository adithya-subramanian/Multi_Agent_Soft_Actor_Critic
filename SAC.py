import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic_Q, Critic_V

import torch
import torch.nn.functional as F
import torch.optim as optim

def variable_hook(grad):
    print('variable hook')
    print('grad', grad)
    return grad

BUFFER_SIZE = int(1e5)          # Replay buffer size
BATCH_SIZE = 64                 # Minibatch size
GAMMA = 0.99                    # Discount factor
TAU = 0.005                     # Soft update of target parameters
LR_ACTOR = 3e-4                 # Learning rate of the actor 
LR_CRITIC = 3e-4                # Learning rate of the critic
WEIGHT_DECAY = 0                # L2 weight decay
ALPHA = 0.0025                  # Entropy weight parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed = 0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        # Initializes state size of the environment
        self.state_size = state_size
        # Initializes action size of the environment
        self.action_size = action_size
        # Intialzes seed
        self.seed = random.seed(random_seed)
        # Initializes the Actor model for agent 0
        self.actor_local0 = Actor(state_size,action_size).to(device)
        # Initializes the Actor model for agent 1
        self.actor_local1  = Actor(state_size,action_size).to(device)
        # Initializes the Q-Critic model 1 for agent 0
        self.critic_q_1_local0 = Critic_Q(state_size, action_size, self.seed).to(device)
        # Initializes the Q-Critic model 1 for agent 1
        self.critic_q_1_local1 = Critic_Q(state_size, action_size, self.seed).to(device)
        # Initializes the Q-Critic model 2 for agent 0
        self.critic_q_2_local0 = Critic_Q(state_size, action_size, self.seed).to(device)
        # Initializes the Q-Critic model 2 for agent 1
        self.critic_q_2_local1 = Critic_Q(state_size, action_size, self.seed).to(device)
        # Initializes the V-Critic local model for agent 0
        self.critic_v_local0 = Critic_V(state_size, self.seed).to(device)
        # Initializes the V-Critic local model for agent 1
        self.critic_v_local1 = Critic_V(state_size, self.seed).to(device)
        # Initializes the V-Critic target model for agent 0
        self.critic_v_target0 = Critic_V(state_size, self.seed).to(device)
        # Initializes the V-Critic local model for agent 1
        self.critic_v_target1 = Critic_V(state_size, self.seed).to(device)
        # Initializes the Adam optimizer for agent 0's actor parameters
        self.actor_optimizer0 = optim.Adam(self.actor_local0.parameters(),lr=LR_ACTOR,weight_decay = WEIGHT_DECAY)
        # Initializes the Adam optimizer for agent 1's actor parameters
        self.actor_optimizer1 = optim.Adam(self.actor_local1.parameters(),lr=LR_ACTOR,weight_decay = WEIGHT_DECAY)
        # Initializes the Adam optimizer for agent 0's q1-critic parameters
        self.critic_q_1_optimizer0 = optim.Adam(self.critic_q_1_local0.parameters(),lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)
        # Initializes the Adam optimizer for agent 1's q1-critic parameters
        self.critic_q_1_optimizer1 = optim.Adam(self.critic_q_1_local1.parameters(),lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)
        # Initializes the Adam optimizer for agent 0's q2-critic parameters
        self.critic_q_2_optimizer0 = optim.Adam(self.critic_q_2_local0.parameters(),lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)
        # Initializes the Adam optimizer for agent 1's q2-critic parameters
        self.critic_q_2_optimizer1 = optim.Adam(self.critic_q_2_local1.parameters(),lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)
        # Initializes the Adam optimizer for agent 0's v-critic parameters
        self.critic_v_optimizer0 = optim.Adam(self.critic_v_local0.parameters(),lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)
        # Initializes the Adam optimizer for agent 1's v-critic parameters
        self.critic_v_optimizer1 = optim.Adam(self.critic_v_local1.parameters(),lr=LR_CRITIC,weight_decay = WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed) # Initializing the replay buffer
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)    # Adding an experience to the replay buffer

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:                          
            experiences = self.memory.sample()                      # Sampling episode from the replay buffer
            self.learn(experiences, GAMMA)                     # Calls the learn function responsible for loss computation and updating the model's parameter

    def act(self,state):
        """Returns actions for given state as per current policy
        ."""
        # list containing actions of all the model for current state
        actions = []
        # state as viewed by agent 0
        states0 = torch.from_numpy(state[0]).float().to(device)
        # state as viewed by agent 1
        states1 = torch.from_numpy(state[1]).float().to(device)
        # setting the actor local to eval so as to avoid gradient updates
        self.actor_local0.eval()
        self.actor_local1.eval()
        with torch.no_grad():
            # action performed agent 0 when it is in state 0 following the current policy 
            action0 = self.actor_local0(states0)[0].cpu().data.numpy()
            # action performed agent 1 when it is in state 1 following the current policy
            action1 = self.actor_local1(states1)[0].cpu().data.numpy()
            # clipping action in the range
            actions.append(np.clip(action0, -1, 1))
            # clipping action in the range
            actions.append(np.clip(action1, -1, 1))
        
        # setting agent 0's weights back to trainable weights
        self.actor_local0.train()
        # setting agent 1's weights back to trainable weights
        self.actor_local1.train()
        return np.vstack(actions)                          

    def learn(self, experiences, gamma):
        """Update policy, Q-value and V - value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s0, a0, r0, s'0, done0,s1, a1, r1, s'1, done1) tuples 
            gamma (float): discount factor
        """
        # Sample instances from replay buffer
        states0, actions0, rewards0, next_states0, dones0, states1, actions1, rewards1, next_states1, dones1 = experiences

        # ---------------------------- update critic ---------------------------- #
        # Q-value from model 1 of agent 0 on observing the entire state and action information
        Q0_1 = self.critic_q_1_local0(states0,states1,actions0,actions1)
        # Q-value from model 1 of agent 1 on observing the entire state and action information
        Q1_1 = self.critic_q_1_local1(states0.clone(),states1.clone(),actions0.clone(),actions1.clone())
        # Q-value from model 2 of agent 0 on observing the entire state and action information
        Q0_2 = self.critic_q_2_local0(states0.clone(),states1.clone(),actions0.clone(),actions1.clone())
        # Q-value from model 2 of agent 1 on observing the entire state and action information
        Q1_2 = self.critic_q_2_local1(states0.clone(),states1.clone(),actions1.clone(),actions1.clone())
        # V-value of agent 0 on observing the entire state information
        V0 = self.critic_v_local0(states0,states1)
        # V-value of agent 1 on observing the entire state information
        V1 = self.critic_v_local1(states0.clone(),states1.clone())
        # Sample action and pdf value from the policy of agent 0
        action_sample_0,log_prob1,_, mean0, log_std0 = self.actor_local0(states0)
        # Sample action and pdf value from the policy of agent 1
        action_sample_1,log_prob2,_, mean1, log_std1 =  self.actor_local1(states1)
        # Choose minimum between model 1 and model 2 Q-value for agent 0
        V0_1_target = torch.min(self.critic_q_1_local0(states0,states1,action_sample_0,action_sample_1.clone().detach()),self.critic_q_2_local0(states0,states1,action_sample_0,action_sample_1.clone().detach()))
        # Choose minimum between model 1 and model 2 Q-value for agent 0
        V1_1_target = torch.min(self.critic_q_1_local1(states0.clone(),states1.clone(),action_sample_0.clone().detach(),action_sample_1),self.critic_q_2_local1(states0.clone(),states1.clone(),action_sample_0.clone().detach(),action_sample_1))

        with torch.no_grad():
            # Compute the V target value for agent 0
            V0_targets = rewards0.view(-1,1) + (gamma * self.critic_v_target0(next_states0,next_states1) * (1 - dones0).view(-1,1))
            # Compute the V target value for agent 1
            V1_targets = rewards1.view(-1,1) + (gamma * self.critic_v_target1(next_states0,next_states1) * (1 - dones1).view(-1,1))
        
        # loss
        critic_0_1_loss = F.mse_loss(Q0_1,V0_targets)
        critic_1_1_loss = F.mse_loss(Q1_1,V1_targets)                                             
        critic_0_2_loss = F.mse_loss(Q0_2,V0_targets)
        critic_1_2_loss = F.mse_loss(Q1_2,V1_targets)
        critic_0_v_loss = F.mse_loss(V0,(V0_1_target - ALPHA*log_prob1).clone().detach())
        critic_1_v_loss = F.mse_loss(V1,(V1_1_target - ALPHA*log_prob2).clone().detach())                                                   
        self.critic_q_1_optimizer0.zero_grad()
        self.critic_q_2_optimizer0.zero_grad()
        self.critic_v_optimizer0.zero_grad()
        self.critic_q_1_optimizer1.zero_grad()
        self.critic_q_2_optimizer1.zero_grad()
        self.critic_v_optimizer1.zero_grad()
        # Compute gradients
        critic_0_1_loss.backward()
        critic_1_1_loss.backward()                                             
        critic_0_2_loss.backward()
        critic_1_2_loss.backward()
        critic_0_v_loss.backward()
        critic_1_v_loss.backward()                                                   
        # Update weights
        self.critic_q_1_optimizer0.step()
        self.critic_q_2_optimizer0.step()
        self.critic_v_optimizer0.step()
        self.critic_q_1_optimizer1.step()
        self.critic_q_2_optimizer1.step()
        self.critic_v_optimizer1.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actor_loss0 = -(V0_1_target - ALPHA*log_prob1).mean()
        actor_loss1 = -(V1_1_target - ALPHA*log_prob2).mean()
        self.actor_optimizer0.zero_grad()
        self.actor_optimizer1.zero_grad()
        
        actor_loss0.backward()
        actor_loss1.backward()
        self.actor_optimizer0.step()
        self.actor_optimizer1.step()
        

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_v_local0, self.critic_v_target0, TAU)
        self.soft_update(self.critic_v_local1, self.critic_v_target1, TAU)                                        # Soft update of the target networks                      

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for i in range(2):
            states.append(torch.tensor(np.stack([e.state[i] for e in experiences if e is not None]),requires_grad = True).float().to(device))
            actions.append(torch.tensor(np.stack([e.action[i] for e in experiences if e is not None]),requires_grad = True).float().to(device))
            rewards.append(torch.tensor(np.stack([e.reward[i] for e in experiences if e is not None]),requires_grad = True).float().to(device))
            next_states.append(torch.tensor(np.stack([e.next_state[i] for e in experiences if e is not None]),requires_grad = True).float().to(device))
            dones.append(torch.tensor(np.stack([e.done[i] for e in experiences if e is not None]).astype(np.uint8),requires_grad = True).float().to(device))

        return (states[0], actions[0], rewards[0], next_states[0], dones[0],states[1], actions[1], rewards[1], next_states[1], dones[1])

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)