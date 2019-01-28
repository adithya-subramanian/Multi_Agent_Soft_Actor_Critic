import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self,state_size,action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_size,1024)
        self.linear2 = nn.Linear(1024,512)
        self.skip13 = nn.Linear(1024,256)
        self.linear3 = nn.Linear(512,256)
        self.skip35 = nn.Linear(256,64)
        self.linear4 = nn.Linear(256,128)
        self.linear5 = nn.Linear(128,64)
        self.linear6 = nn.Linear(64,32)
        self.mean_linear = nn.Linear(32,action_size)
        self.log_std_linear = nn.Linear(32,action_size)


    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x)) + F.relu(self.skip13(x1)) 
        x = F.relu(self.linear4(x1))
        x = F.relu(self.linear5(x)) + F.relu(self.skip35(x1))   
        x = F.relu(self.linear6(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

class Critic_Q(nn.Module):

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic_Q, self).__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(2*state_size + 2*action_size,1024)
        self.fc2 = nn.Linear(2*action_size + 1024,512)
        self.fc3 = nn.Linear(512,256)
        self.skip13 = nn.Linear(1024,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,64)
        self.skip35 = nn.Linear(256,64)
        self.fc6 = nn.Linear(64,32)
        self.fc7 = nn.Linear(32,1)
        self.act1 = nn.Tanh()
        self.act2 = nn.ReLU()

    def forward(self,x,y,z,w):
        x1 = self.act2(self.fc1(torch.cat([x,y,z,w],dim = 1)))
        x = self.act2(self.fc2(torch.cat([x1,z,w],dim = 1)))
        x1 = self.act2(self.fc3(x)) + self.act2(self.skip13(x1))
        x = self.act2(self.fc4(x1))
        x = self.act2(self.fc5(x)) + self.act2(self.skip35(x1)) 
        x = self.act2(self.fc6(x))
        x = self.act1(self.fc7(x))
        return x

class Critic_V(nn.Module):

    def __init__(self, state_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
        """
        super(Critic_V, self).__init__()
        self.state_size = state_size
        self.fc1 = nn.Linear(state_size*2,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.skip13 = nn.Linear(1024,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,64)
        self.skip35 = nn.Linear(256,64)
        self.fc6 = nn.Linear(64,32)
        self.fc7 = nn.Linear(32,1)
        self.act1 = nn.Tanh()
        self.act2 = nn.ReLU()

    def forward(self,x,y):
        x1 = self.act2(self.fc1(torch.cat([x,y],dim = 1)))
        x = self.act2(self.fc2(x1))
        x1 = self.act2(self.fc3(x)) + self.act2(self.skip13(x1))
        x = self.act2(self.fc4(x1))
        x = self.act2(self.fc5(x)) + self.act2(self.skip35(x1)) 
        x = self.act2(self.fc6(x))
        x = self.act1(self.fc7(x))
        return x