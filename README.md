# Multi_Agent_Soft_Actor_Critic
A Pytorch Implementation of Multi Agent Soft Actor Critic

# Project Details

<ul>
  <li> The environment consists of multiple agents where the task of the agent hit the ball and keep it in the air without allowing it to fall on the ground.</li>
  <li> The current state of the environment is represented by 24 dimensional feature vector which conist the position of the ball and speed of the ball</li
  <li> Action space is continous and thus it represent by a vector with 2 numbers, corresponding to position of the bat ranging between -1 and 1 in each dimension.</li>
  <li> A reward of +0.1 is provided for time the agent's hits the ball and -0.1 if the agent miss it or shoots the ball away from the court.</li>
  <li> The task is episoidic, and in order to solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes</li>
</ul>

# Technical Dependencies

<ol>
  <li> Python 3.6 :
  <li> PyTorch (0.4,CUDA 9.0) : pip3 install torch torchvision</li>
  <li> ML-agents (0.4) : Refer to <a href = "https://github.com/Unity-Technologies/ml-agents/">ml-agents</a> for installation</li>
  <li> Numpy (1.14.5) : pip3 install numpy</li>
  <li> Matplotlib (3.0.2) : pip3 install matplotlib</li>
  <li> Jupyter notebook : pip3 install jupyter </li>
  <li> Download the environment from <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip">here</a> and place it in the same folder as that of Tennis.ipynb file  </li>
</ol>

# Network details

- [x] Value - function Network
- [x] Entropy Regularization
- [x] Two Action - Value Networks
- [x] Centralized Training
- [x] Decentralized Execution


# Installation Instructions :
`
step 1 : Install all the dependencies
`
<br>
`
step 2 : git clone https://github.com/adithya-subramanian/Multi_Agent_Soft_Actor_Critic.git
`
<br>
`
step 3 : jupyter notebook
`
<br>
`
step 4 : Run all cells in the Tennis.ipynb file
`
# Acknowledgment

Certain parts of SAC.py,model.py and Tennis.ipynb has been partially taken from the Udacity's deep reinforcement learning Nanodegree.
