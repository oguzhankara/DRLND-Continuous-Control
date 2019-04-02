# Training Robotic Arms by Continuous Control using Deep Reinforcement Learning
##### &nbsp;
![Trained Agent](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent")

## Introduction
This repository contains a Deep Deterministic Policy Gradients (DDPG) agent running in the [Unity ML Agent Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. The goal is to get 20 different robotic arms to maintain contact with the green spheres as long as possible. Therefore, the agent is trained in the Second Version of the environment. In the second version of the project environment, there are 20 identical copies of the agent. It has been shown that having multiple copies of the same agent sharing experience can accelerate learning.



The DDPG is implemented in Python 3 using PyTorch.

##### &nbsp;

## Environment

- _**Set-up**_: Double-jointed arm which can move to target locations.
- _**Goal**_: Each agent must move its hand to the goal location, and keep it there.
- _**Agents**_: The environment contains 20 agents linked to a single Brain.
- _**Agent Reward Function (independent)**_:
  - +0.1 for each timestep agent's hand is in goal location.
- _**Brains**_: One Brain with the following observation/action space.
  - Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Vector Action space: (Continuous) Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
  - Visual Observations: None.
- _**Reset Parameters**_: Two, corresponding to goal size, and goal movement speed.
- _**Environment Solving Criteria**_: The target for the agent is to solve the environment by achieving a score of +30 averaged across all 20 agents for 100 consecutive episodes.

##### &nbsp;

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)


##### &nbsp;

## Repository Structure
The code is structured as follows:
* **Continuous_Control.ipynb**: This is where the _DDPG agent_ is trained.
* **ddpg_agent.py**: This module implements a class to represent a __DDPG agent_.
* **model.py**: This module contains the implementation of the _Actor and Critic_ neural networks.
* **checkpoint_actor.pth**: This is the binary containing the trained neural network weights for Actor.
* **checkpoint_critic.pth**: This is the binary containing the trained neural network weights for Critic.
* **Report.md**: Project report and result analysis.

##### &nbsp;

## Dependencies
* python 3.6
* numpy: Install with 'pip install numpy'.
* PyTorch: Install by following the instructions [here](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows).
* ml-agents: Install by following instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md).

##### &nbsp;

## Instructions

Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  
Trained model weights is included for quickly running the agent and seeing the result in Unity ML Agent.
- Run the last cell of the notebook `Continuous_Control.ipynb`.
