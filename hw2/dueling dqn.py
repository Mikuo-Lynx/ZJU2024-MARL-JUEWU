import copy
import random
import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replaybuffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        '''
        定义一个三层神经网络，其中
        输入层：全连接层： 状态空间作为输入，256作为输出，Relu激活函数
        中间层：全连接层：256输入256输出，Relu激活
        输出层：全连接层：256输入，动作空间输出
        '''
        self.layer1 = nn.Linear(env.observation_space.shape[0], 256)
        self.layer2 = nn.Linear(256, 256)
        # Dueling DQN:分别定义value层和advantage层
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, env.action_space.n)

    def forward(self, x):
        '''
        前向计算
        '''
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        value=self.V(x)
        advantage=self.A(x)
        # 用均值代替最大化操作
        x=value+advantage-advantage.mean(dim=1,keepdim=True)
        return x


if __name__ == "__main__":
    # Parameters
    learning_rate = 3e-4
    buffer_size = int(1e5)
    total_timesteps = int(1e6)
    epsilon = 0.01
    gamma = 0.99
    tau = 1.0

    learning_starts = 10000
    train_frequency = 4
    log_frequency = 500
    target_frequency = 1000
    batch_size = 256

    # wandb.init(project="DQN", name="Freeway")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/Freeway-v5", obs_type="ram")
    eval_env = gym.make("ALE/Freeway-v5", obs_type="ram")
   
    q_network = QNetwork(env).to(device)
    target_network = copy.deepcopy(q_network)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    buffer = ReplayBuffer(env.observation_space.shape[0], env.action_space.n, buffer_size, device)

    obs, _ = env.reset()
    total_rewards = []
    total_reward = 0
    for step in range(total_timesteps):

        # Epsilon greedy
        if random.random() < epsilon:
            actions = env.action_space.sample()
            
        else:

            q_values = q_network(torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(dim=0))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
         
        if type(actions) == np.ndarray:
            actions = actions.item()

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += rewards

        real_next_obs = infos["final_observation"] if truncations else next_obs.copy()
        buffer.add(obs, actions, real_next_obs, rewards, terminations)
        obs = next_obs

        if terminations:
            obs, _ = env.reset()
            total_rewards.append(total_reward)
            total_reward = 0

        # Training.
        if step > learning_starts:
            if step % train_frequency == 0:
                data = buffer.sample(batch_size)
                buffer_obs, act, next_buffer_obs, rew, cont = data
                with torch.no_grad():
                    target_max = target_network(buffer_obs).max(dim=1, keepdim=True)[0]
                    td_target = rew + gamma * target_max * cont
                old_val = q_network(buffer_obs).gather(1, act)
                loss = F.mse_loss(td_target, old_val)

                if step % log_frequency == 0:
                    # wandb.log({"td_loss": loss.item(), "q_values": old_val.mean().item()}, step=step)
                    print('td_loss: {}\t q_values: {}\t step: {}, avg_rewards: {}'.format(loss.item(), old_val.mean().item(), step, np.mean(total_rewards[-100:])))
                    pass
                
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if step % target_frequency == 0:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    env.close()
