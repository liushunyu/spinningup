from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.dqn.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def dqn(env_fn, q_network=core.MLPQNetwork, q_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, replay_size=int(1e6), gamma=0.99,
        q_lr=0.00025, batch_size=32, epsilon_start=1, epsilon_decay=10000, epsilon_end=0.1,
        start_steps=5000, update_after=1000, update_every=4, update_target_every=1000,
        num_test_episodes=10, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """
    Deep Q Network (DQN)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        q_network: The constructor method for a PyTorch Module with an ``act``
            method and a ``q`` module. The ``act`` method and ``q`` should accept
            a batch of observations as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (1,)              | actions for observation.
            ``q``        (batch, act_n)    | Tensor containing the current estimate
                                           | of each action's Q value for the provided
                                           | observations.
            ===========  ================  ======================================

        q_kwargs (dict): Any kwargs appropriate for the QNetwork object
            you provided to DQN.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        epsilon_start (float): Start value of epsilon in epsilon-greedy exploration.

        epsilon_decay (float): The number of frames over which the initial value
        of epsilon is linearly annealed to its final value.

        epsilon_end (float): Final value of epsilon in epsilon-greedy exploration.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        update_target_every (int): Number of gradient descent updates that
            should elapse between target network updates.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module and target networks
    q_network = q_network(env.observation_space, env.action_space, **q_kwargs)
    q_target_network = deepcopy(q_network)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in q_target_network.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = core.count_vars(q_network)
    logger.log('\nNumber of parameters: \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q_value = q_network.q(o)
        q_value = q_value.gather(dim=1, index=a.unsqueeze(1).long()).squeeze(1)

        # Bellman backup for Q function
        with torch.no_grad():
            q_target_value = q_target_network.q(o2).max(1)[0]
            backup = r + gamma * (1 - d) * q_target_value

        # MSE loss against Bellman backup
        loss_q = ((q_value - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q_value.detach().numpy())

        return loss_q, loss_info

    # Set up optimizers for policy and q-function
    q_optimizer = Adam(q_network.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(q_network)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

    def update_target():
        # Update target networks.
        q_target_network.load_state_dict(q_network.state_dict())

    def get_action(o, eps):
        if np.random.random() > eps:
            a = q_network.act(torch.as_tensor(o, dtype=torch.float32))
        else:
            a = env.action_space.sample()
        return a

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = q_network.act(torch.as_tensor(o, dtype=torch.float32))
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    param_update_cnt = 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via epsilon).
        epsilon = epsilon_start - (epsilon_start - epsilon_end) / epsilon_decay * t
        if epsilon < epsilon_end:
            epsilon = epsilon_end

        if t > start_steps:
            a = get_action(o, epsilon)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            batch = replay_buffer.sample_batch(batch_size)
            update(data=batch)
            param_update_cnt += 1

            if param_update_cnt % update_target_every == 0:
                update_target()

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Epsilon', epsilon)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='dqn')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn(lambda : gym.make(args.env), q_network=core.MLPQNetwork,
        q_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
