import numpy as np
import tensorflow as tf

from env_gym import GameEnv
from ppo import PPOAgent

dic_agent_conf = {
    "STATE_DIM": (8, ),
    "ACTOR_LEARNING_RATE": 1e-5,
    "CRITIC_LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 10,
    "GAMMA": 0.99,
    "PATIENCE": 10,
    "NUM_LAYERS": 2,
    "D_DENSE": 32,
    "ACTOR_LOSS": "Clipped",  # or "KL-DIVERGENCE"
    "CLIPPING_LOSS_RATIO": 0.1,
    "ENTROPY_LOSS_RATIO": 0.2,
    "CRITIC_LOSS": "mean_squared_error",
    "OPTIMIZER": "Adam",
    "TARGET_UPDATE_ALPHA": 0.9,
}

dic_env_conf = {
    "ENV_NAME": "LunarLander-v2",
    "GYM_SEED": 1,
    "LIST_STATE_NAME": ["state"],
    "ACTION_RANGE": "-1-1", # or "-1~1"
    "POSITIVE_REWARD": True
}

dic_path ={
    "PPO": "records/PPO/"
}

dic_exp_conf = {
    "TRAIN_ITERATIONS": 1000,
    "MAX_EPISODE_LENGTH": 1000,
    "TEST_ITERATIONS": 10
}


env = GameEnv()
state = tf.reshape(env.reset(), [-1])
print(state.shape)

dic_agent_conf["ACTION_DIM"] = env.action_num
dic_agent_conf["STATE_DIM"] = (state.shape[0] , )

agent = PPOAgent(dic_agent_conf, dic_path, dic_env_conf)

for cnt_episode in range(dic_exp_conf["TRAIN_ITERATIONS"]):
    s = env.reset()
    s = np.reshape(s, [1, -1])
    r_sum = 0
    for cnt_step in range(dic_exp_conf["MAX_EPISODE_LENGTH"]):
        # if cnt_episode > dic_exp_conf["TRAIN_ITERATIONS"] - 10:
        #     env.render()
        if cnt_episode < 0:
            a = np.random.choice(env.action_num)
        else:
            a = agent.action(s)
        _, r, done, _ = env.step(a)

        r /= 500
        r_sum += r
        s_ = np.reshape(env.state(), [-1])

        # if done:
        #     r = -1

        agent.store(s, a, s_, r, done)
        # if cnt_step % dic_agent_conf["BATCH_SIZE"] == 0 and cnt_step != 0:
        #     agent.experience_replay()
        s = s_

        agent.experience_replay()
    print("Episode:{}, step:{}, r_sum:{}".format(cnt_episode, cnt_step, r_sum))