from env_gym import GameEnv
import matplotlib.pyplot as plt
import matplotlib.animation

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

env = GameEnv() #agent_hidden=20, food_hidden=20)


env = DummyVecEnv([lambda: env])
env.render()

# Custom MLP policy of two layers of size 256 and 24
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 24],
                                           layer_norm=False,
                                           feature_extraction="mlp")


model = DQN(CustomDQNPolicy, env,  learning_rate=0.0001, policy_kwargs=dict(dueling=False), prioritized_replay=True, verbose=1, tensorboard_log="./logs/DQN_policy", double_q=False, target_network_update_freq=3000, train_freq=1, batch_size=24, learning_starts=0, exploration_final_eps=0.1, exploration_fraction=0.05)
model.learn(total_timesteps=500000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()



WRITE_VIDEO = True
TEST_Episodes = 5


# Test the agent that was trained
for e_test in range(TEST_Episodes):
    obs = env.reset()
    tot_rewards = 0

    if WRITE_VIDEO and e_test == 0:
        fig = plt.figure()
        frames = []

    for t_test in range(1000):
        if WRITE_VIDEO and e_test == 0:
            temp = env.render()
            frames.append([
                plt.text(0, -1, "Time: " + str(t_test), fontsize=8), 
                plt.text(7, -1, "Total reward - player 1: " + str(tot_rewards) + ",  player 2: " + str(0), fontsize=8), 
                plt.imshow(temp,animated=True)])

        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        tot_rewards += rewards

        if t_test == 999: 
            print("episode: {}/{}, score: {}"
                .format(e_test, TEST_Episodes, tot_rewards))
            break

    if WRITE_VIDEO and e_test == 0:
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = matplotlib.animation.ArtistAnimation(fig, frames, interval=20, blit=True)
        ani.save('movies/dqn_stable_prior.mp4',  writer=writer)
        print(f'Video saved.')


env.close()