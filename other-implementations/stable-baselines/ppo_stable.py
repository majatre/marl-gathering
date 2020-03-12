from env_gym import GameEnv
import matplotlib.pyplot as plt
import matplotlib.animation

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = GameEnv()
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
env = DummyVecEnv([lambda: env])

policy_kwargs = dict(net_arch=[256, 32])

model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=2, tensorboard_log="./logs/PPO_network")
model.learn(total_timesteps=1000000)


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
        ani.save('movies/ppo_stable_test.mp4',  writer=writer)
        print(f'Video saved.')


env.close()