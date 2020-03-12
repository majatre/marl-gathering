import time, datetime
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import tensorflow as tf

from environments.env import GameEnv
from agents.dqn import DeepQNetwork, batch_size, learning_rate, discount_rate

WRITE_VIDEO = False
EPISODES = 500
TRAIN_END = 0

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
NAME = f"DQNvsR-lr{learning_rate()}-dr{discount_rate()}-bs{batch_size()}-{current_time}"

#Create the env
env = GameEnv()
env.reset()
state = tf.reshape(env.contribute_metrix(), [-1])

#Create agent
nS = state.shape[0] 
nA = env.action_num #Actions

agent = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.9999 )
batch_size = batch_size()

save_file = "saved_models/" + NAME 
writer = tf.summary.create_file_writer("logs/" + NAME)

def plot(rewards):
    plt.clf()
    plt.plot(rewards)
    plt.xlim(0, len(rewards))
    plt.savefig(save_file+'-plot.png')
    plt.show()

#Training
rewards = [] #Store rewards for graphing
epsilons = [] # Store the Explore/Exploit
best_rewards = 0
TEST_Episodes = 0
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    if WRITE_VIDEO and e%100 == 0:
        fig = plt.figure()
        frames = []
    for time in range(1000): 
        if WRITE_VIDEO and e%100 == 0:
            temp = env.render_env()
            frames.append([
                plt.text(0, -1, "Time: " + str(time), fontsize=8), 
                plt.imshow(temp,animated=True)])
        
        action = agent.action(state)

        reward, _ = env.move(action, np.random.randint(8))
        nstate = tf.reshape(env.contribute_metrix(), [-1])
        done = 0

        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        agent.store(state, action, reward, nstate, done) 
        state = nstate

        if done or time == 999:
            rewards.append(tot_rewards)
            epsilons.append(agent.epsilon)
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e, EPISODES, tot_rewards, agent.epsilon))
            with writer.as_default():
                tf.summary.scalar('Reward', tot_rewards, e)
                tf.summary.scalar('Epsilon', agent.epsilon, e)
            break
        #Experience Replay
        if len(agent.memory) > batch_size:
            agent.experience_replay(batch_size)

    if WRITE_VIDEO and e%100 == 0:
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = matplotlib.animation.ArtistAnimation(fig, frames, interval=20, blit=True)
        ani.save('movies/'+current_time+'_episode_'+str(e)+'.mp4',  writer=writer)
        print(f'Video for episode {e} saved.')
    #If our current NN passes we are done
    if len(rewards) > 5 and np.average(rewards[-5:]) > best_rewards:
        best_rewards = np.average(rewards[-5:])
        agent.save(save_file)
        print(f"  (Saved model to {save_file})")

    if e%50 == 0:
        plot(rewards)

plot(rewards)
