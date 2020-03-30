import time, datetime
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import tensorflow as tf

from environments.env_team import GameEnv
from agents.dqn import DeepQNetwork
from agents.ddqn import DoubleDeepQNetwork

#Hyper Parameters
def discount_rate(): #Gamma
    return 0.995

def learning_rate(): #Alpha
    return 0.0001

def batch_size(): #Size of the batch used in the experience replay
    return 16

WRITE_VIDEO = False
EPISODES = 2000
TRAIN_END = 0

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
NAME = f"teams-lr{learning_rate()}-dr{discount_rate()}-bs{batch_size()}-{current_time}"

#Create the env
env = GameEnv()
env.reset()
state = tf.reshape(env.contribute_metrix(), [-1])

#Create agent
nS = state.shape[0] 
nA = env.action_num #Actions

# agent1 = DeepQNetwork.restore("saved_models/teams/teams-lr0.0001-dr0.99-bs24-20200312-1120-1")
# agent2 = DeepQNetwork.restore("saved_models/teams/teams-lr0.0001-dr0.99-bs24-20200312-1120-2")
# agent3 = DeepQNetwork.restore("saved_models/teams-lr0.0001-dr0.99-bs24-20200307-1240-3")
# agent4 = DeepQNetwork.restore("saved_models/teams-lr0.0001-dr0.99-bs24-20200307-1240-4")

agent1 = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.99999 )
agent2 = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.99999 )
agent3 = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.9999 )
agent4 = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.9999 )

# agent2 = DoubleDeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.9999 )

batch_size = batch_size()

save_file = "saved_models/teams/" + NAME 
writer = tf.summary.create_file_writer("logs/" + NAME)

def plot(rewards1, rewards2 = None):
    plt.clf()
    plt.plot(rewards1, label='DQN rewards')
    if rewards2:
        plt.plot(rewards2, label='DDQN rewards')
    plt.xlim(0, len(rewards1))
    # rolling_average = np.convolve(rewards, np.ones(100)/100)
    # plt.plot(rolling_average, color='black')
    #Plot the line where TESTING begins
    plt.xlabel('Episodes')
    plt.ylabel('Total rewards')
    plt.legend()
    plt.savefig(save_file+'-plot.png')
    plt.show()


#Training
rewards = [] #Store rewards for graphing
rewards1 = []
rewards2 = []
epsilons = [] # Store the Explore/Exploit
best_rewards = 0
TEST_Episodes = 0
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    tot_rewards1 = 0
    tot_rewards2 = 0
    tot_rewards_blue = 0

    if WRITE_VIDEO and e%100 == 0:
        fig = plt.figure()
        frames = []
    for time in range(1000): 
        if WRITE_VIDEO and e%100 == 0:
            temp = env.render_env()
            frames.append([
                plt.text(0, -1, "Time: " + str(time), fontsize=8), 
                plt.imshow(temp,animated=True)])
        
        action1 = agent1.action(state)
        action2 = agent2.action(state)
        action3 = agent3.action(state)
        action4 = agent4.action(state)

        r1, r2, r3, r4 = env.move(action1, action2, action3, action4)
        nstate = tf.reshape(env.contribute_metrix(), [-1])
        done = 0

        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += r1 + r2 + r3 + r4
        tot_rewards1 += r1 
        tot_rewards2 += r2 
        tot_rewards_blue += r3+r4


        agent1.store(state, action1, r1 + 0.5*r2, nstate, done)
        agent2.store(state, action2, r2 + 0.5*r1, nstate, done)
        agent3.store(state, action3, r3 + 0.5*r4, nstate, done)
        agent4.store(state, action4, r4 + 0.5*r3, nstate, done)

        state = nstate

        if done or time == 999:
            rewards.append(tot_rewards1 + tot_rewards2)
            rewards1.append(tot_rewards1)
            rewards2.append(tot_rewards2)

            epsilons.append(agent1.epsilon)
            print("episode: {}/{}, score 1: {}, score 2: {}, e: {}"
                  .format(e, EPISODES, tot_rewards1, tot_rewards2, agent1.epsilon))
            with writer.as_default():
                tf.summary.scalar('Reward', tot_rewards, e)
                tf.summary.scalar('Reward 1', tot_rewards1, e)
                tf.summary.scalar('Reward 2', tot_rewards2, e)
                tf.summary.scalar('Reward 3 and 4', tot_rewards_blue, e)
                tf.summary.scalar('Epsilon', agent1.epsilon, e)
            break
        #Experience Replay
        if len(agent1.memory) > batch_size:
            agent1.experience_replay(batch_size)
            agent2.experience_replay(batch_size)
            agent3.experience_replay(batch_size)
            agent4.experience_replay(batch_size)


    if WRITE_VIDEO and e%100 == 0:
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani = matplotlib.animation.ArtistAnimation(fig, frames, interval=20, blit=True)
        ani.save('movies/'+current_time+'_episode_'+str(e)+'.mp4',  writer=writer)
        print(f'Video for episode {e} saved.')
    #If our current NN passes we are done
    if len(rewards) > 5 and np.average(rewards[-5:]) > best_rewards:
        best_rewards = np.average(rewards[-5:])
        agent1.save(save_file+"-1")
        agent2.save(save_file+"-2")
        agent3.save(save_file+"-3")
        agent4.save(save_file+"-4")
        print(f"  (Saved model to {save_file})")

    if e%50 == 0:
        plot(rewards1, rewards2)

plot(rewards1, rewards2)
