import time, datetime
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import tensorflow as tf

from env import GameEnv
from dqn import DeepQNetwork, batch_size, learning_rate, discount_rate

WRITE_VIDEO = True
EPISODES = 500
TRAIN_END = 0

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#Create the env
env = GameEnv()
env.reset()
state = tf.reshape(env.contribute_metrix(), [-1])

#Create agent
nS = state.shape[0] 
nA = env.action_num #Actions
agent = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.1, 0.9995 )
batch_size = batch_size()

save_file = "DQN-%s" % (time.strftime("%Y-%m-%d-%H-%M-%S"))


#Training
rewards = [] #Store rewards for graphing
epsilons = [] # Store the Explore/Exploit
TEST_Episodes = 0
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    if WRITE_VIDEO and e%100 == 0:
        fig = plt.figure()
        frames = []
    for time in range(1000): #200 is when you "solve" the game. This can continue forever as far as I know
        if WRITE_VIDEO and e%100 == 0:
            temp = env.render_env()
            frames.append([
                plt.text(0, -1, "Time: " + str(time), fontsize=8), 
                plt.imshow(temp,animated=True)])
        
        action = agent.action(state)

        reward, _ = env.move(action, 7)
        nstate = tf.reshape(env.contribute_metrix(), [-1])
        done = 0

        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        agent.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
        state = nstate

        if done or time == 999:
            rewards.append(tot_rewards)
            epsilons.append(agent.epsilon)
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e, EPISODES, tot_rewards, agent.epsilon))
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
    if len(rewards) > 5 and np.average(rewards[-5:]) > 400:
        #Set the rest of the EPISODES for testing
        TEST_Episodes = EPISODES - e
        TRAIN_END = e

        agent.save(save_file)
        print(f"  (Saved model to {save_file})")
        break


# Test the agent that was trained
# In this section we ALWAYS use exploit don't train any more
for e_test in range(TEST_Episodes):
    state = env.reset()
    state = np.reshape(state, [1, nS])
    tot_rewards = 0
    for t_test in range(1000):
        action = agent.test_action(state)
        reward, _ = env.move(action, 7)
        nstate = tf.reshape(env.contribute_metrix(), [-1])
        nstate = np.reshape( nstate, [1, nS])
        tot_rewards += reward
        #DON'T STORE ANYTHING DURING TESTING
        state = nstate
        if t_test == 999: 
            rewards.append(tot_rewards)
            epsilons.append(0) #We are doing full exploit
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e_test, TEST_Episodes, tot_rewards, 0))
            break

rolling_average = np.convolve(rewards, np.ones(100)/100)

plt.plot(rewards)
plt.plot(rolling_average, color='black')
#Scale Epsilon (0.001 - 1.0) to match reward (0 - 200) range
eps_graph = [200*x for x in epsilons]
plt.plot(eps_graph, color='g', linestyle='-')
#Plot the line where TESTING begins
plt.axvline(x=TRAIN_END, color='y', linestyle='-')
plt.savefig('training.png')
plt.show()
