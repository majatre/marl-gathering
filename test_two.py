#!/usr/bin/env python
"""
Usage:
    test_single.py [options] TRAINED_MODEL_1 TRAINED_MODEL_2

Uses trained model to test the agent.

Options:
    -h --help        Show this screen.
    --debug          Enable debug routines. [default: False]
"""

from docopt import docopt
from dpu_utils.utils import run_and_debug

import time, datetime
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
import tensorflow as tf

from env import GameEnv
from dqn import DeepQNetwork, batch_size, learning_rate, discount_rate

WRITE_VIDEO = True
SHOW_GAME = False
TEST_Episodes = 5

def show_game(temp, time, r1, r2):
    plt.imshow(temp)
    plt.text(0, -1, "Time: " + str(time), fontsize=8), 
    plt.text(7, -1, "Total reward - player 1: " + str(r1) + ",  player 2: " + str(r2), fontsize=8)
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()


def run(arguments) -> None:
    #Create the env
    env = GameEnv()
    agent1 = DeepQNetwork.restore(arguments["TRAINED_MODEL_1"])
    agent2 = DeepQNetwork.restore(arguments["TRAINED_MODEL_2"])

    # Test the agent that was trained
    for e_test in range(TEST_Episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent1.nS])
        tot_reward1 = 0
        tot_reward2 = 0


        if WRITE_VIDEO and e_test == 0:
            fig = plt.figure()
            frames = []

        for t_test in range(1000):
            if SHOW_GAME:
                show_game(env.render_env(), t_test, tot_rewards1, tot_rewards2)
            if WRITE_VIDEO and e_test == 0:
                temp = env.render_env()
                frames.append([
                    plt.text(0, -1, "Time: " + str(t_test), fontsize=8), 
                    plt.text(7, -1, "Total reward - player 1: " + str(tot_reward1) + ",  player 2: " + str(tot_reward2), fontsize=8), 
                    plt.imshow(temp,animated=True)])

            agent1_action = agent1.test_action(state)
            agent2_action = agent2.test_action(state)
            reward1, reward2 = env.move(agent1_action, agent2_action)
            nstate = tf.reshape(env.contribute_metrix(), [-1])
            nstate = np.reshape(nstate, [1, agent1.nS])
            tot_reward1 += reward1
            tot_reward2 += reward2

            #DON'T STORE ANYTHING DURING TESTING
            state = nstate
            if t_test == 999: 
                print("episode: {}/{}, scores: {}, {}"
                    .format(e_test, TEST_Episodes, tot_reward1, tot_reward2))
                break

        if WRITE_VIDEO and e_test == 0:
            Writer = matplotlib.animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani = matplotlib.animation.ArtistAnimation(fig, frames, interval=20, blit=True)
            ani.save('movies/'+arguments["TRAINED_MODEL_1"].split('/')[-1]+'_test_2players.mp4',  writer=writer)
            print(f'Video saved.')



if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
