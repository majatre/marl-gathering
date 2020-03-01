#!/usr/bin/env python
"""
Usage:
    test_single.py [options] TRAINED_MODEL ...

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
TEST_Episodes = 5

def run(arguments) -> None:
    #Create the env
    env = GameEnv()
    agent = DeepQNetwork.restore(arguments["TRAINED_MODEL"][0])

    # Test the agent that was trained
    for e_test in range(TEST_Episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.nS])
        tot_rewards = 0

        if WRITE_VIDEO and e_test == 0:
            fig = plt.figure()
            frames = []

        for t_test in range(1000):
            if WRITE_VIDEO and e_test == 0:
                temp = env.render_env()
                frames.append([
                    plt.text(0, -1, "Time: " + str(t_test), fontsize=8), 
                    plt.text(7, -1, "Total reward: " + str(tot_rewards), fontsize=8), 
                    plt.imshow(temp,animated=True)])

            action = agent.test_action(state)
            reward, _ = env.move(action, 7)
            nstate = tf.reshape(env.contribute_metrix(), [-1])
            nstate = np.reshape( nstate, [1, agent.nS])
            tot_rewards += reward
            #DON'T STORE ANYTHING DURING TESTING
            state = nstate
            if t_test == 999: 
                print("episode: {}/{}, score: {}"
                    .format(e_test, TEST_Episodes, tot_rewards))
                break

        if WRITE_VIDEO and e_test == 0:
            Writer = matplotlib.animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani = matplotlib.animation.ArtistAnimation(fig, frames, interval=20, blit=True)
            ani.save('movies/'+arguments["TRAINED_MODEL"][0].split('/')[-1]+'_test.mp4',  writer=writer)
            print(f'Video saved.')



if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
