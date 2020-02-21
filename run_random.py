#!/usr/bin/env python3
# encoding=utf-8

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

from env import GameEnv

env = GameEnv()
env.reset()

temp = env.render_env()
i = 0
fig = plt.figure()
frames = []
while i<100:
    temp = env.render_env()
    frames.append([plt.text(0, -1, "Time: " + str(i), fontsize=8), plt.imshow(temp,animated=True)])
    # plt.imshow(temp)
    # plt.text(3, -1, f'reward 1: {r1}, reward 2: {r2}', fontsize=8)
    # plt.show(block=False)
    # plt.pause(0.01)
    action1 = np.random.randint(8)
    action2 = np.random.randint(8)

    r1, r2 = env.move(action1, action2)
    i += 1
    if r1 or r2:
        print(i, 'r1: ', r1, 'r2', r2)


Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani = matplotlib.animation.ArtistAnimation(fig, frames, interval=20, blit=True)
ani.save('movies/random.mp4',  writer=writer)