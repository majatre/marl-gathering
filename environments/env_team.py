# Implementation based on https://github.com/wwxFromTju/deepmind_MAS_enviroment

#!/usr/bin/env python3
# encoding=utf-8


import numpy as np
import scipy.misc
from PIL import Image


class AgentObj:
    def __init__(self, coordinates, type, name, direction=0, mark=0, hidden=0):
        self.x = coordinates[0]
        self.y = coordinates[1]
        #0: r, 1: g, 3: b
        self.type = type
        self.name = name
        self.hidden = hidden

        # 0: right, 1:top 2: left. 3: bottom
        self.direction = direction
        self.mark = mark

    def is_hidden(self):
        return self.hidden > 0

    def add_mark(self, agent_hidden):
        self.mark += 1
        if self.mark >= 2:
            self.mark = 0
            self.hidden = agent_hidden
        return self.mark

    def sub_hidden(self):
        self.hidden -= 1
        self.hidden = 0 if self.hidden <=0 else self.hidden
        return self.hidden

    def turn_left(self, **kwargs):
        self.direction = (self.direction + 1) % 4
        return self.direction

    def turn_right(self, **kwargs):
        self.direction = (self.direction - 1 + 4) % 4
        return self.direction

    def move_forward_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 1, 0
        elif self.direction == 1:
            delta_x, delta_y = 0, -1
        elif self.direction == 2:
            delta_x, delta_y = -1, 0
        elif self.direction == 3:
            delta_x, delta_y = 0, 1
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_left_delta(self):
        if self.direction == 0:
            delta_x, delta_y = 0, -1
        elif self.direction == 1:
            delta_x, delta_y = -1, 0
        elif self.direction == 2:
            delta_x, delta_y = 0, 1
        elif self.direction == 3:
            delta_x, delta_y = 1, 0
        else:
            assert self.direction in range(4), 'wrong direction'

        return delta_x, delta_y

    def move_forward(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_forward_delta()

        self.x = self.x + delta_x if self.x + delta_x >=0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >=0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_backward(self, env_x_size, env_y_size):
        forward_delta_x, forward_delta_y = self.move_forward_delta()
        delta_x, delta_y = -forward_delta_x, -forward_delta_y

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_left(self, env_x_size, env_y_size):
        delta_x, delta_y = self.move_left_delta()

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def move_right(self, env_x_size, env_y_size):
        left_delta_x, left_delta_y = self.move_left_delta()
        delta_x, delta_y = -left_delta_x, -left_delta_y

        self.x = self.x + delta_x if self.x + delta_x >= 0 and self.x + delta_x <= env_x_size - 1 else self.x
        self.y = self.y + delta_y if self.y + delta_y >= 0 and self.y + delta_y <= env_y_size - 1 else self.y
        return self.x, self.y

    def stay(self, **kwargs):
        pass

    def beam(self, env_x_size, env_y_size):
        if self.direction == 0:
            beam_set = [(i + 1, self.y) for i in range(self.x, env_x_size - 1)]
        elif self.direction == 1:
            beam_set = [(self.x, i - 1) for i in range(self.y, 0, -1)]
        elif self.direction == 2:
            beam_set = [(i - 1, self.y) for i in range(self.x, 0, -1)]
        elif self.direction == 3:
            beam_set = [(self.x, i + 1) for i in range(self.y, env_y_size - 1)]
        else:
            assert self.direction in range(4), 'wrong direction'
        return beam_set


class FoodObj:
    def __init__(self, coordinates, type=1, hidden=0, reward=1):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.type = type
        self.hidden = hidden
        self.reward = reward

    def is_hidden(self):
        return self.hidden > 0

    def eat(self, food_hidden):
        self.hidden = food_hidden
        return self.reward

    def sub_hidden(self):
        self.hidden -= 1
        self.hidden = 0 if self.hidden <= 0 else self.hidden
        return self.hidden



class GameEnv:
    def __init__(self, widht=31, hight=11, agent_hidden=5, food_hidden=4):
        self.size_x = widht
        self.size_y = hight
        self.objects = []
        self.agent_hidden = agent_hidden
        self.food_hidden = food_hidden

        # 0: forward, 1: backward, 2: left, 3: right
        # 4: trun lelf, 5:turn right, 6: beam, 7: stay
        self.action_num = 8

        self.reset()

    def reset(self):
        self.red1 = AgentObj(coordinates=(0, 4), type=2, name='red1')
        self.red2 = AgentObj(coordinates=(0, 6), type=2, name='red2')

        self.blue1 = AgentObj(coordinates=(30, 4), type=0, name='blue1', direction=2)
        self.blue2 = AgentObj(coordinates=(30, 6), type=0, name='blue2', direction=2)

        self.red1_actions = [self.red1.move_forward, self.red1.move_backward, self.red1.move_left, self.red1.move_right,
                               self.red1.turn_left, self.red1.turn_right, self.red1.beam, self.red1.stay]
        self.blue1_actions = [self.blue1.move_forward, self.blue1.move_backward, self.blue1.move_left, self.blue1.move_right,
                               self.blue1.turn_left, self.blue1.turn_right, self.blue1.beam, self.blue1.stay]
        self.red1_beam_set = []
        self.blue1_beam_set = []

        self.red2_actions = [self.red2.move_forward, self.red2.move_backward, self.red2.move_left, self.red2.move_right,
                               self.red2.turn_left, self.red2.turn_right, self.red2.beam, self.red2.stay]
        self.blue2_actions = [self.blue2.move_forward, self.blue2.move_backward, self.blue2.move_left, self.blue2.move_right,
                               self.blue2.turn_left, self.blue2.turn_right, self.blue2.beam, self.blue2.stay]
        self.red2_beam_set = []
        self.blue2_beam_set = []


        self.food_objects = []

        for x in range(13, 18):
            delta = x - 13 if x -13 < 17 - x else 17 -x
            self.food_objects.append(FoodObj(coordinates=(x, 5)))
            for i in range(delta):
                self.food_objects.append(FoodObj(coordinates=(x, 4 - i)))
                self.food_objects.append(FoodObj(coordinates=(x, 6 + i)))

        return self.contribute_metrix()

    def move(self, red1_action, red2_action, blue1_action, blue2_action):
        assert red1_action in range(8), 'red1 take wrong action'
        assert blue1_action in range(8), 'blue1 take wrong action'

        red1_old_x, red1_old_y = self.red1.x, self.red1.y
        blue1_old_x, blue1_old_y = self.blue1.x, self.blue1.y
        red2_old_x, red2_old_y = self.red2.x, self.red2.y
        blue2_old_x, blue2_old_y = self.blue2.x, self.blue2.y

        
        for agent in [self.red1, self.red2, self.blue1, self.blue2]:
            agent.sub_hidden()

        self.red1_beam_set = []
        self.blue1_beam_set = []
        self.red2_beam_set = []
        self.blue2_beam_set = []
        if not self.red1.is_hidden():
            red1_action_return = self.red1_actions[red1_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.red1_beam_set = [] if red1_action != 6 else red1_action_return
        if not self.blue1.is_hidden():
            blue1_action_return = self.blue1_actions[blue1_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.blue1_beam_set = [] if blue1_action != 6 else blue1_action_return
        if not self.red2.is_hidden():
            red2_action_return = self.red2_actions[red2_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.red2_beam_set = [] if red2_action != 6 else red2_action_return
        if not self.blue2.is_hidden():
            blue2_action_return = self.blue2_actions[blue2_action](env_x_size=self.size_x, env_y_size=self.size_y)
            self.blue2_beam_set = [] if blue2_action != 6 else blue2_action_return

        for agent1 in [self.red1, self.red2, self.blue1, self.blue2]:
            for agent2 in [self.red1, self.red2, self.blue1, self.blue2]:
                if agent1 != agent2:
                    agent1_old_x, agent1_old_y = agent1.x, agent1.y
                    agent2_old_x, agent2_old_y = agent2.x, agent2.y

                    if not agent1.is_hidden() and not agent2.is_hidden() and\
                            ((agent1.x == agent2.x and agent1.y == agent2.y) or
                                (agent1.x == agent2_old_x and agent1.y == agent2_old_y and
                                        agent2.x == agent1_old_x and agent2.y == agent1_old_y)):

                        agent1.x, agent1.y = agent1_old_x, agent1_old_y
                        agent2.x, agent2.y = agent2_old_x, agent2_old_y


        red_reward = 0
        blue_reward = 0
        for food in self.food_objects:
            food.sub_hidden()
            if not food.is_hidden():
                for a in [self.red1, self.red2]:
                    if not a.is_hidden() and food.x == a.x and food.y == a.y:
                        red_reward += food.eat(self.food_hidden)
                for a in [self.blue1, self.blue2]:
                    if not a.is_hidden() and food.x == a.x and food.y == a.y:
                        blue_reward += food.eat(self.food_hidden)

        if (self.red1.x, self.red1.y) in self.blue1_beam_set + self.blue2_beam_set:
            self.red1.add_mark(self.agent_hidden)
        if (self.blue1.x, self.blue1.y) in self.red1_beam_set + self.red2_beam_set:
            self.blue1.add_mark(self.agent_hidden)
        if (self.red2.x, self.red2.y) in self.blue1_beam_set + self.blue2_beam_set:
            self.red2.add_mark(self.agent_hidden)
        if (self.blue2.x, self.blue2.y) in self.red1_beam_set + self.red2_beam_set:
            self.blue2.add_mark(self.agent_hidden)

        return red_reward, blue_reward


    def contribute_metrix(self):
        a = np.ones([self.size_y + 2, self.size_x + 2, 3])
        a[1:-1, 1:-1, :] = 0

        for x, y in self.red1_beam_set + self.red2_beam_set + self.blue1_beam_set + self.blue2_beam_set:
            a[y + 1, x + 1, 0] = 0.5
            a[y + 1, x + 1, 1] = 0.5
            a[y + 1, x + 1, 2] = 0.5

        for food in self.food_objects:
            if not food.is_hidden():
                for i in range(3):
                    a[food.y + 1, food.x + 1, i] = 1 if i == food.type else 0

        for i in range(3):
            for agent in [self.red1, self.red2, self.blue1, self.blue2]:
                if not agent.is_hidden():
                    delta_x, delta_y = agent.move_forward_delta()
                    a[agent.y + 1 + delta_y, agent.x + 1 + delta_x, i] = 0.5
                if not agent.is_hidden():
                    if agent.name[-1] == '1':
                        a[agent.y + 1, agent.x + 1, i] = 1 if i == agent.type else 0
                    if agent.name[-1] == '2':
                        a[agent.y + 1, agent.x + 1, i] = 1 if i == agent.type else 0.5


        return a

    def render_env(self):
        a = self.contribute_metrix()
        a = np.stack([a[:, :, 0], a[:, :, 1], a[:, :, 2]], axis=2)
        return a

    def train_render(self):
        a = self.contribute_metrix()

        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')

        a = np.stack([b, c, d], axis=2)
        return a
