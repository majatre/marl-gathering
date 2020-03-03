# Implementation based on https://colab.research.google.com/github/ehennis/ReinforcementLearning/blob/master/05-DQN.ipynb

import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
import random
import time, datetime
import pickle

from env import GameEnv

#Hyper Parameters
def discount_rate(): #Gamma
    return 0.99

def learning_rate(): #Alpha
    return 0.0001

def batch_size(): #Size of the batch used in the experience replay
    return 24


class DoubleDeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=25000)
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model_update = 3
        self.step = 0

        self.loss = []
        
    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(256, input_dim=self.nS, activation='relu')) #[Input] -> Layer 1
        model.add(keras.layers.Dense(24, activation='relu')) #Layer 2 -> 3
        model.add(keras.layers.Dense(self.nA, activation='linear')) #Layer 3 -> [output]
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(lr=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
        return model

    def clone_model(self, model, custom_objects={}):
        config = {
            'class_name': model.__class__.__name__,
            'config': model.get_config(),
        }
        clone = model_from_config(config, custom_objects=custom_objects)
        clone.set_weights(model.get_weights())
        return clone

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path: str) -> None:
        # We store things in two steps: One .pkl file for metadata (hypers, vocab, etc.)
        # and then the default TF weight saving.
        data_to_store = {
            "model_class": self.__class__.__name__,
            "num_states": self.nS,
            "num_actions": self.nA,
        }
        with open(path, "wb") as out_file:
            pickle.dump(data_to_store, out_file, pickle.HIGHEST_PROTOCOL)
        self.target_model.save_weights(path, save_format="tf")

    @classmethod
    def restore(cls, saved_model_path: str):
        with open(saved_model_path, "rb") as fh:
            saved_data = pickle.load(fh)

        agent = cls(saved_data["num_states"], saved_data["num_actions"], learning_rate(), discount_rate(), 1, 0.1, 0.9995)
        agent.target_model.build(tf.TensorShape([None, None, None]))
        agent.target_model.load_weights(saved_model_path)

        agent.model.build(tf.TensorShape([None, None, None]))
        agent.model.load_weights(saved_model_path)
        return agent

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA) #Explore
        action_vals = self.target_model.predict(state) #Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state): #Exploit
        action_vals = self.target_model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0,self.nS)) #States
        nst = np.zeros( (0,self.nS) )#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
            nst = np.append( nst, np_array[i,3], axis=0)
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.target_model.predict(nst) #Predict from the TARGET
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)] #Using Q to get T is Double DQN
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i] )
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        self.step += 1

