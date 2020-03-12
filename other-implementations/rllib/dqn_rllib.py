import gym
from gym.spaces import Discrete, Box
from ray import tune
from env_gym import GameEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.misc import normc_initializer

tf = try_import_tf()

class CustomModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        print(model_config)
        # {'conv_filters': None, 'conv_activation': 'relu', 'fcnet_activation': 'tanh', 'fcnet_hiddens': [256, 256], 'free_log_std': False, 'no_final_linear': False, 'vf_share_layers': False, 'use_lstm': False, 'max_seq_len': 20, 'lstm_cell_size': 256, 'lstm_use_prev_action_reward': False, 'state_shape': None, 'framestack': False, 'dim': 84, 'grayscale': False, 'zero_mean': True, 'custom_model': 'my_model', 'custom_action_dist': None, 'custom_options': {}, 'custom_preprocessor': None}
        print(obs_space.shape)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            256,
            name="my_layer1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_2 = tf.keras.layers.Dense(
            24,
            name="my_layer2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(layer_1)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)

        value_layer_1 = tf.keras.layers.Dense(
            256,
            name="value_layer1",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        value_layer_2 = tf.keras.layers.Dense(
            24,
            name="value_layer2",
            activation=tf.nn.tanh,
            kernel_initializer=normc_initializer(1.0))(value_layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(value_layer_2)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

env = GameEnv()
ModelCatalog.register_custom_model("my_model", CustomModel)

tune.run(
    "DQN",
    config={
        "env": GameEnv,
        # "use_gae": False,
        "num_workers": 1,
        "env_config": {},
        "train_batch_size": 1000,
        "target_network_update_freq": 3000,
        "double_q": False,
        "model": {
                "custom_model": "my_model",
                "framestack": False
            },
        })
