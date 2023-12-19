import tensorflow as tf
import numpy as np
from tensorflow import keras
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    # returns [states, actions, rewards, next_states, done_flags]
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch


class DQAgent:
    '''
    Initializes the Deep Q Network agent
    '''

    def __init__(self, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=500,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), replay_start_size=None):

        assert len(activations) == len(n_neurons) + 1
        self.state_size = 4
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self.build_model()
        self.replay_buffer = ReplayBuffer(mem_size)

    '''
    Builds the model with two hidden layers and one activiation layer
    '''

    def build_model(self):
        input_shape = 4
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                self.n_neurons[0], input_dim=input_shape, activation='relu'),
            tf.keras.layers.Dense(self.n_neurons[1], activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    '''
    Append memory
    '''

    def append_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    '''Policy
    '''

    def policy(self, states):
        max_value = None
        best_state = None

        # epsilon-greedy policy
        if random.random() <= self.epsilon:
            return random.choice(states)
        else:
            for state in states:
                value = self.model(np.reshape(
                    state[2], [1, self.state_size]))[0]
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    '''
    Train
    '''

    def train(self, batch_size=32, epochs=5):
        if len(self.replay_buffer.buffer) <= batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        next_states = np.array([x[3] for x in batch])
        next_qs = [x[0] for x in self.model.predict(next_states)]

        x = []
        y = []

        # Build xy structure to fit the model in batch (better performance)
        for i, (state, _, reward, nextstate, done) in enumerate(batch):

            if state:

                # computation of reward in terms of nextstate values
                reward = reward + \
                    (-.51*nextstate[1] + .76*nextstate[2] - .36 *
                     nextstate[3] - .18*nextstate[0])*.5
                if not done:
                    # Bellman Q formula
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward*.6  # penalize if game over

                x.append(state)
                y.append(new_q)

        # fit the model to the given values
        self.model.fit(np.array(x), np.array(y), epochs=epochs,
                       batch_size=batch_size, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
