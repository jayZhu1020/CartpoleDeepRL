'''

Jiang Zhu

Discription: A DQN Agent for CartPole openai gym environment

Date: April 3 2021

'''

# Importing Libearies
import shutil
import os
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import gym
from collections import deque
import random
import time

# Initializing environment and set hyperparameters
MODEL_PATH = 'DQN_Model'
env = gym.make('CartPole-v1')
STATE_SHAPE = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n
SAVE_MODEL_EVERY = 100
EPSILON_DECAY = 0.95
MIN_REPLAY_MEMORY_SIZE = 1000
MAX_REPLAY_MEMORY_SIZE = 5000
MIN_EPSILON = 0.01
PREDICTION_MODEL_UPDATE = 10
MINI_BATCH_SIZE = 32
NUM_EPISODES = 501
RENDER_EVERY_EPISODE = 250
BEGIN_EPSILON = 1.0
UPDATE_PREDICTION_EVERY = 10
GAMMA = 0.95

# The DQN agent class
class DQNAgent:
    def __init__(self):
        self.EPSILON = BEGIN_EPSILON

        # final model we want to output
        self.target_model = self.create_model()

        # the regularly updated model to create stationary objective function
        self.prediction_model = self.create_model()
        self.update_prediction_model()

        # the replay memory that only keeps track of certain amount of previous memory
        # this deck will consist of tuples (state, action, reward, new_state, done)
        self.replay_memory = deque(maxlen = MAX_REPLAY_MEMORY_SIZE)

        # counter on perdiction model update
        self.pred_model_update_counter = 0



    def create_model(self):
        # Create a Deep NN model
        nn = Sequential()
        nn.add(Dense(4, input_dim = STATE_SHAPE, activation = 'relu'))
        nn.add(Dense(4,activation = 'relu'))
        nn.add(Dense(NUM_ACTIONS))
        nn.compile(optimizer='Adam', loss = 'mse', metrics=['accuracy'])
        return nn

    def replay(self):
        # train the model only when we have enough replay memory
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Randomly sample from the replay memory
        minibatch = random.sample(self.replay_memory, MINI_BATCH_SIZE)

        X = np.stack([memory_instance[0] for memory_instance in minibatch],axis = 1)[0]
        X_new_states = np.stack([memory_instance[3] for memory_instance in minibatch],axis = 1)[0]
        y = self.prediction_model.predict(X)
        y_new_states = self.prediction_model.predict(X_new_states)
        for i, (state, action, reward, new_state, done) in enumerate(minibatch):
          if done:
            y[i, action] = reward
          else:
            y[i, action] = reward + GAMMA * np.amax(y_new_states[i])

        self.target_model.fit(X, y, batch_size=MINI_BATCH_SIZE,shuffle=False,epochs=1, verbose=0)

        self.pred_model_update_counter += 1
        if (self.pred_model_update_counter >= UPDATE_PREDICTION_EVERY):
            self.update_prediction_model()
            self.pred_model_update_counter = 0

    def update_prediction_model(self):
        self.prediction_model.set_weights(self.target_model.get_weights())

    def update_epsilon(self):
        self.EPSILON = max(self.EPSILON* EPSILON_DECAY,MIN_EPSILON)

    def run(self):
        if os.path.exists(MODEL_PATH):
            print("Solution already exists!")
            str = input("Enter R to render the solution. Otherwise improve the model.")
            if str == "R":
                print("Rendering the current solution.")
                self.load_model()
                self.render()
            else:
                print("Improving the model.")
                self.load_model()
                self.EPSILON = 0.5
                self.train()
        else:
            print("No solution exists, start training a model.")
            self.train()

    def load_model(self):
        self.target_model = load_model(MODEL_PATH)
        self.prediction_model = load_model(MODEL_PATH)

    def train(self):
        start_tic = time.time()
        for epi in range(NUM_EPISODES):

            state = env.reset()
            state = np.reshape(state, (1,STATE_SHAPE))
            done = False

            reward_total = 0
            steps = 0
            tic = time.time()
            while not done:

                # select an action with epsilon-greedy
                if np.random.binomial(1, self.EPSILON):
                    action = np.random.randint(NUM_ACTIONS)
                else:
                    action = np.argmax(self.prediction_model.predict(state)[0])
                self.update_epsilon()

                # step wth the selected action
                new_state, reward, done, _= env.step(action)
                new_state = np.reshape(new_state, (1,STATE_SHAPE))
                steps += 1
                reward_total += reward

                # append this instance to the memory and train the model
                self.replay_memory.append((state, action, reward, new_state, done))
                self.replay()

                # set the state to new state

                state = new_state
            # Computing time

            tac = time.time()
            time_elapse, time_remain = compute_time(tic,tac,epi)
            print('episode {}: \t reward: {} \t time: {} \t esimated time remaining: {}'.format(epi, reward_total, time_elapse, time_remain))
            if epi % SAVE_MODEL_EVERY == 0:
                self.save_model()
        end_tac = time.time()
        self.save_model()
        time_elapse_total, _useless = compute_time(start_tic, end_tac, NUM_EPISODES)
        print('total time used: {}'.format(time_elapse_total))

    def render(self):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(self.target_model.predict(state.reshape(1,STATE_SHAPE)))
            new_state, reward, done, _ = env.step(action)
            env.render()
            state = new_state

    def save_model(self):
        if os.path.exists(MODEL_PATH):
          shutil.rmtree(MODEL_PATH)
        self.target_model.save(MODEL_PATH)

def compute_time(tic, tac, epi):
    total_time = tac - tic
    time_elapse = '{:.2f} seconds'.format(tac - tic)
    time_remaining = int((NUM_EPISODES - epi) * total_time)
    hours = time_remaining // 3600
    time_remaining %= 3600
    minutes = time_remaining // 60
    seconds = time_remaining % 60
    time_remain = '{:02.0f}:{:02.0f}:{:02.0f}'.format(hours, minutes, seconds)
    return time_elapse, time_remain


if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    env.close()