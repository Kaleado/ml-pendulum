import gym
import tensorflow as tf
import numpy as np
import math
import random
from collections import deque

# Learning environment parameters
ENV_NAME = 'Pendulum-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# Hyperparameters
DEBUG_CSV = False
GAMMA = 0.9# discount factor
INITIAL_EPSILON = 1.0 # Starting value of epsilon - chosen so that we
                      # always choose randomly to start.
FINAL_EPSILON = 0.0 # final value of epsilon
EPSILON_DECAY_STEPS = 200
LEARNING_RATE = 0.002
HIDDEN_NODES = 128
BATCH_SIZE = 500
REPLAY_MEMORY_SIZE = 20000
TARGET_REWARD = 200
REWARD_HISTORY_SIZE = 10

# Set up OpenAI gym environment
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
granularity = 0.01
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = int(2*env.action_space.high[0] / granularity + 1);

# Our action space is a continuous line from -env.max_torque to
# +env.max_torque. I will pretend we can only work on a certain granularity;
# that is, we can only choose to exert forces in multiples of our granularity.
# I realise this is not a 'proper' solution, but I'm hoping to get something
# that works.

# Placeholders
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

first_layer = tf.layers.dense(state_in, HIDDEN_NODES, name="first_layer",
                              activation=tf.nn.tanh, trainable=True,
                              kernel_initializer=tf.truncated_normal_initializer)
second_layer = tf.layers.dense(first_layer, HIDDEN_NODES, name="second_layer",
                               activation=tf.nn.tanh, trainable=True,
                               kernel_initializer=tf.truncated_normal_initializer)
out = tf.layers.dense(second_layer, ACTION_DIM, name="out",
                      kernel_initializer=tf.truncated_normal_initializer)
q_values = tf.reshape(out, [ACTION_DIM], name="q_values")


q_action = tf.convert_to_tensor([tf.reduce_max(tf.multiply(action_in, q_values))])

# TODO: Loss/Optimizer Definition

# This is the equation given on slide 11 of the 'deep reinforcement learning' lecture slides.
loss = tf.losses.mean_squared_error(target_in, q_action)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

def newEpsilon(target, actual):
    global epsilon
    f = max(epsilon * 0, 1 - (actual/(target * 1.0)))
    epsilon = max(FINAL_EPSILON, min(epsilon, f))

# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action


# Main learning loop
replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)
past = []

episode = 0
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    step = 0
    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step([(np.argmax(action) - (ACTION_DIM - 1)/2)*granularity])

        # Add replay
        if(len(replay_buffer) >= REPLAY_MEMORY_SIZE):
            pass # do nothing when buffer is full
        else:
            replay_buffer.append((state, action, reward, next_state, done))
        if done:
            break
        # Update state
        state = next_state

    # Randomly sample replays
    if len(replay_buffer) >= BATCH_SIZE:
        sample_replays = random.sample(replay_buffer, BATCH_SIZE)
        for sample_replay in sample_replays:
            (train_state, train_action, train_reward, next_train_state, train_done) = sample_replay
            nexttrain_state_q_values = q_values.eval(feed_dict={
                state_in: [next_train_state]
            })
            if train_done:
                train_target = train_reward
            else:
                train_target = train_reward + GAMMA * (np.max(nexttrain_state_q_values))
            # Do one training step
            session.run([optimizer], feed_dict={
                target_in: [train_target],
                action_in: [train_action],
                state_in: [train_state]
            })
    past.append(step + 1)
    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS
    #newEpsilon(TARGET_REWARD, step + 1)
    epsilon = max(epsilon, FINAL_EPSILON)
    #avg_rew = sum([past[x] for x in range(episode-REWARD_HISTORY_SIZE, episode)])/(REWARD_HISTORY_SIZE) if (episode > REWARD_HISTORY_SIZE) else 0
    #if avg_rew >= 190:
    #    BATCH_SIZE = 0
    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                # env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step([(action - (ACTION_DIM - 1)/2)*granularity])
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
              'Average Reward:', ave_reward)

env.close()
