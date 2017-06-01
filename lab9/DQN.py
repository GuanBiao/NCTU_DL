import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple

# Hyper Parameters:
GAMMA = 0.99                        # decay rate of past observations

# Epsilon
INITIAL_EPSILON = 1.0               # 0.01 # starting value of epsilon
FINAL_EPSILON = 0.1                 # 0.001 # final value of epsilon
EXPLORE_STEPS = 500000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 200000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

MONITOR_PATH = 'breakout_videos/'
RECORD_VIDEO_EVERY = 1000

SAVE_MODEL_EVERY = 1000

# Valid actions for breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
VALID_ACTIONS = [0, 1, 2, 3]

class ObservationProcessor():
    """
    Processes a raw Atari image. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)              # input image
            self.output = tf.image.rgb_to_grayscale(self.input_state)                           # rgb to grayscale
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)           # crop image
            self.output = tf.image.resize_images(                                               # resize image
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)                                               # remove rgb dimension

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84, 1] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })

class DQN():
    # Define the following things about Deep Q Network here:
    #   1. Network Structure (Check lab spec for details)
    #       * tf.contrib.layers.conv2d()
    #       * tf.contrib.layers.flatten()
    #       * tf.contrib.layers.fully_connected()
    #       * You may need to use tf.variable_scope in order to set different variable names for 2 Q-networks
    #   2. Target value & loss
    #   3. Network optimizer: tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    #   4. Training operation for tensorflow

    ''' You may need 3 placeholders for input: 4 input images, target Q value, action index
    def _build_network(self):
        # Placeholders for our input
        # Our input are 4 grayscale frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
    '''
    def __init__(self, name):
        with tf.variable_scope(name):
            self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
            self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
            self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            self.X = tf.cast(self.X_pl, tf.float32) / 255.0

            self.conv1 = tf.contrib.layers.conv2d(inputs=self.X, num_outputs=32, kernel_size=8, stride=4, padding="SAME", activation_fn=tf.nn.relu)
            self.conv2 = tf.contrib.layers.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=4, stride=2, padding="SAME", activation_fn=tf.nn.relu)
            self.conv3 = tf.contrib.layers.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=3, stride=1, padding="SAME", activation_fn=tf.nn.relu)
            self.conv3 = tf.contrib.layers.flatten(inputs=self.conv3)
            self.fc1 = tf.contrib.layers.fully_connected(inputs=self.conv3, num_outputs=512, activation_fn=tf.nn.relu)
            self.out = tf.contrib.layers.fully_connected(inputs=self.fc1, num_outputs=len(VALID_ACTIONS), activation_fn=None)

            self.max_Q = tf.reduce_max(self.out, axis=1)
            self.argmax_Q = tf.argmax(self.out, axis=1)
            self.Q = tf.reduce_sum(tf.multiply(self.out, tf.one_hot(self.actions_pl, len(VALID_ACTIONS))), axis=1)

            self.loss = tf.losses.mean_squared_error(labels=self.y_pl, predictions=self.Q, weights=1.0)
            self.opt = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss)

    def get_max_Q(self, sess, X):
        return sess.run(self.max_Q, { self.X_pl: X })

    def get_argmax_Q(self, sess, X):
        return sess.run(self.argmax_Q, { self.X_pl: X })

    def update(self, sess, X, y, actions):
        sess.run(self.opt, { self.X_pl: X, self.y_pl: y, self.actions_pl: actions })

def update_target_network(sess, behavior_Q, target_Q):
    # copy weights from behavior Q-network to target Q-network
    # Hint:
    #   * tf.trainable_variables()                  https://www.tensorflow.org/api_docs/python/tf/trainable_variables
    #   * variable.name.startswith(scope_name)      https://docs.python.org/3/library/stdtypes.html#str.startswith
    #   * assign                                    https://www.tensorflow.org/api_docs/python/tf/assign
    behavior_vars = [var for var in tf.trainable_variables() if var.name.startswith("behavior")]
    target_vars = [var for var in tf.trainable_variables() if var.name.startswith("target")]

    update_ops = []
    for t, b in zip(target_vars, behavior_vars):
        op = tf.assign(t, b)
        update_ops.append(op)

    sess.run(update_ops)


def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")

    # Define Transition tuple
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # create a observation processor
    ob_proc = ObservationProcessor()

    # Behavior Network & Target Network
    behavior_Q = DQN("behavior")
    target_Q = DQN("target")

    # tensorflow session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = env.reset()                               # retrive first env image
    observation = ob_proc.process(sess, observation)        # process the image
    state = np.stack([observation] * 4, axis=2)             # stack the image 4 times
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        '''
        *** This part is just pseudo code ***

        action = None
        if random.random() <= epsilon
            action = random_action
        else
            action = DQN_action
        '''
        action = random.randint(0, len(VALID_ACTIONS) - 1)

        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_observation = ob_proc.process(sess, next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        # Current game episode is over
        if done:
            observation = env.reset()
            observation = ob_proc.process(sess, observation)
            state = np.stack([observation] * 4, axis=2)

        # Not over yet
        else:
            state = next_state


    # record videos
    env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda count: count % RECORD_VIDEO_EVERY == 0, resume=True)

    # total steps
    total_t = 0

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)

    for episode in range(TRAINING_EPISODES):

        # Reset the environment
        observation = env.reset()
        observation = ob_proc.process(sess, observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward
        '''
        How to update episode reward:
        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        episode_reward += reward
        '''

        for t in itertools.count():

            # choose a action
            epsilon = None
            if total_t < EXPLORE_STEPS:
                epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * total_t / (EXPLORE_STEPS - 1)
            else:
                epsilon = FINAL_EPSILON
            action = None
            if random.random() <= epsilon:
                action = random.randint(0, len(VALID_ACTIONS) - 1)
            else:
                action = behavior_Q.get_argmax_Q(sess, np.expand_dims(state, axis=0))[0]

            epsilon_summary = tf.Summary()
            epsilon_summary.value.add(simple_value=epsilon, tag="epsilon")
            summary_writer.add_summary(epsilon_summary, total_t)

            # execute the action
            next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
            episode_reward += reward

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) == REPLAY_MEMORY_SIZE:
                replay_memory.pop(0)

            # save the transition to replay buffer
            next_observation = ob_proc.process(sess, next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            replay_memory.append(Transition(state, action, reward, next_state, done))

            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            samples = random.sample(replay_memory, BATCH_SIZE)
            s, a, r, next_s, d = map(np.array, zip(*samples))

            # calculate target Q values by target network
            max_Qs = target_Q.get_max_Q(sess, next_s)
            y = r + (1 - d) * GAMMA * max_Qs

            # Update network
            behavior_Q.update(sess, s, y, a)

            # Update target network every FREQ_UPDATE_TARGET_Q steps
            if total_t % FREQ_UPDATE_TARGET_Q == 0:
                update_target_network(sess, behavior_Q, target_Q)

            if done:
                print (episode, "Episode reward: ", episode_reward)
                break

            state = next_state
            total_t += 1

        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=episode_reward, tag="reward")
        summary_writer.add_summary(episode_summary, episode)

        if (episode + 1) % SAVE_MODEL_EVERY == 0:
            saver.save(sess, "checkpoint_files/model", episode)


if __name__ == '__main__':
    tf.app.run()
