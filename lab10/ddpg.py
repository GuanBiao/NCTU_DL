import gym
from gym.wrappers import Monitor
import numpy as np
import sys
import os
import itertools
import tensorflow as tf

from replay_buffer import ReplayBuffer

MAX_EPISODES = 3500                         # Training Episodes
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001                                 # parameter for soft update

# Directory for storing gym results
RECORD_VIDEOS = True
MONITOR_DIR = './gym_ddpg'
RANDOM_SEED = 1234                          # random seed for game enviornment

BUFFER_SIZE = 10000                         # size of replay buffer
MINIBATCH_SIZE = 64

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, scope):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.scope = scope

        # behavior network
        self.input, self.out, self.scaled_out = self.create_actor_network(scope=self.scope)

        # This gradient will be provided by critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # network parameters
        self.network_params = sorted([t for t in tf.trainable_variables() if t.name.startswith(self.get_scope())], key=lambda v: v.name)

        # combine the gradients here
        # Hint: tf.gradients()      https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/gradients
        self.actor_gradients = tf.gradients(self.scaled_out, self.get_params(), -self.action_gradient)

        # apply gradient to actor network
        # Hint:
        #   apply_gradients     https://www.tensorflow.org/versions/r0.11/api_docs/python/train/optimizers#Optimizer.apply_gradients
        #   zip                 https://docs.python.org/3.3/library/functions.html#zip
        # Please use AdamOptimizer
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.get_params()))

    def create_actor_network(self, scope):
        with tf.variable_scope(scope):
            X = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            fc1 = tf.contrib.layers.fully_connected(X, 400, activation_fn=tf.nn.relu,
                weights_initializer=tf.random_uniform_initializer(minval=-tf.div(1., tf.sqrt(3.)), maxval=tf.div(1., tf.sqrt(3.))),
                biases_initializer=tf.zeros_initializer())
            fc2 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=tf.nn.relu,
                weights_initializer=tf.random_uniform_initializer(minval=-tf.div(1., tf.sqrt(400.)), maxval=tf.div(1., tf.sqrt(400.))),
                biases_initializer=tf.zeros_initializer())

            out = tf.contrib.layers.fully_connected(fc2, self.a_dim, activation_fn=tf.nn.tanh, weights_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003), biases_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
            scaled_out = tf.multiply(out, self.action_bound)
        return X, out, scaled_out


    def train(self, X, a_gradient):
        return self.sess.run(self.optimize, feed_dict={self.input: X, self.action_gradient: a_gradient})

    def predict(self, X):
        return self.sess.run(self.scaled_out, feed_dict={self.input: X})

    def get_scope(self):
        return self.scope

    def get_params(self):
        return self.network_params


class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, scope):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.scope = scope

        # Create the critic network
        self.input, self.action, self.out = self.create_critic_network(scope=self.scope)

        # network parameters
        self.network_params = sorted([t for t in tf.trainable_variables() if t.name.startswith(self.get_scope())], key=lambda v: v.name)

        self.target_q_value = tf.placeholder(tf.float32, [None, 1])

        # loss & optimize op
        self.loss = tf.losses.mean_squared_error(labels=self.target_q_value, predictions=self.out, weights=1./MINIBATCH_SIZE)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # compute the partial derivatives of self.out with respect to self.action
        # Hint: tf.gradients()      https://www.tensorflow.org/versions/r1.2/api_docs/python/tf/gradients
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, scope):
        with tf.variable_scope(scope):
            X = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            action = tf.placeholder(tf.float32, shape=[None, self.a_dim])
            fc1 = tf.contrib.layers.fully_connected(X, 400, activation_fn=tf.nn.relu,
                weights_initializer=tf.random_uniform_initializer(minval=-tf.div(1., tf.sqrt(3.)), maxval=tf.div(1., tf.sqrt(3.))),
                biases_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                biases_regularizer=tf.contrib.layers.l2_regularizer(0.01))

            t1 = tf.contrib.layers.fully_connected(fc1, 300, activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(minval=-tf.div(1., tf.sqrt(400.)), maxval=tf.div(1., tf.sqrt(400.))),
                biases_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                biases_regularizer=tf.contrib.layers.l2_regularizer(0.01))

            t2 = tf.contrib.layers.fully_connected(action, 300, activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(minval=-tf.div(1., tf.sqrt(400.)), maxval=tf.div(1., tf.sqrt(400.))),
                biases_initializer=tf.zeros_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                biases_regularizer=tf.contrib.layers.l2_regularizer(0.01))

            fc2 = tf.nn.relu(tf.add(t1, t2))

            out = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003),
                biases_initializer=tf.random_uniform_initializer(minval=-0.0003, maxval=0.0003),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                biases_regularizer=tf.contrib.layers.l2_regularizer(0.01))

        return X, action, out


    def train(self, X, action, target_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={self.input: X, self.action: action, self.target_q_value: target_q_value})

    def predict(self, X, action):
        return self.sess.run(self.out, feed_dict={self.input: X, self.action: action})

    def action_gradients(self, X, action):
        return self.sess.run(self.action_grads, feed_dict={self.input: X, self.action: action})

    def get_scope(self):
        return self.scope

    def get_params(self):
        return self.network_params

class OU_process:
    # Ref: https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    def __init__(self, theta, mu, sigma):
        self.X = 0.
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        self.X = self.X + self.theta * (self.mu - self.X) + self.sigma * np.random.randn()
        return self.X

    def reset(self):
        self.X = 0.

def copy_weights_ops(sess, to_net, from_net):
    update_ops = []
    for from_v, to_v in zip(from_net.get_params(), to_net.get_params()):
        op = to_v.assign(from_v)
        update_ops.append(op)

    return update_ops

def soft_update_ops(sess, target_net, behavior_net):
    update_ops = []
    for behavior_v, target_v in zip(behavior_net.get_params(), target_net.get_params()):
        # soft update
        op = target_v.assign(TAU * behavior_v + (1 - TAU) * target_v)
        update_ops.append(op)

    return update_ops

def train(sess, env, behavior_actor, behavior_critic, target_actor, target_critic):

    sess.run(tf.global_variables_initializer())

    # copy weights ops
    actor_copy_ops = copy_weights_ops(sess, target_actor, behavior_actor)
    critic_copy_ops = copy_weights_ops(sess, target_critic, behavior_critic)

    # soft update ops
    actor_update_ops = soft_update_ops(sess, target_actor, behavior_actor)
    critic_update_ops = soft_update_ops(sess, target_critic, behavior_critic)

    # copy parameters
    sess.run([actor_copy_ops, critic_copy_ops])

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    exploration_noise = OU_process(0.15, 0., 0.3)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter("logs/", sess.graph)

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        ep_avg_max_q = 0
        exploration_noise.reset()

        for j in itertools.count():
            a = behavior_actor.predict(np.reshape(s, (1, env.observation_space.shape[0]))) + exploration_noise.noise()
            s2, r, done, _ = env.step(a[0])

            replay_buffer.add(np.reshape(s, (behavior_actor.s_dim, )), np.reshape(a, (behavior_actor.a_dim, )), r, done, np.reshape(s2, (behavior_actor.s_dim, )))

            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                target_q = target_critic.predict(s2_batch, target_actor.predict(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # update critic
                predicted_q_value, _ = behavior_critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_avg_max_q += np.amax(predicted_q_value)

                # ----- Update actor -----

                # 1. Use behavior_actor to predict actions
                predicted_a = behavior_actor.predict(s_batch)

                # 2. Use behavior critic to compute action gradients
                a_gradients = behavior_critic.action_gradients(s_batch, predicted_a)

                # 3. Update behavior actor
                behavior_actor.train(s_batch, a_gradients[0])

                # ------------------------

                # soft update target networks
                sess.run([actor_update_ops, critic_update_ops])

            s = s2
            ep_reward += r

            if done:
                print ('| Reward: %.2i' % int(ep_reward), " | Episode", i, '| Qmax: %.4f' % (ep_avg_max_q / float(j)))
                break

        ep_summary = tf.Summary()
        ep_summary.value.add(simple_value=ep_reward, tag="reward")
        summary_writer.add_summary(ep_summary, i)

        if (i + 1) % 100 == 0:
            saver.save(sess, "checkpoint_files/model", i)

def main(_):
    with tf.Session() as sess:

        env = gym.envs.make('Pendulum-v0')
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        # behavior networks
        b_actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, "ba")
        b_critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, "bc")
        # target networks
        t_actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, "ta")
        t_critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, "tc")

        if RECORD_VIDEOS:
            env = Monitor(env, directory=MONITOR_DIR, video_callable=lambda count: count % 50 == 0, force=True)

        train(sess, env, b_actor, b_critic, t_actor, t_critic)

        if RECORD_VIDEOS:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
