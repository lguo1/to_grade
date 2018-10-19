import pickle
import tensorflow as tf
import numpy as np
import argparse
import tf_util
import os
import load_policy

def tf_reset(seed = 0):
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    return sess

def create_model():
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None,376])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None,17])
    W0 = tf.get_variable(name="w0",shape=[376,200], initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(name="w1",shape=[200,200], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[200,17], initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=[200], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[200], initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[17], initializer=tf.constant_initializer(0.))

    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    activations = [tf.nn.relu, tf.nn.relu, None]

    # create computation grnp.random.seed(0)aph
    layer = input_ph
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer

    return input_ph, output_ph, output_pred

def imitate(envname, dagger_step, iteration):
    print('imitating expert ...')
    with open('expert_data/'+ envname + '_' + str(dagger_step) + '.pkl', 'rb') as f:
        data = pickle.loads(f.read())
    inputs = data['observations']
    outputs = data['actions']

    imitator = tf_reset()
    input_ph, output_ph, output_pred = create_model()
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))
    opt = tf.train.AdamOptimizer().minimize(mse)
    
    imitator.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if dagger_step != 0:
        saver.restore(imitator, "dagger/%s.ckpt"%envname)

    batch_size = len(inputs)//2
    print("batch size %d"%batch_size)
    for training_step in range(iteration):
        indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
        input_batch = inputs[indices]
        output_batch = outputs[indices,0]
        _, mse_run = imitator.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})

        if training_step % 1000 == 0:
            print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
            saver.save(imitator, 'dagger/%s.ckpt'%envname)

def expert(envname, dagger_step, num_rollouts, max_timesteps=None, render=False):
    print('generating expert data ...')
    imitator = tf_reset()
    input_ph, output_ph, output_pred = create_model()
    saver = tf.train.Saver()
    saver.restore(imitator, "dagger/%s.ckpt"%envname)
    policy_fn = load_policy.load_policy('experts/' + envname + '.pkl')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                imitation = imitator.run(output_pred, feed_dict={input_ph: obs[None,:]})
                obs, r, done, _ = env.step(imitation)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)
        mean = np.mean(returns)
        std = np.std(returns)
        print("mean %d, std %d" %(mean, std))
        with open('expert_data/'+ envname + '_' + str(dagger_step) +'.pkl', 'rb') as f:
            expert_data = pickle.loads(f.read())
        expert_data['observations'] = np.concatenate((expert_data['observations'], np.array(observations)))
        expert_data['actions'] = np.concatenate((expert_data['actions'], np.array(actions)))
        with open('expert_data/'+ envname + '_' + str(dagger_step + 1) +'.pkl', 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
        return mean, std

def dagger(envname, iter, bc_iter):
    stats = [[], []]
    for step in range(iter):
        print("---------------------------")
        print('dagger %d' %step)
        imitate(envname, step, bc_iter)
        mean, std = expert(envname, step, 20)
        stats[0].append(mean)
        stats[1].append(std)
    print("result", stats)
    print("saving the result to dagger")
    with open('dagger/'+ envname + '.pkl', 'wb') as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('iter', type=int)
    parser.add_argument('--bc', type=int, default=10000)
    args = parser.parse_args()
    dagger(args.envname, args.iter, args.bc)

main()
