import pickle, tensorflow as tf, numpy as np, argparse, matplotlib.pyplot as plt, tf_util, os

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

#Humanoid
def create_model():
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None,376])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None,17])
    W0 = tf.get_variable(name="w0",shape=[376,600], initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(name="w1",shape=[600,600], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[600,17], initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=[600], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[600], initializer=tf.constant_initializer(0.))
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
'''
#Ant
def create_model():
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None,111])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None,8])
    W0 = tf.get_variable(name="w0",shape=[111,50], initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(name="w1",shape=[50,20], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[20,8], initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=[50], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[20], initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[8], initializer=tf.constant_initializer(0.))

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

#HalfCheetah & Walker2d
def create_model():
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None,17])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None,6])
    W0 = tf.get_variable(name="w0",shape=[17,400], initializer=tf.contrib.layers.xavier_initializer())
    W1 = tf.get_variable(name="w1",shape=[400,400], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', shape=[400,6], initializer=tf.contrib.layers.xavier_initializer())

    b0 = tf.get_variable(name='b0', shape=[400], initializer=tf.constant_initializer(0.))
    b1 = tf.get_variable(name='b1', shape=[400], initializer=tf.constant_initializer(0.))
    b2 = tf.get_variable(name='b2', shape=[6], initializer=tf.constant_initializer(0.))

    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    activations = [tf.nn.relu, tf.nn.relu, None]

    layer = input_ph
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer

    return input_ph, output_ph, output_pred
'''
def train(envname):
    with open('expert_data/'+ envname + '.pkl', 'rb') as f:
        data = pickle.loads(f.read())
    inputs = data['observations']
    outputs = data['actions']

    sess = tf_reset()
    input_ph, output_ph, output_pred = create_model()
    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))
    opt = tf.train.AdamOptimizer().minimize(mse)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    batch_size = len(inputs)//2
    for training_step in range(5000):
        indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
        input_batch = inputs[indices]
        output_batch = outputs[indices,0]
        _, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})

        if training_step % 1000 == 0:
            print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
            saver.save(sess, 'training/%s.ckpt'%envname)

def run(envname, num_rollouts, max_timesteps=None, render=True):
    sess = tf_reset()
    print('loading and building imitation policy')
    input_ph, output_ph, output_pred = create_model()
    saver = tf.train.Saver()
    saver.restore(sess, "training/%s.ckpt"%(envname))
    print('loaded and built')
    with tf.Session():
        tf_util.initialize()
        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = sess.run(output_pred, feed_dict={input_ph: obs[None,:]})
                obs, r, done, _ = env.step(action)
                observations.append(obs)
                actions.append(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('learner_data', envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    #parser.add_argument('rollouts', type=str)
    args = parser.parse_args()
    train(args.envname)
    #predict(envname)
    #run(args.envname, 20)

if __name__ == '__main__':
    main()
