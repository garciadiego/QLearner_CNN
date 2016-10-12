import tensorflow as tf
import numpy as np
import gym
import random



####Hyper Parameters####
Num_actions = 6
Gamma = 0.99
Time_Steps = 500
Explore = 500
Explore_Decay = 0.99
Batch = 32
Learning_rate = 1e-6
max_episodes = 500
max_steps = 250



#Weight and Bias Initialization (TF website)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

#Convolution and Pooling
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


#Enviroment Load
env = gym.make("BipedalWalker-v2")
#action_dimensions = env.action_space.n


#Convolutional Layers Parameters

Patch_size_cv1 = 5
Patch_size2_cv1 = 5
Input_channels_cv1 = 3
Output_channels_cv1 = 25

Patch_size_cv2 = 5
Patch_size2_cv2 = 5
Input_channels_cv2 = 3
Output_channels_cv2 = 25

Patch_size_cv3 = 5
Patch_size2_cv3 = 5
Input_channels_cv3 = 3
Output_channels_cv3 = 25

output_shape = env.action_space
tf.reset_default_graph()
Input_layer = tf.placeholder(tf.float32, [None])

#Input_layer = tf.placeholder(tf.float32, [None, 80, 80, 4])

#Convolutional Layer 1
Weight_cv1 = weight_variable([Patch_size_cv1, Patch_size2_cv1, Input_channels_cv1, Output_channels_cv1])
Bias_cv1 = bias_variable([Output_channels_cv1])

hidden_cv1 = tf.nn.relu(conv2d(Input_layer, Weight_cv1) + Bias_cv1)
hidden_pool1 = max_pool_2x2(hidden_cv1)

Weight_cv2 = weight_variable([Patch_size_cv2, Patch_size2_cv2, Input_channels_cv2, Output_channels_cv2])
Bias_cv2 = bias_variable([Output_channels_cv2])

hidden_cv2 = tf.nn.relu(conv2d(hidden_pool1, Weight_cv2) + Bias_cv2)
hidden_pool2 = max_pool_2x2(hidden_cv2)

Weight_cv3 = weight_variable([Patch_size_cv3, Patch_size2_cv3, Input_channels_cv3, Output_channels_cv3 ])
Bias_cv3 = bias_variable([Output_channels_cv3])

hidden_cv3 = tf.nn.relu(conv2d(hidden_pool2, Weight_cv3) + Bias_cv3)
hidden_pool3 = max_pool_2x2(hidden_cv3)

#Densely Connected Layer Parameters (Reshape)
Total_Neurons = 512
Image_reduction = 1600  #Image reduction to 5x5

#Densely Connected Layer (Reshape)
Weight_fc1 = weight_variable([Image_reduction, Total_Neurons])
Bias_fc1 = weight_variable([Total_Neurons])

hidden_cnv_reshape = tf.reshape(hidden_pool3, [-1, Image_reducion])
hidden_fc1 = tf.nn.relu(tf.matmul(hidden_cnv_reshape, Weight_fc1) + Bias_fc1)

#Prob = weight_variable(tf.float32)
#hidden_fc1_dropout = tf.nn.dropout(hidden_fc1, Prob)

Weight_fc2 = weight_variable([Total_Neurons, output_shape])
Bias_fc2 = weight_variable([output_shape])


Q_value = tf.matmul(hidden_fc1, Weight_fc2) + Bias_fc2

######################################################################################################################
######################################################################################################################
Input_layer_prime = tf.placeholder(tf.float32, [None, shape_1, shape_2, shape_3])

#Convolutional Layer 2
Weight_cv1_prime = weight_variable([Patch_size_cv1, Patch_size2_cv1, Input_channels_cv1, Output_channels_cv1])
Bias_cv1_prime = bias_variable([Output_channels_cv1])

hidden_cv1_prime = tf.nn.relu(conv2d(Input_layer_prime, Weight_cv1_prime) + Bias_cv1_prime)
hidden_pool1_prime = max_pool_2x2(hidden_cv1_prime)

Weight_cv2_prime = weight_variable([Patch_size_cv2, Patch_size2_cv2, Input_channels_cv2, Output_channels_cv2])
Bias_cv2_prime = bias_variable([Output_channels_cv2])

hidden_cv2_prime = tf.nn.relu(conv2d(hidden_pool1_prime, Weight_cv2_prime) + Bias_cv2_prime)
hidden_pool2_prime = max_pool_2x2(hidden_cv2_prime)

Weight_cv3_prime = weight_variable([Patch_size_cv3, Patch_size2_cv3, Input_channels_cv3, Output_channels_cv3 ])
Bias_cv3_prime = bias_variable([Output_channels_cv3])

hidden_cv3_prime = tf.nn.relu(conv2d(hidden_pool2_prime, Weight_cv3_prime) + Bias_cv3_prime)
hidden_pool3_prime = max_pool_2x2(hidden_cv3_prime)


#Densely Connected Layer (Reshape)
Weight_fc1_prime = weight_variable([Image_reduction, Total_Neurons])
Bias_fc1_prime = weight_variable([Total_Neurons])

hidden_cnv_reshape_prime = tf.reshape(hidden_pool3_prime, [-1, Image_reducion])
hidden_fc1_prime = tf.nn.relu(tf.matmul(hidden_cnv_reshape_prime, Weight_fc1_prime) + Bias_fc1_prime)

#Prob_prime = weight_variable(tf.float32)
#hidden_fc1_dropout_prime = tf.nn.dropout(hidden_fc1_prime, Prob_prime)

Weight_fc2_prime = weight_variable([Total_Neurons, output_shape])
Bias_fc2_prime = weight_variable([output_shape])


Q_value_prime = tf.matmul(hidden_fc1_prime, Weight_fc2_prime) + Bias_fc2_prime

#########################################################################################################

#Q to Q'

Weight_cv1_updt = Weight_cv1_prime.assign(Weight_cv1)
Bias_cv1_updt = Bias_cv1_prime.assign(Bias_cv1)

Weight_cv2_updt = Weight_cv2_prime.assign(Weight_cv2)
Bias_cv2_updt = Bias_cv2_prime.assign(Bias_cv2)

Weight_cv3_updt = Weight_cv3_prime.assign(Weight_cv3)
Bias_cv3_updt = Bias_cv3_prime.assign(Bias_cv3)

Weight_fc1_updt = Weight_fc1_prime.assign(Weight_fc1)
Bias_fc1_updt = Bias_fc1_prime.assign(Bias_fc1)

Weight_fc2_updt = Weight_fc2_prime.assign(Weight_fc2)
Bias_fc2_updt = Bias_fc2_prime.assign(Bias_fc2)

assign_all = [Weight_cv1_updt, Bias_cv1_updt, Weight_cv2_updt, Bias_cv2_updt, Weight_cv3_updt, Bias_cv3_updt,
Weight_fc1_updt, Bias_fc1_updt, Weight_fc2_updt, Bias_fc2_updt]

#Training

reward = tf.placeholder(tf.float32, [None])
action = tf.placeholder(tf.float32, [None])
one_hot = tf.one_hot(action, 3)
read_out_action = tf.reduce_sum(tf.mul(Q_value, one_hot), reduction_indices = 1)
Loss = tf.reduce_mean(tf.square(reward - read_out_action))
train_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss)

init = tf.initialize_all_variables()

#Enviroment set up

D = []          #empty array to hold values
explore = 1.0
rewardList = []
past_actions = []
episode_number = 0
episode_reward = 0
reward_sum = 0

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    sess.run(assign_all)

    for episode in xrange(max_episodes):
        print 'Reward for episode %f is %f. Explore is %f' %(episode,reward_sum, explore)
        reward_sum = 0
        new_state = env.reset()

        for step in xrange(max_steps):
            if episode % batch_number == 0:
                env.render()

            state = list(new_state);

            if explore > random.random():
                action_sample = env.action_space.sample()
                action = np.argmax(action_sample)

            else:
                results = sess.run(action_values, feed_dict={states: np.array([new_state]), keep_prob_ : 1})
                action = (np.argmax(results[0]))

            curr_action = action;
            action_temp = [-1.0,1.0,0.0]
            new_state, reward, done, _ = env.step(action_temp)
            reward_sum += reward

            D.append([state, curr_action, reward, new_state, done])

            if len(D) > 5000:
                D.pop(0)

            sample_size = len(D)
            if sample_size > 500:
                sample_size = 500
            else:
                sample_size = sample_size

            if True:
                samples = [ D[i] for i in random.sample(xrange(len(D)), sample_size) ]
                new_states_for_q = [ x[3] for x in samples]
                all_q_prime = sess.run(Q_, feed_dict={images_: new_states_for_q, keep_prob_ : 1})
                y_ = []
                states_samples = []
                next_states_samples = []
                actions_samples = []
                for ind, i_sample in enumerate(samples):

                    if i_sample[4] == True:

                        y_.append(reward)

                    else:
                        this_q_prime = all_q_prime[ind]
                        maxq = max(this_q_prime)
                        y_.append(reward + (gamma * maxq))

                    states_samples.append(i_sample[0])
                    next_states_samples.append(i_sample[3])
                    actions_samples.append(i_sample[1])

                sess.run(train, feed_dict={images: states_samples, rewards: y_, keep_prob : .7, actions: actions_samples})


                if done:
                    break

        if episode % num_of_episodes_between_q_copies == 0:
            sess.run(assign_all)

writer = tf.train.SummaryWriter("/tmp/QLearner/cartpole", sess.graph)

#env.monitor.close()
