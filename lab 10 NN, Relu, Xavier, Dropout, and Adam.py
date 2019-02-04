# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 ~ 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

# Xavier initializer
W1 = tf.get_variable("W1", shape=[784, 512],
                                  initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name='bias1')
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# dropout(keep_prob) rate 0.7 on training, but should be 1 for testing
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512],
                                  initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512],
                                  initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name='bias3')
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512],
                                  initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name='bias4')
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 10],
                                  initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name='bias5')
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch
            
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        # Test model
        is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, 
                                                               keep_prob: 1}))

"""
Epoch: 0001 cost = 0.482302167
Accuracy:  0.9554
Epoch: 0002 cost = 0.173120963
Accuracy:  0.9668
Epoch: 0003 cost = 0.133391453
Accuracy:  0.9717
Epoch: 0004 cost = 0.108938191
Accuracy:  0.9753
Epoch: 0005 cost = 0.096030890
Accuracy:  0.9758
Epoch: 0006 cost = 0.082554804
Accuracy:  0.9778
Epoch: 0007 cost = 0.076522581
Accuracy:  0.9776
Epoch: 0008 cost = 0.068917474
Accuracy:  0.9807
Epoch: 0009 cost = 0.062527569
Accuracy:  0.981
Epoch: 0010 cost = 0.062774394
Accuracy:  0.9792
Epoch: 0011 cost = 0.055174339
Accuracy:  0.9795
Epoch: 0012 cost = 0.052368757
Accuracy:  0.9807
Epoch: 0013 cost = 0.048064015
Accuracy:  0.981
Epoch: 0014 cost = 0.048202867
Accuracy:  0.9825
Epoch: 0015 cost = 0.043365308
Accuracy:  0.9808
"""
