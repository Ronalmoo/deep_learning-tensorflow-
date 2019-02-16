# 알파벳을 리스트에 넣고 해당 글자의 인덱스를 수할 수 있는 딕셔너리를 만들어 준다.
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 학습에 사용할 단어를 배열로 저장
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load',
            'love', 'kiss', 'kind']

def make_batch(seq_data):
    input_batch = []
    target_batch = []
    
    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)
        
    return input_batch, target_batch

# parameter
learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3 # 한 단어 안에서(4개의 알파벳) 중 3개의 알파벳만 학습할 것이므로
n_input = n_class = dic_len

# 신경망 모델 구성

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

## 모델 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    input_batch, target_batch = make_batch(seq_data)
    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={X: input_batch, Y: target_batch})
        print('Epoch:', '%04d' % (epoch + 1))
        print('cost =', '{:.3f}'.format(loss))
    
    print('최적화 완료')
    
    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    prediction_check = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))
    
    input_batch, target_batch = make_batch(seq_data)
    predict, accuracy_val = sess.run([prediction, accuracy],
                                     feed_dict={X: input_batch, Y: target_batch})
    
    predict_words = []
    for idx, val in enumerate(seq_data):
        last_char = char_arr[predict[idx]]
        predict_words.append(val[:3] + last_char)
        
    print('====예측 결과====')
    print('입력값:', [w[:3] + ' ' for w in seq_data])
    print('예측값:', predict_words)
    print('정확도:', accuracy_val)