# !/usr/bin/python
# -- coding:utf8 --


import tensorflow as tf
from tensorflow.contrib import rnn


inputs = tf.placeholder(tf.float32, [None, 64, 5])
lstm_cell_fw_1 = rnn.LSTMCell(num_units=128)
lstm_cell_fw_2 = rnn.LSTMCell(num_units=100, num_proj=200)
lstm_cell_bw_1 = rnn.LSTMCell(num_units=64)
lstm_cell_bw_2 = rnn.LSTMCell(num_units=50)
lstm_cell_fw = rnn.MultiRNNCell(cells=[lstm_cell_fw_1, lstm_cell_fw_2])
lstm_cell_bw = rnn.MultiRNNCell(cells=[lstm_cell_bw_1, lstm_cell_bw_2])

outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                 cell_bw=lstm_cell_bw,
                                                 inputs=inputs, dtype=tf.float32)
outputs_fw = outputs[0]  # [-1, 64, 100]
print(outputs_fw.shape)
outputs_bw = outputs[1]  # [-1, 64, 50]
print(outputs_bw.shape)

state_fw_2 = state[0][1]
state_fw_2_c = state[0][1][0]  # state: [fw, bw]  fw:[一层， 二层]（bw同理）  一层: [c, h]
state_fw_2_h = state[0][1][1]
print('state_fw_1_h: ', state_fw_2_h.shape)  # [-1, 100]
print('state_fw_1_c: ', state_fw_2_c.shape)  # [-1, 100]
# 单层的outputs 包含所有time的输出h，格式为[batch_size, time_steps, cell_units]
# 单层的state   包含最后一个time_step的输出h以及状态c.格式均为[batch_size, cell_units]
# tf.keras.layers.RNN