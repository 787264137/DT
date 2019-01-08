# coding:utf-8
import tensorflow as tf


def conv2d(inputs, filters, kernel_size, name):
    with tf.name_scope(name=name):
        outputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, padding='same')
        outputs = tf.layers.batch_normalization(outputs)
        outputs = tf.nn.relu(outputs)
        return outputs


def cnn(inputs):
    outputs = conv2d(inputs, 32, 3, 'conv1')
    outputs = tf.layers.max_pooling2d(outputs, 2, 2, name='pool1')

    outputs = conv2d(outputs, 64, 3, 'conv2')
    outputs = tf.layers.max_pooling2d(outputs, 2, 2, name='pool2')

    outputs = conv2d(outputs, 128, 3, 'conv3')
    outputs = tf.layers.max_pooling2d(outputs, 2, 2, name='pool3')

    size = outputs.get_shape().as_list()
    outputs = tf.layers.average_pooling2d(outputs, pool_size=size[0], strides=size[0], name='global_avg')

    outputs = tf.squeeze(outputs)
    outputs = tf.layers.dense(outputs, units=1)
    return outputs


def _generate_data_and_label_batch(data, label, min_queue_examples,
                                   batch_size, shuffle):
    """Construct a queued batch of data and labels.
    Args:
      data: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of data per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      data: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [data, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size, # 定义预取样空间的大小
            min_after_dequeue=min_queue_examples)
        # 定义了随机取样的缓冲区大小,此参数越大表示更大级别的混合但是会导致启动更加缓慢,并且会占用更多的内存

    else:
        images, label_batch = tf.train.batch(
            [data, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return data, tf.reshape(label_batch, [batch_size])


if __name__ == '__main__':

    train_data, train_label = [], []
    valid_data, valid_label = [], []
    test_data, test_label = [], []

    batch_size = 64
    input_size = []
    inputs = tf.placeholder(dtype=tf.float64, shape=[batch_size] + input_size, name='inputs')
    target = tf.placeholder(dtype=tf.float64, shape=[batch_size, 1], name='target')

    logit = cnn(inputs)

    loss = tf.losses.sigmoid_cross_entropy(target, logit)

    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss=loss)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(1000):
            train_data_batch, label_batch = _generate_data_and_label_batch(train_data,train_label)
            feed_dict = {inputs: train_data, target: train_label}
            sess.run(train_step, feed_dict=feed_dict)
