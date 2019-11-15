import tensorflow as tf
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)
    input("Input:")