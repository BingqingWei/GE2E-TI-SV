import tensorflow as tf
import os
from pre.model import train, test
from pre.configuration import get_config

config = get_config()
tf.reset_default_graph()

if __name__ == "__main__":
    # start training
    if config.train:
        print("\nTraining Session")
        os.makedirs(config.model_path)
        train(config.model_path)
    # start test
    else:
        print("\nTest session")
        if os.path.isdir(config.model_path):
            test(os.path.join(config.model_path, 'check_point', 'model.ckpt-13'))
        else:
            raise AssertionError("model path doesn't exist!")