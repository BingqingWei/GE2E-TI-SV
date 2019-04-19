import shutil
from models.base import *
from models.rnn import *
from config import *

if __name__ == "__main__":
    if config.redirect_stdout:
        sys.stdout = open(os.path.join('.', 'output.txt'), 'w')
        print('stdout redirected')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
    tf.reset_default_graph()
    sess = tf.Session(config=tf_config)
    model = LSTM_Model()
    if config.mode == 'train':
        print("\nTraining Session")
        if os.path.exists(config.model_path):
            shutil.rmtree(config.model_path)
        os.makedirs(config.model_path)
        model.train(sess, config.model_path)

    elif config.mode == 'test':
        print("\nTest Session")
        if os.path.isdir(config.model_path):
            model.test(sess, os.path.join(config.model_path, 'check_point', 'model.ckpt-5'))
        else:
            raise AssertionError("model path doesn't exist!")
    else:
        print("\nInfer Session")
        model.infer(sess, path=os.path.join(config.model_path, 'check_point', 'model.ckpt-8'), thres=0.57)