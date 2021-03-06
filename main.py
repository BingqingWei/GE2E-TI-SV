import shutil
from models.rnn import *
from models.cnn import *
from config import *

def main():
    if config.mode == 'train':
        if os.path.exists(config.model_path):
            shutil.rmtree(config.model_path)

    if not os.path.exists(config.model_path):
        os.mkdir(os.path.join(config.model_path))
    if config.redirect_stdout:
        print('stdout redirected')
        sys.stdout = open(os.path.join(config.model_path, config.redirect_fname), 'w')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
    tf.reset_default_graph()
    sess = tf.Session(config=tf_config)
    model = GRU_Model()

    if config.mode == 'train':
        print("\nTraining Session")
        shutil.copy(os.path.join('.', 'config.py'), config.model_path)
        model.train(sess, config.model_path)

    elif config.mode == 'test':
        print("\nTest Session")
        if os.path.isdir(config.model_path):
            model.test(sess, get_latest_ckpt(os.path.join(config.model_path, 'check_point')))
        else:
            raise AssertionError("model path doesn't exist!")

    else:
        print("\nInfer Session")
        model.restore(sess, path=get_latest_ckpt(os.path.join(config.model_path, 'check_point')))
        model.infer(sess, thres=config.infer_thres)

if __name__ == "__main__":
    main()
