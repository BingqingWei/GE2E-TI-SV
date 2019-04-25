__author__ = 'Bingqing Wei'
from main import *
from adv import infer_verif_path, infer_enroll_path
import random
from distutils.dir_util import copy_tree

speaker_path = os.path.join(work_dir, 'Speakers')
failure_path = os.path.join(work_dir, 'failure')

def cross_test(func):
    speakers = os.listdir(speaker_path)
    files = dict()
    for speaker in speakers:
        l = []
        for file in os.listdir(os.path.join(speaker_path, speaker)):
            l.append(os.path.join(speaker_path, speaker, file))
        files[speaker] = l

    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for i in range(len(speakers)):
        for j in range(len(speakers)):
            enrolls = files[speakers[i]]
            verifs = files[speakers[j]]
            if i == j:
                for k in range(len(enrolls) - 1):
                    copy_to_infer(enrolls[:k] + enrolls[k + 2:], [enrolls[k], enrolls[k + 1]])
                    try:
                        if not func():
                            print('Failure in detecting speaker-{}'.format(speakers[i]))
                            save_failure()
                            false_negative += 1
                        else:
                            true_positive += 1
                    except Exception as ex:
                        print('Skipping')
            else:
                for k in range(7):
                    enroll_idx = random.sample(range(len(enrolls)), min(len(enrolls), 4))
                    verif_idx = random.sample(range(len(verifs)), min(len(verifs), 2))
                    copy_to_infer([enrolls[x] for x in enroll_idx], [verifs[x] for x in verif_idx])
                    try:
                        if func():
                            print('Failure in distinguishing speaker-{} & speaker-{}'
                                  .format(speakers[i], speakers[j]))
                            save_failure()
                            false_positive += 1
                        else:
                            true_negative += 1
                    except Exception as ex:
                        print('Skipping')
    print('false negative rate-{}\n'
          'false positive rate-{}\n'
          'accuracy-{}\n'
          'total-{}'.format(
        false_positive / (false_positive + true_negative),
        false_negative / (false_negative + true_positive),
        (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative),
        int(false_negative + false_positive + true_negative + true_positive)
    ))



def save_failure():
    if not os.path.exists(failure_path):
        os.mkdir(failure_path)
        start_id = 0
    else:
        start_id = len(os.listdir(failure_path))
    copy_tree(config.infer_path, os.path.join(failure_path, '{}'.format(start_id)))

def copy_to_infer(enrolls, verifs):
    if os.path.exists(config.infer_path):
        shutil.rmtree(config.infer_path)
    os.mkdir(config.infer_path)
    os.mkdir(infer_enroll_path)
    os.mkdir(infer_verif_path)
    for e in enrolls:
        shutil.copy(e, infer_enroll_path)
    for v in verifs:
        shutil.copy(v, infer_verif_path)

if __name__ == '__main__':
    assert config.mode == 'infer'
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_fraction
    tf.reset_default_graph()
    sess = tf.Session(config=tf_config)
    model = GRU_Model()
    model.restore(sess, get_latest_ckpt(os.path.join(config.model_path, 'check_point')))
    print("\nInfer Session")
    cross_test(lambda : model.infer(sess, thres=config.infer_thres))

