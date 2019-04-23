# W4995 deep learning project (Generalized End-to-End Loss for Speaker Verification)

Authors: Bingqing Wei, Yue Luo, Gleb Vizitiv

The project is built based on the repository at 'https://github.com/Janghyun1230/Speaker\_Verification'

### Google Cloud tutorial
Using Google Cloud for computation.
`$ gcloud compute scp [LOCAL_FILE_PATH] [INSTANCE_NAME]:~`

`$ gcloud compute scp --recurse [INSTANCE_NAME]:[REMOTE_DIR] [LOCAL_DIR]`

### Hardware
We used following hardware settings
- 4 vCPUs, 16 GB RAM
- 100+ GB SSD
- Nvidia K80 12GB VRAM (8GB VRAM is enough for the largest model)

### Python Packages
- python 3.5.3
- tensorflow-gpu 1.13.1
- numpy 1.16.2
- librosa 0.6.3
- pyaudio x.x.x

### Run the code
1. git clone

2. Download the dataset [Voxceleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)

### Data Preprocessing
Change the work\_dir and data\_dir in `config.py`. Run `python data.py` for data preprocessing.

### Training
Run `python main.py` for training. Before you run the training, you should set all the parameters you need in `config.py`

### Testing
Set mode as `'test'` in `config.py` first. Then run `python main.py` for testing.

### Inference
Set mode as `'infer'` in `config.py` first. Then run `python audio.py` for inference.
