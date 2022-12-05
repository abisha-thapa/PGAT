# Performance Analysis of Graph Neural Network on Small Point Cloud Object

This is a course research project. We have considered Point-GNN as the baseline of this project. Pedestrians and cyclists objects are comparatively smaller than car and are objects of interest for this project. 
We have changed the feature update process of GNN in Point-GNN with an attention mechanism.


If you find this code useful in your research, please consider citing their work:
```ruby
@InProceedings{Point-GNN,
author = {Shi, Weijing and Rajkumar, Ragunathan (Raj)},
title = {Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

# Getting started

## Prerequisites

We use Tensorflow 1.15 for this implementation. Please install CUDA if you want GPU support.
```
pip3 install --user tensorflow-gpu==1.15.0
```
To install other dependencies:
```
pip3 install --user opencv-python
pip3 install --user open3d-python==0.7.0.0
pip3 install --user scikit-learn
pip3 install --user tqdm
pip3 install --user shapely
```
# Download PointGAT
Clone the repository recursively:
```
git clone git@github.com:abisha-thapa/PointGAT.git --recursive
```
# KITTI Dataset
Following is the KITTI dataset structure:
```
DATASET_ROOT_DIR
├── image                    #  Left color images
│   ├── training
|   |   └── image_2            
│   └── testing
|       └── image_2 
├── velodyne                 # Velodyne point cloud files
│   ├── training
|   |   └── velodyne            
│   └── testing
|       └── velodyne 
├── calib                    # Calibration files
│   ├── training
|   |   └──calib            
│   └── testing
|       └── calib 
├── labels                   # Training labels
│   └── training
|       └── label_2
└── 3DOP_splits              # split files.
    ├── train.txt
    ├── train_car.txt
    └── ...
```

## Training
We put training parameters in a train_config file. Enable feature update process in a train_config to attention mechanism by adding following in each graph neural network layer.
```ruby
  "attention": true
```
To start training, we need both the train_config and config.

```
usage: train.py [-h] [--dataset_root_dir DATASET_ROOT_DIR]
                [--dataset_split_file DATASET_SPLIT_FILE]
                train_config_path config_path

positional arguments:
  train_config_path     Path to train_config
  config_path           Path to config

optional arguments:
  -h, --help            show this help message and exit
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split file.Default="DATASET_ROOT
                        _DIR/3DOP_splits/train_config["train_dataset"]"
```
For example:
```
python3 train.py configs/ped_cyl_auto_T2_attn_trainval_train_config configs/ped_cyl_auto_T2_attn_trainval_config --dataset_root_dir dataset --dataset_split_file splits/train_attn_pedestrian_cyclist.txt
```

## Inference
### Run a checkpoint
Test on the validation split:
```
python3 run.py checkpoints/ped_cyl_auto_attn_T2_trainval/ --dataset_root_dir DATASET_ROOT_DIR --output_dir DIR_TO_SAVE_RESULTS
```
Test on the test dataset:
```
python3 run.py checkpoints/ped_cyl_auto_attn_T2_trainval/ --test --dataset_root_dir dataset --output_dir DIR_TO_SAVE_RESULTS
```
```
usage: run.py [-h] [-l LEVEL] [--test] [--no-box-merge] [--no-box-score]
              [--dataset_root_dir DATASET_ROOT_DIR]
              [--dataset_split_file DATASET_SPLIT_FILE]
              [--output_dir OUTPUT_DIR]
              checkpoint_path

positional arguments:
  checkpoint_path       Path to checkpoint

optional arguments:
  -h, --help            show this help message and exit
  -l LEVEL, --level LEVEL
                        Visualization level, 0 to disable,1 to nonblocking
                        visualization, 2 to block.Default=0
  --test                Enable test model
  --no-box-merge        Disable box merge.
  --no-box-score        Disable box score.
  --dataset_root_dir DATASET_ROOT_DIR
                        Path to KITTI dataset. Default="../dataset/kitti/"
  --dataset_split_file DATASET_SPLIT_FILE
                        Path to KITTI dataset split
                        file.Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"
  --output_dir OUTPUT_DIR
                        Path to save the detection
                        resultsDefault="CHECKPOINT_PATH/eval/"
 ```


### Performance
Install kitti_native_evaluation offline evaluation:
```
cd kitti_native_evaluation
cmake ./
make
```
Evaluate output results on the validation split:
```
evaluate_object_offline DATASET_ROOT_DIR/labels/training/label_2/ DIR_TO_SAVE_RESULTS
```

We can view train_config for training and make change in parameter depending on our evaluation.
Some common parameters which you might want to change first:
```
train_dir     The directory where checkpoints and logs are stored.
train_dataset The dataset split file for training. 
NUM_GPU       The number of GPUs to use. We used two GPUs for the reference model. 
              If you want to use a single GPU, you might also need to reduce the batch size by half to save GPU memory.
              Similarly, you might want to increase the batch size if you want to utilize more GPUs. 
              Check the train.py for details.               
```
We also provide an evaluation script to evaluate the checkpoints periodically. For example:
```
python3 eval.py configs/ped_cyl_auto_attn_T2_train_eval_config
```
You can use a tensorboard to view the training and evaluation status. 
```
tensorboard --logdir=./train_dir
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
