# A Naive Approach to Deblur Photos Using Pix2Pix

(WARNING: Only a demonstrative project. Far from production ready.)

[Blog post](https://medium.com/@ceshine/deblur-photos-using-generic-pix2pix-6f8774f9701e)

## Pre-requisites

  * Nvidia GPU (Sorry no CPU mode)
  * pytorch 0.2.0
  * The dataset should be organized as:
      * dataset_root/
          * train/
              * |arbitrary_naming|/
                  * train images
          * val/
              * |arbitrary_naming|/
                  * validation images

The easiest way to reproduce results is to use nvidia-docker. This project is tested
on ceshine/cuda-pytorch:0.2.0 image. There is also a Dockerfile in the root folder. Use `docker build -t deblur .` to build the docker image.

## Training

The following assumes you have built a Docker image named *deblur*.

Start the docker image (replace `/mnt/SSD_Data/mirflickr/` with the path to your dataset):

```
nvidia-docker run -ti --init -v /mnt/SSD_Data/mirflickr/:/data \
                  --name deblur-c --ipc=host deblur bash
```

Start training (use `python train.py -h` to see the list of available command-line parameters):

```
python train.py --dataset /data --batchSize 16 --nEpochs 200 \
                --cuda --testBatchSize 32 --lamb 5 --lrG 5e-5 --lrD 5e-5
```

Debug images will be write to `debug` folder. Modek checkpoints will be write to `checkpoint`.

## Predicting / Deblurring

**Work In Progress** (deblur.py)

Currently only photos with longer edge shorter than 512 px are supported (Given that your GPU has enough RAM).
