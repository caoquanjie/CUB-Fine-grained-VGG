# CUB-Fine-grained-VGG
## A tensorflow version of CUB fine-grained recognition using VGG model(pretrained on Imagenet).

CUB-200-2011 Birds Dataset is a famous dataset for fine-grained classification 
that contains 6k training and 6k test images of 200 species of birds, each with roughly 30 training images and 30 testing images.
The dataset can be download in [Caltech-UCSD Webpage](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).</br>
The pretrained model on Imagenet can be download in [VGG19NPY](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)
In this repository, I train a pretrained VGG model in tensorflow and the accuracy is reached `77.4%`.

## Requirements
python 3.6</br>
tensorflow 1.4.0</br>
numpy 1.15.0</br>
matplotlib 2.0.0

## Training details
I processed this dataset itno `tfrecords` format without bounding box, and resize the original image to 256 pixels and crop it randomly to 224.
I start the training process from a pre-trained model on Imagenet. 
First, I finetune the model using only fc8 layer with learning rate of 1e-3 for 5000 steps 
and then train all variables(including convolutional layers) with learning rate of 1e-3 for 10000steps.
Finally, use the learning rate of 1e-4 to train 10,000 steps in the same way as before. 
I chose SGD later and then I got `77.4%` accuracy. The model was implemented by Tensorﬂow 1.4 
and was trained on a workstation with NVIDIA Titan X GPU and 32Gb system RAM.

## Usage
First, after you download the dataset, run `python dataset_to_tfrecords.py` to get `train.tfrecords` and `test.tfrecords`.</br>
Then run `python vgg_finetune.py --batch_size 32 --learning_rate1 1e-3 --learning_rate2 1e-3 --learning_rate3 1e-4 --learning_rate4 1e-5 --data_dir your dir`.

