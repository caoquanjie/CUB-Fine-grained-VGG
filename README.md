# CUB-Fine-grained-VGG
## a tensorflow version of CUB fine-grained recognition using VGG model(pretrained on Imagenet).

CUB-200-2011 Birds Dataset is a famous dataset for fine-grained classification 
that contains 6k training and 6k test images of 200 species of birds, each with roughly 30 training images and 30 testing images.
The dataset can be download in [Caltech-UCSD Webpage](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).</br>
In this repository, I train a pretrained VGG model in tensorflow and the accuracy is reached `77.4%`.

##requirements
python 3.6</br>
tensorflow 1.4.0</br>
numpy 1.15.0
matplotlib 2.0.0

## training details
I start the training process from a pre-trained model on Imagenet. 
First, I finetune the model using only fc8 layer with learning rate of 1e-3 for 5000 steps 
and then train all variables(including convolutional layers) with learning rate of 1e-3 for 10000steps.
Finally, use the learning rate of 1e-4 to train 10,000 steps in the same way as before. 
I chose SGD later and then I got 77.4% accuracy. The model was implemented by Tensorﬂow 1.4 
and was trained on a workstation with NVIDIA Titan X GPU and 32Gb system RAM.


