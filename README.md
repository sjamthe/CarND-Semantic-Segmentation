# Semantic Segmentation
### Introduction
This project demonstrates use of Fully Convolutional Network (FCN) for semantic segmentation (label the pixels of a road).
It follows the concepts published in paper [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Evan Shelhamer, and Trevor Darrell. The model is explained well in [this](http://techtalks.tv/talks/fully-convolutional-networks-for-semantic-segmentation/61606/) video.

### Implement

For encoder we use VGG-16 model pre-trained on Imagenet classification. We take the output of last FC-4096 layer (we drop FC-1000 layer) and perform  a 1x1 convolution followed by a transpose convolution of 4x4 with 2x2 strides. to increase clarity we add skip layers by taking output of maxpool layer4 and perform a 1x1 convolution then add this layer to the previous output. We perform another transpose convolution of 4x4 with 2x2 strides to upsample further. We repeat this process once again by taking output of maxpool layer3 and performing a 1x1 convolution, then adding it with previous output. We then perform final transpose convolution of 16x16 and stride of 4x4.
This gives us output which has the same size as the input image but each layer corresponds to each class (thus we get a mask defining pixels in each layer). (all these decoding layers are added in layers module).

### Training
We trained the model using Kitti Road dataset (see below). 
The model was trained for 50 epochs and batch size of 5, on a Macbook Pro for 30+ hours with learning rate of .0009
As you can see after 44 epochs model was not gaining much. 
![](images/training.jpg) 

### Testing
The following 4 images were randomly chosen from a test dataset.
![](images/img1.jpg) 
![](images/img2.jpg) 
![](images/img3.jpg) 
![](images/img4.jpg) 
### Inference
I applied the model on a [video](https://www.youtube.com/watch?v=OdZJMFDMVc8) taken by my dashcam. It is not perfect in indentifying the road but does ok for short amount ot training.

[![Semantic Segmentation Video](https://img.youtube.com/vi/OdZJMFDMVc8/0.jpg)](https://www.youtube.com/watch?v=OdZJMFDMVc8)


##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

