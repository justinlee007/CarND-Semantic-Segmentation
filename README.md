# Semantic Segmentation Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project uses a Fully Convolutional Network (FCN) to classify areas of road/non-road in a series of images.  The process of semantic segmentation assigns an object classification to _each pixel_ of an image.  The FCN used in this project mimics the one built in [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

![](output_sample.gif)
## Dependencies
Make sure you have the following is installed:
- [Python 3](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
## Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

The dataset consists of 289 training and 290 test images with three different categories of road scenes:
- uu - urban unmarked (98/100)
- um - urban marked (95/96)
- umm - urban multiple marked lanes (96/94)

## Runing the Code
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

# Project Goals and [Rubric](https://review.udacity.com/#!/rubrics/989/view)
The goals of this project are the following:

* The function `load_vgg` is implemented correctly.
* The function `layers` is implemented correctly.
* The function `optimize` is implemented correctly.
* The function `train_nn` is implemented correctly.
* Ensure all the unit tests.
* The loss of the network should be printed while the network is training.

## FCN Architecture
Fully convolutional networks are trained end-to-end and include an encoding/decoding phase that preserves spatial information throughout the entire network.  Included with the decoding phase are skip connections.

![](fcn_arch.png)

### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
# Results

# Lessons Learned

