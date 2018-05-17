---
title: On Fully Convolutional Neural Network for Semantic Segmentation
layout: post
categories: [machine-learning, misc, math]
published: true

---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } },
  tex2jax: {
    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
    processEscapes: true
  }
});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<p style="text-align: justify;">
  Guess who's at work again and has nothing better to do than to stare at ceilings and write blog posts that noone reads? I do have a nice Wodehouse novel in my bag that I'm dying to get at - but I guess my boss would rather see my nose stuck in a code screen than a Jeeves omnibus... I bet Jeeves would've had a solution to my conundrum. Anyway, here's the dealio.
</p>

<p style="text-align: justify;">
  I've been reading a bunch of papers and articles and such on object detection and semantic sengmentation - learning about the standard deep learning models used for these tasks. The segmentation task has particularly captured my fancy, mainly because the results look really cool:
</p>

<!-- INSERT SEMANTIC SEGMENTATION RESULT HERE -->
  
<p style="text-align: justify;">
  One model - called the Fully Convolutional Network - has found mention all over the place in segmentation literature, and doesn't look overly complicated. Consequently, I've chosen to implement its simplest variant - FCN-8s. It's really simple to understand, there are only three salient points that one needs to remember:
</p>

<ol>
  <li style="text-align: justify;">The initial layers - about 7 of them - are the same as a VGG net. If you've never seen a VGG-Net architecture, then it would suffice to know that: a. it is used for image classfication; and b. it is about 11-16 layers deep and has only 3x3 convolutions and max poolings followed by three fully-connected layers. FCN uses the weights from a trained VGG-net for its initial layers (sort of a transfer learning thing that goes on here)</li>
  <li style="text-align: justify;">The fully connected layers of the VGG-net are replaced by 1x1 convolutions and there is an upsampling step at the very end of the network which resizes the feature map to the original input image size.</li>
  <li style="text-align: justify;">The upsampling at the very end is aided by max pooling outputs from earlier layers in the network - via "skip connections". More on this later, when we get to the implementation.</li>
</ol>

<h4>Some ntoes on development environmnet, tools and dataset</h4>
<p style="text-align: justify;">
  I'm a bit of a stickler for doing things right when it comes to development environments. I wasn't always this anal retentive, I have a certain juggler to thank for it. Anyway, here is a brief description of how I'm setting the project up: I've set up a conda environment with python 3.6 installed. I've further installed tensorflow and jupyter in the envirionment. Note that I only use conda to create the environment, all packages inside the environment are installed using pip (the reason for this is that later on, I can just run a `pip freeze > requirements.txt` to solve all the package dependencies. So, the conda environment protects my global package settings and the pip install make my environment reproducible. Pretty neat, right?
</p>

<p style="text-align: justify;">
  I'm using jupyter notebooks for initial experimentation and getting a rough model working. I'll probably consolidate it into a Python script towards te end of the project. Also, if you haven't yet figured it out - this is all going to be in Tensorflow.
</p>

<p style="text-align: justify;">
  About the dataset I've planned to train and test on the PASCAL VOC 2011 Segmentation Challenge data - which is pretty popular. Has roughly 15,000 images each in the training and testing sets. The images are not of any fixed dimensions, and each image may contain one or more instances of 20 object categories. More information about the dataset can be found [here](#).
</p>

<h4>And away we go...</h4>
<p style="text-align: justify;">
  First, I need to find a pretrained VGG-Net and load its weights (note, when I say "weights" I really mean all weights and biases) into my own VGG-Net model. I should also probably verify that the weights are correct by running the model on some images. Now, there are two issues I'll probably need to handle:
</p>

<ol>
  <li style="text-align: justify;">The original VGG-Net weights (made available by the original authors of VGG-Net) are in caffemodel format (because the guys at Oxford who wrote VGG-Net wrote it in Caffe). This needs to be converted into a format amenable to Tensorflow.</li>
  <li style="text-align: justify;">The original VGG-Net was trained to perform image classification on the ILSVRC challenge. Here we are using a different dataset and have a different application in mind. It will probably not be very easy to determine if the pretrained weights have been correctly loaded.</li>
</ol>

<p style="text-align: justify;">The first issue is resolved with some effort - and code written by others. First, we need the pretrained weights. K Simonyan and A Zisserman from the Visual Geometry Group at University of Oxford are the original authors of VGG-Net. They've kindly made the weights of a trained VGG-Net available on their group's webpage. Their original model was written in Caffe - and so, their model is made available as two files:</p>

<ol>
  <li style="text-align: justify;">A weights file that actually contains the weights and biases of the trained VGG-Net.</li>
  <li style="text-align: justify;">A model definition file which precisely defines the model architecture.</li>
</ol>

<p style="text-align: justify;">There is a very nice utility called <a href="https://github.com/ethereon/caffe-tensorflow">caffe-tensorflow</a> which can be used to convert the model weights into native numpy format. Again, thanks to <a href="https://github.com/ethereon">ethereon</a> for making this open-source. I used that, with a <a href="https://github.com/ethereon/caffe-tensorflow/issues/39#issuecomment-225670914">slight modification</a> to the model definition file to obtain the weights in native numpy format.</p>

```
$ python convert.py --caffemodel ../VGG_ILSVRC_16_layers.caffemodel --data-output-path ../weights.dat --code-output-path ../model-def.py ../VGG_ILSVRC_16_layers_deploy.prototxt
```

<p style="text-align: justify;">The above command produces the weights binary file in weights.dat and a model definition python class in model-def.py. Now from a Python REPL, we can:</p>

```
>>> import numpy as np
>>> params = np.load('../weights.dat')
>>> params = np.reshape(params, (1,))[0]
>>> params.keys()
[u'conv5_1', u'fc6', u'conv5_3', u'fc7', u'fc8', u'conv5_2', u'conv4_1', u'conv4_2', u'conv4_3', u'conv3_3', u'conv3_2', u'conv3_1', u'conv1_1', u'conv1_2', u'conv2_2', u'conv2_1']
```

<p style="text-align: justify;">
We have obtained a dict with layer wise weights. Now we must check if the weights are indeed correct. For this, we will need to write a VGG-Net model, load these weights and make some predictions on images.
</p>
