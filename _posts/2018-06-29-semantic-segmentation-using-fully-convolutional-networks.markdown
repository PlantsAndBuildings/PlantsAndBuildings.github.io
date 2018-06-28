---
title: Semantic Segmentation Using Fully Convolutional Networks
layout: post
categories: [machine-learning, deep-learning, computer-vision]
published: true

---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  TeX: { equationNumbers: { autoNumber: "AMS" } },
  tex2jax: {
    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
    processEscapes: true
  },
  CommonHTML: { scale: 85 }
});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<style>

.column {
  float: left;
  width: 33.33%;
  padding: 5px;
}

.row::after {
  content: "";
  clear: both;
  display: table;
}

</style>

<hr/>

<p style="text-align: justify">
  <b>TL,DR: This blog post presents my implementation of the <a href="https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf" target="_blank">FCN paper</a>. Code for this can be found on <a href="http://github.com/plantsandbuildings/FCN" target="_blank">my Github</a>.</b>
</p>

<hr/>

<p style="text-align: justify">
  <b>Disclaimer: What follows is not a tutorial, they're my implementation notes. If you're looking for material on what FCNs are and such - then you probably don't stand to gain much from reading on. For that, I suggest going through the paper linked above. What follows is probably most useful to people who are either planning to or are in the midst of implementing an FCN themselves. My aim here is to talk about details that I only found after scouring the depths of the Internet for several torturous hours.</b>
</p>

<hr/>

<ul>
  <li><a href="#section1">What is Semantic Segmentation?</a></li>
  <li><a href="#section2">How do we do it?</a></li>
  <li><a href="#section3">The Pretrained VGG-Net</a></li>
  <li><a href="#section4">Convolutionalize Dense Layers</a></li>
  <li><a href="#section5">Score Layers and Color Maps</a></li>
  <li><a href="#section6">Upsampling Layers</a></li>
  <li><a href="#section7">Cropping and Skip Connections</a></li>
  <li><a href="#section8">Results</a></li>
</ul>

<hr/>

<h4 id="section1">What is Semantic Segmentation?</h4>
<p style="text-align: justify">
  <b>Semantic Segmentation</b> is the task of labeling each pixel of an input image with class labels from a predetermined set of classes. For example, given the following image:
</p>

<img src="{{ site.url }}/static/img/fcn/2007_006641.jpg"/>

<p style="text-align: justify">
  the segmentation result should look like:
</p>

<img src="{{ site.url }}/static/img/fcn/2007_006641.png"/>

<p style="text-align: justify">
  See how all the pixels in the input image that belong to the cat are color coded brown and the background is color coded black? That is the end result we hope to achieve.
</p>

<hr/>

<h4 id="section2">How do we do it?</h4>

<p style="text-align: justify">
  The title of the post gives it away - but yeah, we use Fully Convolutional Networks to perform semantic segmentation. A blazing fast walkthrough of the paper by Long, Shelhamer and Darrell: We take a VGG-Net trained for image classification on the ILSVRC dataset and "convolutionalize" it (more on this "convolutionalization" in the following sections). We then append a score layer and upsampling layers to the end of the pretrained, convolutionalized VGG-Net and then use per-pixel softmax to obtain dense predictions. Skip connections from the VGG pooling layers to the upsampling layers are also added. We do all of this using Tensorflow.
</p>

<p style="text-align: justify">
  I'll now  analyze each aspect of this walkthrough in greater detail.
</p>

<hr/>

<h4 id="section3">The Pretrained VGG-Net</h4>

<!-- Briefly introduce VGG Architecture -->
<p style="text-align: justify">
  VGG-Net is a CNN architecture originally presented by Simonyan and Zisserman for the ImageNet Classification Task 2014. AFAIK, it did not win first place that year but still went on to become a VERY popular model because of its simplicity. All five variants of VGG-Net presented in the original paper consist only of convolutional and dense layers. The FCN that I've implemented (and the one that is presented in the FCN paper) uses a pretrained VGG-16 for its initial layers. Before I proceed further, it would help to familiarize oneself with the VGG-16 architecture:
</p>

<!-- Image of VGG architecture -->
<img style="display: block; margin-left: auto; margin-right: auto;" src="{{ site.url }}/static/img/vgg-16-better.jpg"/>

<!-- Briefly introduce VGG Architecture -->
<p style="text-align: justify">
  About the notation in the above diagram: each convolution layer is specified by an ordered pair (filter size, number of output channels). Each fully connected layer is specified by a single number (number of output units). It can be seen from the diagram that the above VGG-Net has 16 "trainable" layers (that is, 16 layers with weights) - hence it is called VGG-16. If you're referring the original VGG paper, you'll find that the above architecture is referred to as the D variant. <b>For our FCN implementation, we use all the conv and pooling layers from the above architecture as is (ie. all layers from </b><code>conv1_1</code><b> to </b><code>pool5</code><b>, both inclusive, are used as they are presented above in the FCN implementation).</b> The fully connected layers are converted to convolutional layers (see next section on convolutionalization) and then used.
</p>

<!-- How to obtain the model weights. Talk about caffe-tensorflow -->
<p style="text-align: justify">
  Coming to the more practical aspect of obtaining and using the trained VGG weights. Luckily for us, the authors of VGG-Net have kindly made their trained models publicly availble on their group's webpage. They've written and trained their model in Caffe; which poses a minor hurdle for those of us using Tensorflow. I say minor because we only need to go an extra step of using caffe-tensorflow to convert the caffe weights into numpy binary format.
</p>

<!-- Talk about how training the VGG Net from scratch is not feasible for mere mortals -->
<p style="text-align: justify">
  I should note, at this point, that one might be tempted to not use the pretrained weights at all. I initially had the misconception that initializing with the VGG weights is not very important - that the network would learn the weights when I train it end-to-end. This, unfortunately, is not the case. Training the VGG-Net is not a single step process - it is trained in stages, and with increasingly difficult images. The pretrained VGG that we will take for granted in the remainder of this post is no small feat to achieve. It is critical that we have it in our FCN. I cannot harp on this enough: training a VGG from scratch is a sizeable task - one that took the authors weeks of training to complete - it is not to be underestimated.
</p>

<hr/>

<h4 id="section4">Convolutionalize Dense Layers</h4>

<!-- Why is it needed? What purpose does it solve? -->
<p style="text-align: justify">
  Here's the deal with this. The problem with dense layers is that they require the input to be of a fixed size. Images, as they go, can be of any size. That is, we may have diffferent sized images to train and perform inference on. Note that this issue doesn't present with convolutional layers. The input to a convolutional layer can be of any size and the weights of the layer would work and give output regardless. The VGG-16 architecture presented above has three fully connected (dense) layers at the very end (<code>fc1</code>, <code>fc2</code> and <code>fc3</code>) and we must take care of them. Now as the name goes, "fully convolutional networks" should only have convolutional layers. Hence, to accomodate the name, we must convert the dense layers to convolutional layers. There exists a beautiful way to go about this.
</p>
<!-- How to convert dense layers to conv layers -->
<p style="text-align: justify">
  Suppose we have a dense layer that requires $ HW $ input units and produces $ D $ outputs. Ie. the weight matrix has dimensions $ D \times HW $. This can be convolutionalized as follows: each row of the weight matrix can be reshaped into a filter of dimenstions $ H \times W $. Thus we end up with $ D $ filters of size $ H \times W $. Note that this convolutionalized layer can now take arbitrarily sized inputs. Also note that if the input to this layer is of size $ H \times W $, then it performs the same operation as the original dense layer. I find this quite intuitive and easy to understand, but perhaps a better explanation of this convolutionalization procedure is available in the notes of CS231n <a href="#" target="_blank">here</a>. In code, this reshaping of fully-connected weights of VGG-16 looks like:
</p>

``` python
fc1_convolutionalized_filters = numpy.reshape(vgg_params['fc1']['weights'], [7, 7, 512, 4096])
```

<p style="text-align: justify">
  How did I get the above line of code? Note that the original VGG-16 was trained for the ImageNet classification dataset - where each RGB image was of dimesions $ 224 \times 224 $. So, the input to <code>fc1</code> in the original VGG-16 would have been $ 7 \times 7 $. This is easy to work out from the following facts: the spatial resolution remains unchanged at convolutional layers on account of SAME padding, and the the spatial resolution reduces by factor of $ 2 $ at each pooling layer. There are five pooling layers between <code>conv1_1</code> and <code>fc1</code> and hence, the input to <code>fc1</code> would have the dimenstions $ 224/2^5 \times 224/2^5 $ or $ 7 \times 7 $. Also since the last conv layer before <code>fc1</code> (ie. <code>conv5_3</code>) has $ 512 $ output channels, that would also factor into the input of <code>fc1</code>. All in all, in the original VGG architcture, <code>fc1</code> saw an input of dimensions $ 7 \times 7 \times 512 $ and gave $ 4096 $ outputs. That is, the weight matrix of layer <code>fc1</code> had $ 4096 $ rows and $ 25088 $ columns. Following the convolutionalization logic presented in the last paragraph, each row (of $ 25088 $ weights) can be reshaped into a filter of size $ 7 \times 7 \times 512 $ and we would hence end up with $ 4096 $ such filters. In tensorflow, we are required to specfy convolution layer weights in the format <code>[filter_height, filter_width, filter_depth, output_channels]</code>. Hence we simply reshape the VGG weights into the shape [$ 7 $, $ 7 $, $ 512 $, $ 4096 $]. Note that the bias terms can be used as they are:
</p>

``` python
fc1_convolutionalized_biases = vgg_params['fc1']['biases']
```

<p style="text-align: justify">
  Finally, we can use these numpy arrays to create the convolutionalized <code>fc1</code> tensorflow graph node as follows:
</p>

``` python
fc1 = \
  tf.nn.relu(
    tf.nn.bias_add(
      tf.nn.conv2d(
        pool5,
        fc1_convolutionalized_filters,
        [1, 1, 1, 1],
        'SAME'),
      fc1_convolutionalized_biases
    )
  )
```

<p style="text-align: justify">
  Similarly for <code>fc2</code> and <code>fc3</code>,
</p>

``` python
fc2_convolutionalized_filters = numpy.reshape(vgg_params['fc2']['weights'], [1, 1, 4096, 4096])
fc2_convolutionalized_biases  = vgg_params['fc2']['biases']
fc2 = \
  tf.nn.relu(
    tf.nn.bias_add(
      tf.nn.conv2d(
        fc1,
        fc2_convolutionalized_filters,
        [1, 1, 1, 1],
        'SAME'),
      fc2_convolutionalized_biases
    )
  )
```

``` python
fc3_convolutionalized_filters = numpy.reshape(vgg_params['fc3']['weights'], [1, 1, 4096, 4096])
fc3_convolutionalized_biases  = vgg_params['fc3']['biases']
fc3 = \
  tf.nn.relu(
    tf.nn.bias_add(
      tf.nn.conv2d(
        fc2,
        fc3_convolutionalized_filters,
        [1, 1, 1, 1],
        'SAME'),
      fc3_convolutionalized_biases
    )
  )
```

<hr/>

<h4 id="section5">Score Layers and Color Maps</h4>

<!-- Explain what score layers are and why they're needed -->
<p style="text-align: justify">
  For the first time ever in this post, we present the complete FCN architecture:
</p>

<!-- FCN architecture image -->
<img src="{{ site.url }}/static/img/fcn/fcn.jpg"/>

<p style="text-align: justify">
  First off, note how the fully connected layers from the VGG architecture have now become convolution layers. Also now we see the addition of three new kinds of layers: score layers, upsample layers and addition layers. In this section we will tackle score layers.
</p>

<p style="text-align: justify">
  Score layers are nothing but convolutional layers having kernel dimensions $ 1 \times 1 $ and $ 21 $ output channels. My (intuitive) understanding of the nomenclature is as follows:
</p>

<p style="text-align: justify">
  Suppose the input volume to a score layer has dimensions $ H \times W \times F $. This is thought of as an image of dimensions $ H \times W $ and each pixel of the image is made up of $ F $ features. The score layer goes over each pixel, and at each pixel location predicts (on the basis of the $ F $ features at that pixel location) scores for each possible output class. In our case, there are a total of $ 21 $ output classes - $ 20 $ object classes and background. Hence, the score layer in our case goes over each pixel location and predicts $ 21 $ scores corresponding the output class at each location.
</p>

<!-- Explain color maps -->
<p style="text-align: justify">
  Note that the output of our score layers will have dimesions $ H \times W \times 21 $. At each pixel location, we can obtain a class prediction by simply picking the argmax of all $ 21 $ scores at that location. However, this doesn't lead us to a very visually appealing representation of the segmentation. We would like to associate each each output class with a color, and hence we can map our $ H \times W $ class predictions to an $ H \times W $ image (like the segmented cat image presented at the very beginning of this post). Ie. we need a color map.
</p>

<!-- Present MATLAB code for obtaining colors -->
<p style="text-align: justify">
  The MATLAB modules provided along with the PASCAL-VOC dataset have the following code:
</p>

``` matlab
% VOCLABELCOLORMAP Creates a label color map such that adjacent indices have different
% colors.  Useful for reading and writing index images which contain large indices,
% by encoding them as RGB images.
%
% CMAP = VOCLABELCOLORMAP(N) creates a label color map with N entries.
function cmap = VOClabelcolormap(N)

if nargin==0
    N=256;
end
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;
```

<p style="text-align: justify">
  For the MATLAB averse, <a target="_blank" href="https://github.com/wllhf">wllhf</a> has kindly provided <a target="_blank" href="https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae">this gist</a> which is a Python script that does the exact same thing as the above MATLAB code. That is, this provides a mapping between output classes and colors:
</p>

<!-- Image of the VOC color map -->
<img style="display: block; margin-left: auto; margin-right: auto; max-height: 60%; max-width: 60%; min-height: 0%; min-width:0%; width: auto" src="{{ site.url }}/static/img/fcn/colormap.png"/>

<hr/>

<h4 id="section6">Upsampling Layers</h4>

<p style="text-align: justify">
  The second new kind of layer we see in the FCN architecture are upsampling layers. Specifically, there are three of these: <code>upsample2</code> <code>upsample4</code> and <code>upsample32</code>. As one might expect from the name, upsampling layers increase the dimensions od the input that is provided to them by some factor. The way I've named each of my upsampling layers is: <code>upsampleXYZ</code>, where <code>XYZ</code> is the factor of upsampling wrt <b>the lowest dimesions that the input to the network is pooled to (Ie. the output of </b><code>pool5</code><b>).</b> So, if the output of <code>pool5</code> has spatial dimensions $ H \times W $, then the output of <code>upsample2</code> has dimensions $ 2H \times 2W $; the output of <code>upsample4</code> has dimesions $ 4H \times 4W $ and the output of <code>upsample32</code> has dimensions $ 32H \times 32W $.
</p>

<p style="text-align: justify">
  The upsampling layers are just convolutional layers of a special kind. If you don't understand how I can state that so directly, please refer <a href="https://arxiv.org/abs/1603.07285">chapter 4 of this document</a> on convolution arithmetic. The FCN paper tells us to initialize the filters of the upsample layer with bilinear interpolation filters. It will take me a complete post to explain resampling and bilinear interpolation - I would refer the reader to <a href="http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/" target="_blank">this post on Daniil Pakhomov's blog</a>. I strongly encourage reading the post atleast three times word for word - it contains extremely valuable theory and practical information about resampling, specifically bilinear interpolation. I have adapted the code preented in Daniil's blog post to generate the filters and create the upsampling layers:
</p>

<p>Method to generate the upsampling filters:</p>

``` python
def _get_upconv_params(factor, out_channels, name):
  kernel_sz = 2*factor - factor%2
  weights = np.zeros([kernel_sz, kernel_sz, out_channels, out_channels], dtype=np.float32)

  # Populate weights
  if kernel_sz % 2 == 1:
    center = factor - 1
  else:
    center = factor - 0.5
  tmp = np.ogrid[:kernel_sz, :kernel_sz]
  kernel = (1 - abs(tmp[0] - center)/factor) * (1 - abs(tmp[1] - center)/factor)
  for i in range(out_channels):
    weights[:, :, i, i] = kernel

  # Populate biases
  biases = np.zeros([out_channels,], dtype=np.float32)

  dic = {
    'weights': tf.Variable(weights, '{}_weights'.format(name)),
    'biases': tf.Variable(biases, '{}_biases'.format(name))
  }
  return dic
```

<p><code>upsample2</code> node in the Tensorflow computation graph:</p>

``` python
params['upsample2'] = _get_upconv_params(2, 21, 'upsample2')
net['upsample2'] = tf.nn.conv2d_transpose(
    net['score_fc'],
    params['upsample2']['weights'],
    output_shape = [1, tf.shape(net['score_fc'])[1] * 2, tf.shape(net['score_fc'])[2] * 2, 21],
    strides = [1,2,2,1]
  )
```

<hr/>

<h4 id="section7">Cropping and Skip Connections</h4>

<p style="text-align: justify">
  Finally, the last "new" kind of layer in the FCN architecture is the addition layer (also called a fuse layer or skip connection). This layer takes in two input volumes and adds them elementwise. Note that an important prerequisite of this addition is that the dimensions of the two input volumes must exactly match. Now, since the pool layers follow the 'SAME' scheme of max-pooling they reduce an input of dimensions $ H \times W $ to an output of dimensions $ \lfloor (H+1)/2 \rfloor \times \lfloor (W+1)/2 \rfloor $. However, the upsample layer increase the dimesions of an input ($ H \times W $) to $ 2H \times 2W $ (when the upsampling factor is 2). It is not too hard to show from these two observations that the output of <code>upsample2</code> is always greater than or equal to the output of <code>pool4</code> (equivalently, the output of <code>score_pool4</code>). In order for us to create a fuse layer with <code>score_pool4</code> and <code>upsample2</code> as inputs, we must first crop <code>upsample2</code> to an appropriate size. I have implemented this cropping as follows:
</p>

``` python
def _get_crop_layer(big_batch, small_batch):
  h_s = tf.shape(small_batch)[1]
  w_s = tf.shape(small_batch)[2]

  h_b = tf.shape(big_batch)[1]
  w_b = tf.shape(big_batch)[2]

  return big_batch[:,
                 (h_b - h_s)//2 : (h_b - h_s)//2 + h_s,
                 (w_b - w_s)//2 : (w_b - w_s)//2 + w_s,
                 :]
```

<p style="text-align: justify">
  Now, the fuse layer can be implemented as follows:
</p>

``` python
# Crop upsample2
net['cropped_upsample2'] = _get_crop_layer(net['upsample2'], net['pool4'])

# Score pool4
params['score_pool4'] = _get_score_layer_init_params('score_pool4')
net['score_pool4'] = _get_conv_layer(net['pool4'], params['score_pool4'])

# Fuse pool4
net['fuse_pool4'] = tf.add(net['score_pool4'], net['cropped_upsample2'])
```

<p style="text-align: justify">
  A few minor notes about the above code. The method to initialize score layer parameters (<code>_get_score_layer_init_params</code>) initializes filter weights using the Xavier initialization scheme and the biases to zero. Also it should be noted that for the crop layer implementation to work, the batch size <b>must be fixed to one</b>.
</p>

<hr/>

<h4 id="section8">Results</h4>

<p style="text-align: justify">
  Finally, the section that I've been holding my breath for! Without further ado, here they are:
</p>

{% for i in (1..10) %}
  <div class="row">
    <div class="column">
      <img src="{{ site.url }}/static/img/fcn/results/{{ i }}.png" style="display: block; margin-left: auto; margin-right: auto;"/>
    </div>
    <div class="column">
      <img src="{{ site.url }}/static/img/fcn/results/{{ i }}gt.png" style="display: block; margin-left: auto; margin-right: auto;"/>
    </div>
    <div class="column">
      <img src="{{ site.url }}/static/img/fcn/results/{{ i }}p.png" style="display: block; margin-left: auto; margin-right: auto;"/>
    </div>
  </div>
{% endfor %}

<hr/>

