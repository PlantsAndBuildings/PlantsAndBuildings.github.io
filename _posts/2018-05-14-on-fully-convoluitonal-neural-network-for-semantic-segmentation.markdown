---
title: On Fully Convolutional Neural Networks for Semantic Segmentation
layout: post
categories: [machine-learning, misc, math]

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
  Guess who's at work again and has nothing better to do than stare at ceilings and write blog-posts that noone reads? I do have a nice Wodehouse novel in my bag that I'm dying to get at - but I guess my boss would rather see my nose stuck in a code screen than a Jeeves omnibus... I bet Jeeves would've had a solution to my conundrum. Anyway, here's the dealio.
</p>

<p style="text-align: justify;">
  I've been reading a bunch of papers and articles and such on object detection and semantic sengmentation - learning about the standard deep learning models used for these tasks. The segmentation task has particularly caputured my fancy, mainly because the results look really cool:
</p>

<!-- INSERT SEMANTIC SEGMENTATION RESULT HERE -->
  
<p style="text-align: justify;">
  One model - called the Fully Convolutional Network - has found mention all over the place in segmentation literature, and doesn't look overly complicated. Consequently, I've chosen to implement its simplest variant - FCN-8s. It's really simple to understand really, there are only three salient points that one needs to remember:
</p>

<ol>
  <li style="text-align: justify;">The initial layers - about 7 of them - are the same as a VGG net. If you've never seen a VGG-Net architecture, then it would suffice to know that: a. it is used for image classfication; and b. it is about 11-16 layers deep and has only 3x3 convolutions and max poolings followed by three fully-connected layers. FCN uses the weights from a trained VGG-net for its initial layers (sort of a transfer learning thing that goes on here)</li>
  <li style="text-align: justify;">The fully connected layers of the VGG-net are replaced by 1x1 convolutions and there is an upsampling step at the very end of the network which resizes the feature map to the original input image size.</li>
  <li style="text-align: justify;">The upsampling at the very end is aided by max pooling outputs from earier layers in the network - via "skip connections". More on this later, when we get to the implementation.</li>
</ol>
