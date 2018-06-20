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
  }
});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<hr/>

<p style="text-align: justify">
  <b>TL,DR: This blog post presents my implementation of the <a href="https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf" target="_blank">FCN paper</a>. Code for this can be found on <a href="http://github.com/plantsandbuildings/FCN" target="_blank">my Github</a>.</b>
</p>

<hr/>

<p style="text-align: justify">
  <b>Disclaimer: What follows is not a tutorial, they're my implementation notes. If you're looking for material on what FCNs are and such - then you probably don't stand to gain much from reading on. For that, I suggest going through the paper linked above. What follows is probably most useful to people who are either planning to or are in the midst of implementing an FCN themselves. My aim here is to talk about details that I only found after scouring the depths of the Internet for several torturous hours.</b>
</p>

<hr/>

<h4>What is Semantic Segmentation?</h4>
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

<h4>How do we do it?</h4>
