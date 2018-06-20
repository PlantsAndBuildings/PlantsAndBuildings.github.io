---

title: Toy Fully Convolutional Network for Semantic Segmentation
layout: post
published: false

---


<p style="text-align=justify;">Okay, so I've been trying to write an FCN8s for about two weeks now and I'm finally at a stage where I think I've forseen all possible things that could go wrong. However, just to make sure that what I'm going to do is indeed going to work, I've decided to first write a smaller network (for those familiar with FCN-8s, if is a 21 layer net). So, I'm going to adapt (rather "convolutionalize") the LeNet architecture  as per the method defined in the FCN paper and train the network for the semantic segmentation task.</p>

<p style="text-align:justify;">Also, It seems like a fun idea to me to time this thing. So, in true hacker fashion, all updates to this blogpost will be in real time. Here goes nothing.</p>

<hr/>
<p style="text-align:justify"><b>16:34 Saturday.</b> It begins.</p>

<p style="text-align:justify">I've prepared myself: Untucked the formal shirt I was wearing to get confortable; raised my office chair to just the right height for the typing to be optimal; Slayer's Raining Blood has been set to repeat on a loop and is blasting full volume in my ears; I've consumed a fair amount of water and I don't feel like I'm going to pee for the next 4-5 hours. The only thing is that I'm slightly drowsy from the really heavy lunch I just had. I guess it will pass as I start working.</p>

<p style="text-align:justify">First things first, I need to set up an environment and install the stuffs I'll need. I'll also need to set up a git repository for this project.</p>

<hr/>

<p style="text-align:justify;"><b>17:02 Saturday.</b> Preliminaries.</p>

<p style="text-align:justify;">I've created a new conda environmnet for this project; and installed tensorflow, jupyter, matplotlib and opencv inside it. I like to create conda envs and use pip install to install packages inside them (as opposed to conda install). This is because, I can easily create requirements files using pip freeze later on (something which conda doesn't allow AFAIK).</p>

<p style="text-align:justify;">I've also initialized an empty git repository, but haven't created a remote just yet - I'll wait to push some commits before I do that. Next up, we start on the utils to read the dataset.</p>

<hr/>

<p style="text-align:justify;"><b>Around 19:00 Saturday.</b> Read Dataset.</p>

<p style="text-align:justify;">Yeah, so this entry is actually being written on Monday. I've just completed the module that will read the dataset batch by batch - although most of it was already complete on Satruday. I didn't really feel like working on this yesterday, so I kinda slacked off - but I'm back at it today.</p>

<p style="text-align:justify;">A small note about my dataset batch reader which I think is pretty cool. The way I've implemented the read batch method is:</p>

```python
def next_train_batch(self, batch_sz = 10):
  '''Returns a batch sampled from the training data.

  Performs random uniform sampling from the data. Returns a dict of the form:
  {
    'batch': iterator object
    'annotations': iterator object
  }

  Arguments:
  batch_sz: Size of the batch to be sampled. Default value is 10.
  '''
  files_in_batch = random.sample(self.segmentation_imageset_train_files,\
                                 batch_sz)
  self.logger.debug('Batch selected: {}'.format(files_in_batch))

  data_files_in_batch = [os.path.join(self.train_root_dir,\
      'VOCdevkit/VOC2011/JPEGImages',\
      '{}.jpg'.format(x))\
      for x in files_in_batch]

  batch = map(self._read_image, data_files_in_batch)

  annotation_files_in_batch = [os.path.join(self.train_root_dir,\
      'VOCdevkit/VOC2011/SegmentationClass',\
      '{}.png'.format(x))\
      for x in files_in_batch]

  annotations = map(self._read_annotation_file, annotation_files_in_batch)

  return { 'batch': batch, 'annotations': annotations }
```

<p style="text-align:justify;">See how I'm returning a map objects instead of lists or numpy arrays? Isn't it beautiful? Let me start at the beginning. The complete dataset sizes up to approximately 1.5 GB (just the train set). It is obviously a very bad idea to load all of it into memory. Since we're going to be training our network in batches, we only really need to pull one batch into memory at a time. The dataset creators have kindly provided a file that contains the names of all the image files in the dataset (this, I will call as the "index file"). I've read the index file into a list (let us call this the "index list"). Every time the above method is called (Ie. a new batch is required), I sample a certain number of entries from the index list - which represent the new batch. I then read the image file corresponding to each entry and form a new batch. But here's the cool part:</p>

<p style="text-align:justify;">In Python 2.x, a call to map function returned a list. So, if I were using Python 2.x I would've been reading an entire batch of images into memory for every call to the above method. However, I'm using Python 3.6; where the map function returns an iterator object insted of a list. As a result, I'm only returning an iterator object with each call to the above function. What this means is that the images won't acutally be read into memory till the time they are required. That is, till the time that something like:</p>

```python
ddic = voc_reader.next_train_batch()
batch = ddic['batch']
for img in batch:
  # Image is read into memory at this point.
  process(img)
```

<p>happens.</p>

<p style="text-align:justify;">With that nugget of Pythonic amazement out of the way, I think I'm ready now to begin with the convnet implementation.</p>

<hr/>

<p style="text-align:justify;"><b>15:12 Monday.</b> Convnet Implementation begins.</p>
