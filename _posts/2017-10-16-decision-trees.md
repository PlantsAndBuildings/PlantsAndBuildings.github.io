---
layout: post
title: Decision Trees
published: true
categories: [machine-learning]
---

#### What are decision trees?

It will be beneficial to discuss decision trees with respect to a real-life example. The running example in this blog post will be that of the [Titanic Challenge](https://www.kaggle.com/c/titanic) from the popular website [Kaggle](https://www.kaggle.com). Basically, we're given a whole bunch of information about the passengers on board the famous Titanic - stuff like name, age, sex, number of children/spouse they were travelling with. And most importantly, we are given (for each of the passengers) whether they survived Titanic's sinking or not. We are then asked, given the same characteristics of some random passenger, to predict whether said passenger is likely to have survived the shipwreck or not. Capiche? The reader can familiarize himself with the challenge and the data [here](https://www.kaggle.com/c/titanic).

Now that we have given an overview of the problem that we will attempt to solve, let us understand our strategy (atleast for this blog post) to attack it - decision trees. Decision trees are fairly straightforward things to understand:

- Each leaf node represents a decision or a classification.
- Each internal node represents an attribute or feature to be tested.
- Based on the feature to be tested, there are branches descending from each internal node.

The following figure (taken from [here](https://medium.com/towards-data-science/decision-trees-in-machine-learning-641b9c4e8052)) represents one possible decision tree which uses three features/attributes (namely sex, age and sibsp) to predict the survival classification (Ie. died or survived).

<img src="{{ site.url }}/static/img/titanic-decision-tree-sample.png"/>

It is not too hard to see how we perform classification from this tree. We simply start from the root with our sample point (for which we have to perform the classification) and follow a path down to one of the leaves. Which path we take is dictated by the features of out sample point. The next logical question arises - how do we build decision trees? That is, how do we choose the features for internal nodes such that we obtain a "good" classification for unseen data points. We address this next.

#### How to grow decision trees?

First, let us understand what is meant by growing an optimal decision tree to fit some given training data. We have already seen that a decision tree contains, as internal nodes, features which split data into "buckets". Consider now, all possible permutations of these internal nodes - that is, try and imagine all possible trees that we can construct (by choosing different features at root, children of root and so on). It should intuitively be clear that the number of such decision trees is large - and grows exponentially with the number of features. Of all such possible decision trees (if at all we are able to enumerate all of them), if we select the one that has the highest classification accuracy on the training data - then that decision tree is optimal for the given training data. At this point we make two statements:

- The task of finding an optimal decision tree to fit some given training data is NP complete. I say this without proof - trust me.
- There is absolutely no guarantee that the optimal decision tree for some given training data will generalize the best. Ie. there are no gurantees that it will perform best on unseen data (because unseen data is, well, unseen). However, we can expect it to perform better than most randomly formed decision trees.

The above two statements motivate our decision tree learning algorithm. What follows is a very general description of an algorithm to learn decision trees, and most popular algorithms (CART, ID3, C4.5 etc) are just slight variations of the following steps:

1. We start from the top (Ie. the root). We must first decide the attribute to be tested at the root of the decision tree. How do we select this attribute? This is where different learning algorithms differ from one another. In ID3 we compute the **information gain** obtained by using each attribute to split the data. The attribute that provides the highest information gain is selected for the root.
2. Now we move down - to the children of the root. Here too, we use information gain to select the "best" attribute to split the data. Note that here, we do not consider the attributes that have already been assigned to a higher node. This ensures termination of the algorithm. We terminate when there are no more attributes left to split on **OR** when we reach a state where all the training examples at the current node belong to the same class.

#### Code using Scikit Learn for Titanic Challenge (Kaggle)

I have set up my project directory as follows:

```
- d_tree.py
- dataset/
|- train.csv
|- test.csv
|- local_test.csv
|- local_train.csv
```

```train.csv``` and ```test.csv``` are provided on the [challenge page](https://www.kaggle.com/c/titanic/data). I have split the data from ```train.csv``` into ```local_train.csv``` and ```local_test.csv``` to test and get an idea of the expected accuracy on the actual test data before submission.

I have used:

- Pandas for data manipulation.
- Scikit-Learn for actually building the decision tree model.
- Graphviz for visualizing the decision tree.

Necessary imports are as follows:

``` python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import csv
import graphviz
import subprocess
```

I define a global boolean that allows me to switch between local testing mode and final submission mode. When I'm testing locally, I want to learn the decision tree on ```local_train.csv``` and compute the classification acccuracy on ```local_test.csv```. In the final submission mode, I want to train on the entire ```train.csv```, and generate an output csv file, using ```test.csv```, as per the requirements of the challenge.

``` python
LOCAL_TESTING = True
```

Moving on to actually building the model. We are given 12 features, out of which only 7 (according to me) are of any real use:

- Sex
- Pclass
- Age
- SibSp
- Parch
- Fare
- Embarked

These are stored in a list ```features_to_consider```.

``` python
features_to_consider = ['Sex','Pclass','Age','SibSp','Parch','Fare','Embarked']
```

The following method, reads a csv file - path to which is specified as the ```filename``` argument. It also performs some cleaning up:

- It fills in missing values in the dataset. This procedure should be afforded closer scrutiny - and some statistics would go a long way. For now, I have just put in values that seemed reasonable to me. This should have an effect on the accuracy of the model.
- It also converts string values in the data to numeric data. This is required by the ```DecisionTreeClassifier``` of scikit-learn. I have just assigned an integer to each distinct value of a string type attribute. An alternate is using a one-hot encoding scheme (maybe that performs better, I dont know).

Finally, this method checks if the given file is a training file or a test file (depending on whether the data contains the ```Survived``` column or not). If it does, then the method returns a list of two Pandas ```DataFrames``` - namely, one containing the data points (```X```) and the other containing corresponding classes (```y```). If the ```Survived``` column is absent (that is, we are looking to create the final output file and submit) - then only ```X``` is created and returned.

``` python
def get_dataset(filename):
	X = pd.read_csv(filename)
	X.index = X['PassengerId']

	# Fill in missing values
	X['Sex'].fillna('male', inplace=True)
	X['Pclass'].fillna(1, inplace=True)
	X['Age'].fillna(30, inplace=True)
	X['SibSp'].fillna(2, inplace=True)
	X['Parch'].fillna(0, inplace=True)
	X['Fare'].fillna(25, inplace=True)
	X['Embarked'].fillna('S', inplace=True)

	# Convert string type values to numeric types
	X['Sex'].replace(to_replace='male',value=1,inplace=True)
	X['Sex'].replace(to_replace='female',value=0,inplace=True)
	X['Embarked'].replace(to_replace='S',value=0,inplace=True)
	X['Embarked'].replace(to_replace='C',value=1,inplace=True)
	X['Embarked'].replace(to_replace='Q',value=2,inplace=True)

	if 'Survived' in X.columns:
		y = X['Survived']
		return [X,y]
	else:
		return [X]
```

Get the training data (from different files depending on whether we are tesing locally, or are looking to submit).

``` python
if LOCAL_TESTING:
	X,y = get_dataset('dataset/local_train.csv')
else:
	X,y = get_dataset('dataset/train.csv')
```

Now, we get the ```DecisionTreeClassifier``` object and fit it to our training data. We have specified some keyword arguments while creating the decision tree object. It is important that we understand what each of these is responsible for.

- ```criterion```. This keyword argument specifies the criterion used for splitting nodes of the decision tree. Here we have set it to ```entropy```. By default, it uses the ```gini``` criterion.
- ```splitter```. Now that we have specified the criterion used to evaluate splits, we need to specify a strategy used to split the nodes. Supported strategies are ```best``` and ```random```.
- ```max_depth```. Specifies the maximum depth of the decision tree. This parameter affects the accuracy a LOT.
- ```min_samples_leaf``` and ```min_samples_split```. These parameters are used to prevent overfitting of the decision tree. If the leaf nodes contain only one sample each - then it is unlikely that the model will generalize well to unseen data. Here we have set the minimum number of samples in a leaf to be 5. This parameter should also be varied and checked for best performance.

The reader should go through [this](http://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use) in order to learn some tricks about training decision trees using scikit-learn. Also, a complete list of all keyword arguments to the ```DecisionTreeClassifer``` constructor is given [here](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier).

``` python
clf = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=4,min_samples_leaf=5)
clf.fit(X[features_to_consider].as_matrix(),y.as_matrix())
```

Again, we get the dataset (from different files on the basis of whether we are testing locally or submitting). Next, we get predictions from our model and:

- In the local testing mode, we compute the accuracy and create a visualization of the decision tree.
- In the final submission mode, we generate the output file (```out.csv```) for submission.

```python
if LOCAL_TESTING:
	X,y = get_dataset('dataset/local_test.csv')
	predictions = clf.predict(X[features_to_consider].as_matrix())
	total_examples = len(predictions)
	correct_predictions = 0
	for prediction, actual_class in zip(predictions,y.as_matrix()):
		if prediction == actual_class:
			correct_predictions += 1
	print "Accuracy: ",
	print (1.0*correct_predictions)/(1.0*total_examples)
	visualize_tree(clf)
else:
	X = get_dataset('dataset/test.csv')[0]
	passenger_ids = list(X['PassengerId'].as_matrix())
	predictions = zip(passenger_ids, clf.predict(X[features_to_consider].as_matrix()))
	with open('out.csv','w') as outfile:
		out_writer = csv.writer(outfile)
		out_writer.writerow(["PassengerId","Survived"])
		for (passenger_id, prediction) in predictions:
			out_writer.writerow([passenger_id,prediction])
```

We make a short note of the method used to visualize the decision tree. The ```sklearn.tree``` module contains an ```export_graphviz``` method that exports a ```DecisionTreeClassifier``` object as a dot file. This dot file, can be converted to either a postscript or PNG format using the ```dot``` utility. This is precisely what ```visualize_tree``` achieves.

``` python
def visualize_tree(clf):
	dotfile = 'tree.dot'
	pngfile = 'tree.png'
	export_graphviz(clf,out_file=dotfile)
	command = ['dot','-Tpng',dotfile,'-o',pngfile]
	subprocess.check_call(command)
```

And finally, here what our decision tree looks like.

<img src="{{ site.url }}/static/img/titanic-decision-tree-graphviz.png"/>

The complete code, dataset and visualization can be found on [my github](#). This code, as is, achieves a 0.78 accuracy on Kaggle - which is decent, but there is a lot of scope for improvement. Right now this code only performs a minimal amount of feature engineering; doesn't add any new features and has not been tested for all possible values of parameters like ```max_depth```, ```min_samples_split``` etc.

Thats all.

#### References

- [Decision Trees in Machine Learning (Medium)](https://medium.com/towards-data-science/decision-trees-in-machine-learning-641b9c4e8052)
- Machine Learning. Book by Tom Mitchell
- [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Scikit-Learn User Guide for Decision Trees](http://scikit-learn.org/stable/modules/tree.html)
