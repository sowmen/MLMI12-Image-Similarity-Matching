# Image Similarity Matching

MLMI12 - Computer Vision Mini Project. (<a href="report\CV_Mini_Project_Report.pdf"> Report </a>)

Training a neural network to perform image similarity matching on the Tiny Imagenet dataset. Given two images identify whether they belong to the same class or different classes.

<img src="images\intro.png">

## Methods

#### Multiclass Classification

Classify each image separately and check whether the classes match.
<img src="images\net1.png">

#### Multiclass Features with Cosine Similarity
Instead of getting the final class from the network, take the feature embeddings and calculate the cosine similarity between them. If the similarity is above a threshold, then the images are considered to be similar.
<img src="images/net2.png">

#### Siamese Network with Triplet Loss
Train a siamese network with triplet loss. The network takes in two images and outputs the feature embeddings. The triplet loss is calculated between the anchor, positive and negative images. 

<img src="images/net3.png">

#### End-to-End Binary Classifier
Train a network end-to-end to provide binary classes for either similar or dissimal. 

<img src="images/net4.png">


## Results

Seen-Seen : Both image classes were seen during training
Seen-Unseen : One image class was seen during training and the other was unseen
Unseen-Unseen : Both image classes were unseen during training
<img src="images/results.png">