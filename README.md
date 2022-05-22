# Chihuahua and muffin problem: very similar objects recognition
## Abstract
Since the 2010’s-decade deep learning field progressed taking giant steps at always expected tasks such as object classification, speech recognition, text-written processing, image generation, etc. AI competitions such as the “Imagenet challenge” led to convolutional architectures became so popular when solving object recognition and classification problems, this is because of the accuracy levels reached, from around 70% in 2011 to more than 95% (better than humans even!) in 2015[1].<br><br>

![Camouflaged_owl](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/camouflaged_owl.jpg)
<p><b>Figure 1. </b>Camouflaged owl[2]</p><br>

Very similar objects recognition is not a new task for humans at all, because in nature it happens all the time: we can observe how animals use camouflage to survive, hid from preys to hunt them, and so on. However, during this report, we won’t study camouflage issues, but objects that look alike in completely different contexts: labradoodles and fried chicken, dogs and bagels, sheepdogs and mops and chihuahuas and muffins of course!<br><br>

![Labradoodle_vs_friedChicken](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/labradoodle_vs_friedChicken.jpg)
<p><b>Figure 2. </b>Labradoodle vs fried chicken[2]</p><br><br>

![Dog_vs_bagel](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/dog_vs_bagel.jpg)
<p><b>Figure 3. </b>Dog vs bagel[2]</p><br><br>

![Sheepdog_vs_Mop](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/sheepdog_vs_mop.jpg)
<p><b>Figure 4. </b>Sheepdog vs mop[2]</p><br><br>

![Chihuahua_vs_muffin](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/chihuahua_vs_muffin.jpg)
<p><b>Figure 5. </b>Chihuahua vs muffin[2]</p><br>

## Introduction
As we pointed out in the abstract, our objective is to classify very similar objects that belong to very different contexts. Due to the wide range of knwon (and unknown) examples we will focus on one of the most popular cases: The chihuahua-muffin problem.<br>

To tackle this kind of problems we will use a Convolutional Neural Network (CNN) architecture known as VGG19 by setting up its configuration parameters in a similar way as Ph.D Togootogtokh and Ph.D Amartuvshin did it in their paper "Deep Learning Approach for Very Similar Objects Recognition Application on Chihuahua and Muffin Problem"[2]. Therefore, this implementation has been inspired in the work of both professors, all credits to them, they proposed this state-of-the-art solution in 2018 which can be found in <a href="https://arxiv.org/abs/1801.09573">arxiv.org</a>.

## Materials and methods
## Neural networks
Deep learning is a specific subfield of machine learning: a new take on learning representations from data that puts an emphasis on learning successive layers of increasingly meaningful representations. In deep learning, these layered representations are learned via models called neural networks, structured in literal layers stacked on top of each other.<br><br>

Nowadays, there are numerous neural network architectures aimed at different purposes. However, as mentioned in the abstract we will focus only in Convolutional Neural Networks (Deep Convolutional Networks in the diagram) because of the effectiveness achieved in classification tasks.<br><br>

![nn_architectures](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/nn_architectures.jpeg)
<p><b>Figure 6. </b>Neural Networks Architectures[5]</p><br>

### Convolutional Neural Networks
A Convolutional Neural Network (CNN) characterizes by convolution layers which learn local patterns—in the case of images, patterns found in small 2D windows of the inputs. Thus, convolutions work by sliding these windows of size 3 × 3 or 5 × 5 over the 3D input feature map, stopping at every possible location, and extracting the 3D patch of surrounding features. Where feature map can be understood as follows: every dimension in the depth axis is a feature. <br><br>

![how_convolution_works](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/how_convolution_works.png)
<p><b>Figure 7. </b>How convolution works[1]</p><br>

![cnn_diagram](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/CNN_diagram.png)
<p><b>Figure 8. </b>CNN diagram example, VGG16 Architecture[1]</p><br>

#### VGG19
Created by the Visual Geometry Group at Oxford's this architecture uses some ideas from it's predecessors (AlexNet) and improves them in a significant way that, in 2014 it out-shined other state of the art models and is still preferred for a lot of challenging problems[6].<br>

VGG19 is a variant of VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layers, 5 MaxPool layers and 1 SoftMax layer). There are other variants of VGG like VGG11, VGG16 and others. VGG19 has 19.6 billion Floating Operations (FLOPs). The main purpose for which VGG was designed was to win ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)[6].<br><br>

![vgg19_architecture](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/vgg19_example.png)
<p><b>Figure 9. </b>VGG19 Architecture[7]</p><br>

Brief explanation of how the VGG19 architecture works:<br>

- A fixed size of (224 * 224) RGB image was given as input to this network which means that the matrix was of shape (224,224,3).
- The only preprocessing that was done is that they subtracted the mean RGB value from each pixel, computed over the whole training set.
- Used kernels of (3 * 3) size with a stride size of 1 pixel, this enabled them to cover the whole notion of the image.
- Spatial padding was used to preserve the spatial resolution of the image.
- Max pooling was performed over a 2 * 2 pixel windows with sride 2.
- This was followed by Rectified linear unit(ReLu) to introduce non-linearity to make the model classify better and to improve computational time as the previous models used tanh or sigmoid functions this proved much better than those.
- Implemented three fully connected layers from which first two were of size 4096 and after that a layer with 1000 channels for 1000-way ILSVRC classification and the final layer is a softmax function[6].<br><br>

#### Loss functions
Takes the predictions of the network and the true target (what you wanted the network to output) and computes a distance score, capturing how well the network has done on a specific example. These are the commond used loss functions:<br>

- CategoricalCrossentropy
- SparseCategoricalCrossentropy
- BinaryCrossentropy
- MeanSquaredError
- KLDivergence
- CosineSimilarity

According to the type of problem to be solved, binary_crossentropy is the option due to the fact that only two classes will be predicted "Chihuahua" or "Muffin".<br><br>

#### Optimizers
Determines how the network will be updated based on the loss function. It implements a specific variant of stochastic gradient descent (SGD). These are the common used optimizers:<br>

- SGD (with o without momentum)
- RMSprop
- Adam
- Adagrad
<br>

### Transfer Learning
Focuses on the knowledge gained from previous Machine Learning systems which will be used for another one to learn how to solve similar tasks that will include partially o completely different data[8].<br>

![transfer_learning](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/transfer_learning.png)
<p><b>Figure 10. </b>Transfer learning diagram[8]</p><br>

#### Feature extraction
Feature extraction consists of using the representations learned by a previously trained model to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch[1].<br><br>

![feature_extraction](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/feature_Extraction.png)
<p><b>Figure 11. </b>Feature extraction[1]</p><br>

#### Data augmentation
This technique consists of getting more instences from an image dataset by doing transformations[1] such as:

- rotation
- zoom in / zoom out
- crop
- grayscale
- flip
<br>

### Dataset
All images used for this project belong to third party sources such as:
- <a href="https://storage.googleapis.com/openimages/web/index.html"> Open Images </a>
- <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/"> Oxford's pets dataset </a>
- <a href="https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz"> Imagenet dataset </a>
- <a href="https://stock.adobe.com/mx/search?k=muffin"> adobe stock </a>
- <a href="https://www.istockphoto.com/es/search/2/image?istockcollection=main%2Cvalue&mediatype=photography&page=1&phrase=muffin"> istock photo </a>
- <a href="https://www.gettyimages.com.mx/fotos/muffin?assettype=image&page=1&phrase=muffin&sort=mostpopular&license=rf,rm"> getty images </a>
- <a href="https://www.pexels.com/search/muffin/"> pexels </a>
- <a href="https://unsplash.com/s/photos/muffin"> unsplash </a>

## Results

## Discusion

## Limitations

## References
[1]Chollet, F., 2022. Deep Learning With Python. 2nd ed. Greenwich, USA: Manning Publications.<br>
[2]S. Gettle, "CAMOUFLAGE IN NATURE - Steve Gettle Nature Photography", Steve Gettle Nature Photography, 2022. [Online]. Available: http://stevegettle.com/2008/10/08/camouflage-in-nature/. [Accessed: 19- May- 2022].<br>
[3]A. Gri, "Puppies Or Food? 12 Pics That Will Make You Question Reality", Bored Panda, 2022. [Online]. Available: https://www.boredpanda.com/dog-food-comparison-bagel-muffin-lookalike-teenybiscuit-karen-zack/?utm_source=google&utm_medium=organic&utm_campaign=organic. [Accessed: 19- May- 2022].<br>
[4]E. Togootogtokh and A. Amartuvshin, "Deep Learning Approach for Very Similar Objects Recognition Application on Chihuahua and Muffin Problem", arXiv, 2018. Available: https://arxiv.org/abs/1801.09573. [Accessed 19 May 2022].<br>
[5]"Neural Networks: Chapter 6 - Neural Architectures", Chronicles of AI, 2022. [Online]. Available: https://chroniclesofai.com/neural-networks-chapter-6-neural-architectures/. [Accessed: 20- May- 2022].<br>
[6]A. Kaushik, "Understanding the VGG19 Architecture", OpenGenus IQ: Computing Expertise & Legacy, 2022. [Online]. Available: https://iq.opengenus.org/vgg19-architecture/. [Accessed: 21- May- 2022].<br>
[7]Y. Zheng, C. Yang and A. Merkulov, "Breast cancer screening using convolutional neural network and follow-up digital mammography", Computational Imaging III, 2018. Available: 10.1117/12.2304564 [Accessed 21 May 2022].
[8]K. Shah, "A Quick Overview to the Transfer Learning and it’s Significance in Real World Applications", Medium, 2022. [Online]. Available: https://medium.com/towards-tech-intelligence/a-quick-overview-to-the-transfer-learning-and-its-significance-in-real-world-applications-790fb57debad. [Accessed: 22- May- 2022].
