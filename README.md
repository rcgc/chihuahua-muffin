# Chihuahua and muffin problem: very similar objects recognition
## Abstract
Image recognition and classification has been a complex problem to solve using tehcnology. Deep learning architectures such as Convolutional Neural Networks (CNN) have demonstrated to achieve a high performance accuracy in such tasks. In the present project will be demonstrated how transfer learning techniques using feature extraction and data augmentation tackle this kind of problems where complexity increases drastically, especially in situations that demand classificacion of very similar images belonging to completely different contexts.<br><br>

## Introduction
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

As we pointed out in the abstract, our objective is to classify very similar objects that belong to very different contexts. Due to the wide range of knwon (and unknown) examples we will focus on one of the most popular cases: The chihuahua-muffin problem.<br>

To tackle this kind of problems we will use a Convolutional Neural Network (CNN) architecture known as VGG19 by setting up its configuration parameters in a similar way as Ph.D Togootogtokh and Ph.D Amartuvshin did it in their paper "Deep Learning Approach for Very Similar Objects Recognition Application on Chihuahua and Muffin Problem"[2]. Therefore, this implementation has been inspired in the work of both professors, all credits to them, they proposed this state-of-the-art solution in 2018 which can be found in <a href="https://arxiv.org/abs/1801.09573">arxiv.org</a>.

## Materials and methods
### Neural networks
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

- A fixed size of (224 * 224 originally, but for this project will be 112 * 112) RGB image was given as input to this network which means that the matrix was of shape (224,224,3).
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

According to the type of problem to be solved, sparse_categorical_crossentropy is the option due to the fact that the output could belong to one of the two following classes: "Chihuahua" or "Muffin".<br><br>

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
<p><b>Figure 11. </b>How feature extraction[1]</p><br>

#### Data augmentation
This technique consists of getting more instances from an image dataset by doing transformations[1] such as:

- rotation
- zoom in / zoom out
- crop
- grayscale
- flip
<br>

![data_augmentation_example](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/data_augmentation_example.png)
<p><b>Figure 12. </b>Data augmentation example[1]</p><br>

### Dataset
All images used for this project belong to third party sources such as:
- <a href="https://storage.googleapis.com/openimages/web/index.html" target="_blank"> Open Images </a>
- <a href="https://www.robots.ox.ac.uk/~vgg/data/pets/" target="_blank"> Oxford's pets dataset </a>[9]
- <a href="https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz" target="_blank"> Imagenet dataset </a>[10]
- <a href="https://stock.adobe.com/mx/search?k=muffin" target="_blank"> adobe stock </a>
- <a href="https://www.istockphoto.com/es/search/2/image?istockcollection=main%2Cvalue&mediatype=photography&page=1&phrase=muffin" target="_blank"> istock photo </a>
- <a href="https://www.gettyimages.com.mx/fotos/muffin?assettype=image&page=1&phrase=muffin&sort=mostpopular&license=rf,rm" target="_blank"> getty images </a>
- <a href="https://www.pexels.com/search/muffin/" target="_blank"> pexels </a>
- <a href="https://unsplash.com/s/photos/muffin" target="_blank"> unsplash </a>

### How to use it
- `python setup.py` : creates chihuahua_vs_muffin folder which contains test, train and validation datasets
- `training.py` : trains the CNN, saves the model in vgg19_chihuahua_vs_muffin.h5, and queries the model

#### Querying the model
- Choose a folder: 
  - 2) Muffin folder
  - 1) Chihuahua folder
  - 0) Exit from program
 
- Enter a valid instance id 
  - 1-500 for muffin 
  - 1-900 for chihuahua
<br>

![Querying_model](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/querying_model.png)
<p><b>Figure 13. </b>Querying the model after 4 epochs</p><br>

NOTE #1: 2 extra graphs will appear before querying, that's because of the accuracy metrics shown in the following sections.<br><br>
NOTE #2: To run keras with GPU in Windows you will have to setup some configurations (running this program in CPU will be super slow), I would recommend to follow this tutorial: https://lifewithdata.com/2022/01/16/how-to-install-tensorflow-and-keras-with-gpu-support-on-windows/

## Results
### Training and validation accuracy
![Training_and_validation_accuracy](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/trainAcc_vs_valAcc.png)
<p><b>Figure 14. </b>Training and validation accuracy after 100 epochs</p><br>

### Training and validation loss
![Training_and_validation_loss](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/trainLoss_vs_valLoss.png)
<p><b>Figure 15. </b>Training and validation loss after 100 epochs</p><br>

### Testing accuracy and loss
![Test_accuracy_and_loss](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/testAcc_93.png)
<p><b>Figure 16. </b>Test accuracy and loss</p><br>

![Querying_chihuahua_232](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/querying_chihuahua_232.png)
<p><b>Figure 17. </b>Querying chihuahua_232 after 100 epochs</p><br>

![Querying_muffin_138](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/querying_muffin_138.png)
<p><b>Figure 18. </b>Querying muffin_138 after 100 epochs</p><br>

## Discusion
As we saw training accuracy and validation accuracy have reached around 93% and 97% respectively (sometimes this percentage is bigger) and test accuracy reached 94%. In addition, these metrics could vary depending on the combination of loss functions, new dataset instances, changing the number of neurons in the very last layers, etc.<br>

On the other hand, it has been a wise decision to use data augmentation to tackle this kind of problems because according to the papaer an accuracy of 95% has been reached as well, such percentage means that this very specific problem has been solved, this small project demonstrates the advantages in time and efficiency achieved by the implementation of transfer learning techniques. <br>

## Limitations
Only 1000 images were analyzed (500 chihuahuas and 500 muffins), it cannot be concluded that we will always get such accuracy percentages for this kind of problems: dataset instaces, computing resources, and so on, have a big influence in the final results.<br>

## References
[1]Chollet, F., 2022. Deep Learning With Python. 2nd ed. Greenwich, USA: Manning Publications.<br>
[2]S. Gettle, "CAMOUFLAGE IN NATURE - Steve Gettle Nature Photography", Steve Gettle Nature Photography, 2022. [Online]. Available: http://stevegettle.com/2008/10/08/camouflage-in-nature/. [Accessed: 19- May- 2022].<br>
[3]A. Gri, "Puppies Or Food? 12 Pics That Will Make You Question Reality", Bored Panda, 2022. [Online]. Available: https://www.boredpanda.com/dog-food-comparison-bagel-muffin-lookalike-teenybiscuit-karen-zack/?utm_source=google&utm_medium=organic&utm_campaign=organic. [Accessed: 19- May- 2022].<br>
[4]E. Togootogtokh and A. Amartuvshin, "Deep Learning Approach for Very Similar Objects Recognition Application on Chihuahua and Muffin Problem", arXiv, 2018. Available: https://arxiv.org/abs/1801.09573. [Accessed 19 May 2022].<br>
[5]"Neural Networks: Chapter 6 - Neural Architectures", Chronicles of AI, 2022. [Online]. Available: https://chroniclesofai.com/neural-networks-chapter-6-neural-architectures/. [Accessed: 20- May- 2022].<br>
[6]A. Kaushik, "Understanding the VGG19 Architecture", OpenGenus IQ: Computing Expertise & Legacy, 2022. [Online]. Available: https://iq.opengenus.org/vgg19-architecture/. [Accessed: 21- May- 2022].<br>
[7]Y. Zheng, C. Yang and A. Merkulov, "Breast cancer screening using convolutional neural network and follow-up digital mammography", Computational Imaging III, 2018. Available: 10.1117/12.2304564 [Accessed 21 May 2022].<br>
[8]K. Shah, "A Quick Overview to the Transfer Learning and it’s Significance in Real World Applications", Medium, 2022. [Online]. Available: https://medium.com/towards-tech-intelligence/a-quick-overview-to-the-transfer-learning-and-its-significance-in-real-world-applications-790fb57debad. [Accessed: 22- May- 2022].<br>
[9] Oxford Pet Animal Dataset. http://www.robots.ox.ac.uk/~vgg/data/pets/ <br>
[10] IMAGENET. http://www.image-net.org/
