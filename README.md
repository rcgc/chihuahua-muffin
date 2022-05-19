# Chihuahua and muffin problem: very similar objects recognition
## Abstract
Since the 2010’s-decade deep learning field progressed taking giant steps at always expected tasks such as object classification, speech recognition, text-written processing, image generation, etc. AI competitions such as the “Imagenet challenge” led to convolutional architectures became so popular when solving object recognition and classification problems, this is because of the accuracy levels reached, from around 70% in 2011 to more than 95% (better than humans even!) in 2015[1].<br><br>

![Camouflaged_owl](https://github.com/rcgc/chihuahua-muffin/blob/master/readme_images/camouflaged_owl.jpg)
<p><b>Figure 1. </b>Camouflaged owl[2]</p><br>

Very similar objects recognition is not a new task for humans at all, because in nature it happens all the time: we can observe how animals use camouflage to survive, hid from preys to hunt them, and so on. However, during this report, we won’t study camouflage issues, but objects that look alike in completely different contexts: labradoodles and fried chicken, dogs and bagels, sheepdogs and mops and chihuahuas and muffins of course!<br>

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

To tackle this kind of problems we will use a Convolutional Neural Network architecture known as VGG19 by setting up its configuration parameters in a similar way as Ph.D Togootogtokh and Ph.D Amartuvshin did it in their paper "Deep Learning Approach for Very Similar Objects Recognition Application on Chihuahua and Muffin Problem"[2]. Therefore, this implementation has been inspired in the work of Ph.D Togootogtokh and Ph.D Amartuvshin, all credits to their work that you can find in arxiv.org.

## Materials and methods
## Results
## Duscusion
## Limitations
## References
[1]Chollet, F., 2022. Deep Learning With Python. 2nd ed. Greenwich, USA: Manning Publications.<br>
[2]S. Gettle, "CAMOUFLAGE IN NATURE - Steve Gettle Nature Photography", Steve Gettle Nature Photography, 2022. [Online]. Available: http://stevegettle.com/2008/10/08/camouflage-in-nature/. [Accessed: 19- May- 2022].<br>
[3]A. Gri, "Puppies Or Food? 12 Pics That Will Make You Question Reality", Bored Panda, 2022. [Online]. Available: https://www.boredpanda.com/dog-food-comparison-bagel-muffin-lookalike-teenybiscuit-karen-zack/?utm_source=google&utm_medium=organic&utm_campaign=organic. [Accessed: 19- May- 2022].<br>
[4]E. Togootogtokh and A. Amartuvshin, "Deep Learning Approach for Very Similar Objects Recognition Application on Chihuahua and Muffin Problem", arXiv, 2018. Available: https://arxiv.org/abs/1801.09573. [Accessed 19 May 2022].
