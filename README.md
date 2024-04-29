## Creating a Neural Network to Detect Facial Features

In this project, pytorch is used to create and implement a resnet which is then trained on the CelebA database (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to detect features such as gender, hair color, presence of a beard and whether the person is smiling. Because each image can have multiple labels at the same time, this is an example of a multi-label classification problem. 

This project was created as a fun way to explore how to create a basic neural network in pytorch, load in data to a custom dataset and then train that network and use it to crate predictions. It is by no means perfectly optimized, and certain pitfalls (eg. imbalanced dataset) have been left in deliberately to showcase their effect on the model's performance.

This project is also heavily commented and explained such that it is beginner friendly, and includes details on over-fitting, imbalanced datasets and abstract labels, showing how the latter affects training results.

# File Structure and how to install

The file structure for this project consists of three main files/folders: 

- celeb_dataset is a local download of the CelebA dataset. Inside it contains a few csv. files, the most important of which is 'list_attr_celeba.csv' which contains all the labels for the dataset, and is used to create a custom dataset inside the jupyter notebook. The actual images are contained within the 'img_align_celeba\Celebs' folder. NOTE: you will need to download this yourself from (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and place the images in this file if planning to train the NN yourself.

- test_images is a folder containing additional images (potentially of family or friends!) which the model is tested on after training. Feel free to add more images in here in either png or jpg form.

- Facial Detection CNN is the main interactive notebook which downloads the dataset, sets up the model, trains it and then tests it. This is done in jupyter notebook.

In order to run the code yourself, I would recommend setting up an Anaconda environment which you can do with the following (https://www.anaconda.com/):

```
conda create -n name-of-environment python=3.10
conda activate name-of-environment
```
You will then want to install various packages such as torchvision, jupyter notebook, pandas, tqdm and pytorch. You can do this with pip install (or conda install), such as below ~ make sure you are doing this with your anaconda environment active (follow this guide for pip install of pytorch which is right for you: https://pytorch.org/get-started/locally/):

```
pip install notebook
pip install pandas
pip install tqdm
```
Once all the packages are installed (make sure you have everything listed under the first codeblock ~ useful libraries ~ installed), you can go ahead and explore the notebook, running each codeblock one at a time and making changes as you wish.

# Model Used

The final model used in this project is very similar to the ResNet 15 (https://www.hindawi.com/journals/amete/2020/6972826/), consisting of a four total skip connections and added dropout layers in the classification stage to reduce overfitting. Other models were tried, where simpler models were plateau'ed at a lower accuracy and more complex models had more issues with over-fitting. As an extension to this project, I would recommend trying a range of different architectures to see what works best for your data.

# Performance of the model

In the end, the model was able to achieve a final validation score of 87.79% (based on the F1 metric), which corresponded to a total accuracy of around 93.16%. Looking at the train/validation loss and score graphs, we can see that there was a bit of overfitting after around 5 epochs of training, therefore it would be ideal to stop training around this point. 

A further look into the accuracy of the model on different classes showed that it had a very high accuracy when classifying characteristics with clear visual definitions such as gender (98.12%), presence of a beard (95.37%), wearing a hat (98.89%) and whether the person is smiling (92.90%). With more abstract/harder to define characteristics however, such as attractiveness (81.75%) and youth (88.58%), the model performed a lot worse. This highlights how important it is to have a clearly defined dataset, and that a neural network is only as good as the data it is trained on. Furthermore, for traits such as attractiveness, this is entirely subjective and will depend on whoever labelled the dataset.

It is also interesting to note the discrepancy between recall and precision of the model for over/under-represented traits in the dataset. For example, because the trait 'bald' was under-represented (only a small proportion of the dataset was labelled with it), the network produced a lot more false negatives for this category, as it was skewed towards giving a false output (if it always guessed false, it would be right most of the time). 

On new test images, as expected, the network was very good at detecting features such as sex and smiling, but less so at attractiveness. 

# Improvements that could be made

An easy way to increase the accuracy of the model is to remove the ambiguous classes and even out the spread of the dataset. This was not done in this case to showcase how a non-ideal dataset affects structure. Furthermore, more traits could be tested, and over-fitting could be further reduced with different model architectures and data augmentation which may increase its accuracy when trained over many epochs.