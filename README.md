# DL-course-PyTorch
All this course is taken from this video [PyTorch for Deep Learning & Machine Learning - Full course](https://www.youtube.com/watch?v=V_xro1bcAuA&pp=2AbqoAU%3D)
## 00. PyTorch Fundamentals

0. **Introduction to Tensors** - Tensors are the basic building block of all of machine learning and deep learning.
1. **Creating Tensors** - Tensors can represent almost any kind of data (images, words, tables of numbers).
2. **Getting Information** from Tensors - If you can put information into a tensor, you'll want to get it out too.
3. **Manipulating Tensors** - Machine learning algorithms (like neural networks) involve manipulating tensors in many different ways such as adding, multiplying, combining.
4. **Dealing with Tensor Shapes** - One of the most common issues in machine learning is dealing with shape mismatches (trying to mix wrong-shaped tensors with other tensors).
5. **Indexing on Tensors** - If you've indexed on a Python list or NumPy array, it's very similar with tensors, except they can have far more dimensions.
6. **Mixing PyTorch Tensors and NumPy** - PyTorch works with `torch.Tensor`, NumPy with `np.ndarray` — sometimes you'll want to mix and match them.
7. **Reproducibility** - Machine learning is very experimental and since it uses a lot of randomness to work, sometimes you'll want that randomness to not be so random.
8. **Running Tensors on GPU** - GPUs (Graphics Processing Units) make your code faster — PyTorch makes it easy to run your code on GPUs.

## 01. PyTorch Workflow Fundamentals

In this module we're going to cover a standard PyTorch workflow
![image](https://github.com/user-attachments/assets/6499c066-9b78-4334-8cb1-6cd372ab7909)

1. **Getting data ready** - Data can be almost anything but to get started we're going to create a simple straight line
2. **Building a model** - Here we'll create a model to learn patterns in the data, we'll also choose a loss function, optimizer and build a training loop.
3. **Fitting the model to data (training)** - We've got data and a model, now let's let the model (try to) find patterns in the (training) data.
4. **Making predictions and evaluating a model (inference)** - Our model's found patterns in the data, let's compare its findings to the actual (testing) data.
5. **Saving and loading a model** - You may want to use your model elsewhere, or come back to it later, here we'll cover that.

## 02. PyTorch Neural Network Classification

0. **Architecture of a classification neural network** - Neural networks can come in almost any shape or size, but they typically follow a similar floor plan.
1. **Getting binary classification data ready** - Data can be almost anything but to get started we're going to create a simple binary classification dataset.
2. **Building a PyTorch classification model** - Here we'll create a model to learn patterns in the data, we'll also choose a loss function, optimizer and build a training loop specific to classification.
3. **Fitting the model to data (training)** - We've got data and a model, now let's let the model (try to) find patterns in the (training) data.
4. **Making predictions and evaluating a model (inference)** - Our model's found patterns in the data, let's compare its findings to the actual (testing) data.
5. **Improving a model (from a model perspective)** - We've trained and evaluated a model but it's not working, let's try a few things to improve it.
6. **Non-linearity** - So far our model has only had the ability to model straight lines, what about non-linear (non-straight) lines?
7. **Replicating non-linear functions** - We used non-linear functions to help model non-linear data, but what do these look like?
8. **Putting it all together with multi-class classification** - Let's put everything we've done so far for binary classification together with a multi-class classification problem.

## 03. PyTorch Computer Vision

0. **Computer vision libraries in PyTorch** - PyTorch has a bunch of built-in helpful computer vision libraries, let's check them out.
1. **Load data** - To practice computer vision, we'll start with some images of different pieces of clothing from FashionMNIST.
2. **Prepare data** - We've got some images, let's load them in with a PyTorch DataLoader so we can use them with our training loop.
3. **Model 0: Building a baseline model** - Here we'll create a multi-class classification model to learn patterns in the data, we'll also choose a loss function, optimizer and build a training loop.
4. **Making predictions and evaluating model 0** - Let's make some predictions with our baseline model and evaluate them.
5. **Setup device agnostic code for future models** - It's best practice to write device-agnostic code, so let's set it up.
6. **Model 1: Adding non-linearity** - Experimenting is a large part of machine learning, let's try and improve upon our baseline model by adding non-linear layers.
7. **Model 2: Convolutional Neural Network (CNN)** - Time to get computer vision specific and introduce the powerful convolutional neural network architecture.
8. **Comparing our models** - We've built three different models, let's compare them.
9. **Evaluating our best model** - Let's make some predictions on random images and evaluate our best model.
10. **Making a confusion matrix**	- A confusion matrix is a great way to evaluate a classification model, let's see how we can make one.
11. **Saving and loading the best performing model** - Since we might want to use our model for later, let's save it and make sure it loads back in correctly.

## 04. PyTorch Custom Datasets

0. **Importing PyTorch and setting up device-agnostic code** - Let's get PyTorch loaded and then follow best practice to setup our code to be device-agnostic.
1. **Get data** - We're going to be using our own custom dataset of pizza, steak and sushi images.
2. **Become one with the data (data preparation)** - At the beginning of any new machine learning problem, it's paramount to understand the data you're working with. Here we'll take some steps to figure out what data we have.
3. **Transforming data** - Often, the data you get won't be 100% ready to use with a machine learning model, here we'll look at some steps we can take to transform our images so they're ready to be used with a model.
4. **Loading data with ImageFolder (option 1)** - PyTorch has many in-built data loading functions for common types of data. ImageFolder is helpful if our images are in standard image classification format.
5. **Loading image data with a custom Dataset** - What if PyTorch didn't have an in-built function to load data with? This is where we can build our own custom subclass of torch.utils.data.Dataset.
6. **Other forms of transforms (data augmentation)** - Data augmentation is a common technique for expanding the diversity of your training data. Here we'll explore some of torchvision's in-built data augmentation functions.
7. **Model 0: TinyVGG without data augmentation** - By this stage, we'll have our data ready, let's build a model capable of fitting it. We'll also create some training and testing functions for training and evaluating our model.
8. **Exploring loss curves** - Loss curves are a great way to see how your model is training/improving over time. They're also a good way to see if your model is underfitting or overfitting.
9. **Model 1: TinyVGG with data augmentation** - By now, we've tried a model without, how about we try one with data augmentation?
10. **Compare model results** - Let's compare our different models' loss curves and see which performed better and discuss some options for improving performance.
11. **Making a prediction on a custom image** - Our model is trained to on a dataset of pizza, steak and sushi images. In this section we'll cover how to use our trained model to predict on an image outside of our existing dataset.
