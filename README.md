# Face and Gender Recognition
Grabs celebrities' images from the publicly-available FaceScrub dataset, and uses machine learning to do facial recognition and gender classification on these images

### Software
- Python
- NumPy/SciPy stack

## Overview and Method
Primarily uses **principal components analysis** (using eigenfaces) and other **machine learning** techniques to recognize the faces of celebrities. Additionally, performs gender classification on these celebrity images too. 

The images were first separated into three non-overlapping parts: the training set (100 images per actor), the validation set (10 images per actor), and the test set (10 images per actor). Then, using PCA with top-k eigenfaces, the system is trained on the training set of images. After training, the best value of k (for top-k eigenfaces) is determined by running the system on the validation set. Once the best value of k is obtained, the system is then run on the test set.

## Sample Images
Below are a couple of sample images (from the FaceScrub dataset) that were used to perform gender classification and face recognition. They are, of course, 32x32 in the interests of keeping the PCA computations relatively computationally-efficient.

<img src="/img/agron19c.jpg" width="128px" height="128px"/>

<img src="/img/sandler49c.jpg" width="128px" height="128px"/>

## Results
The best value of k (for top-k eigenfaces) was found to be **k=80**.

The facial recognition accuracy (percentage of correct recognitions), with k=80, for the test set was: **67.5%**.

The gender classification accuracy (percentage of correct classifications), with k=80, for the test set was: **86.25%**.
