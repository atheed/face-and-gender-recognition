from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import re
import urllib
from os import listdir
from os.path import isfile, join
from PIL import Image


act = ['Aaron Eckhart',  'Adam Sandler',   'Adrien Brody',  
        'Andrea Anders',  'Ashley Benson',  'Christina Applegate',    
        'Dianna Agron',  'Gillian Anderson']


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''
    From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    '''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


def pca(X):
    '''    
    Principal Component Analysis
    
    input: X, matrix with training data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance and mean.
        
    From: Jan Erik Solem, Programming Computer Vision with Python
    http://programmingcomputervision.com/
    '''
    
    # get dimensions
    num_data,dim = X.shape
    
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    
    if dim>num_data:
        # PCA - compact trick used
        M = dot(X,X.T) # covariance matrix
        e,EV = linalg.eigh(M) # eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T # this is the compact trick
        V = tmp[::-1] # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data
    
    # return the projection matrix, the variance and the mean
    return V,S,mean_X


def helper_bounding_box(line):
    '''
    Helper function to take face bounds given in a line (string), and 
    return a 4-tuple of the bounding box
    '''
    # splits the line by the commas
    box_bounds = line.split()[1].split(',')
    
    # creates a 4-tuple with the relevant bounds, and returns it
    box = (int(box_bounds[0]), int(box_bounds[1]),
            int(box_bounds[2]), int(box_bounds[3]))
    return box


def get_digit_matrix(img_dir):
    '''
    Returns a 3-tuple representing all images in a given directory
    (for Principal Component Analysis)
    '''
    im_files = sorted([img_dir + filename 
                    for filename in os.listdir(img_dir) 
                    if filename[-4:] == ".jpg"])
    im_shape = array(imread(im_files[0])).shape[:2] # open one image to get the size 
    im_matrix = array([imread(im_file).flatten() for im_file in im_files])
    im_matrix = array([im_matrix[i,:]/(norm(im_matrix[i,:] + 0.0001)) for i in range(im_matrix.shape[0])])
    return (im_matrix, im_shape, im_files)


def display_save_25_comps(proj_matrix, im_shape):
    '''
    Display first 25 components in the projection matrix
    '''
    figure()
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        gray()
        imshow(proj_matrix[i,:].reshape(im_shape))
    savefig('display_save_25_comps.jpg')  
    show()        


# downloading files
def download():
    '''
    Function that was used to download the files from the faces_subset.txt file
    '''
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("faces_subset.txt"):
            if a in line:
                filename = name+str(i)+" "+line.split()[5]+" "+'.'+line.split()[4].split('.')[-1]                
                
                # the bounds of the face in the image are appended to the filename.
                # this is done so that we can later crop the faces using these
                # box bounds. puts these files in the 'uncropped' folder
                timeout(testfile.retrieve, (line.split()[4], "uncropped/" + filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
    
                print filename
                i += 1


def crop_and_grey():
    '''
    Function that takes the downloaded images, crops them and converts them to
    greyscale
    '''
    onlyfiles = [ f for f in listdir("uncropped") if isfile(join("uncropped",f)) ]
    for f in onlyfiles:
        try:
            box = helper_bounding_box(f)
            im = Image.open('uncropped/'+f).convert('L')
            cropped = np.asarray(im.crop(box))
            while cropped.shape[0]>64:
                cropped = imresize(cropped, .5)
            cropped = imresize(cropped, [32,32])    
            img = Image.fromarray(cropped)
            img.save('cropped/'+f.split()[0]+'.jpg')
        except:
            continue


def closest_match(im_matrix, proj_matrix, image, mean_im, k):
    '''
    Finds closest match using eigenfaces.
    '''
    # will hold all weights
    all_weights = []
    
    # iterate through each image in im_matrix (i.e. the training set). at each
    # iteration, we compute the dot product of each eigenface (in range k) with
    # the normalized current image (i.e. mean_im is subtracted) 
    for curr_image in im_matrix:
        weights = []
        for i in range(k):
            weights.append(np.dot(proj_matrix[i], curr_image - mean_im))
        # append these k weights for this particular image to the array 
        # holding all weights
        all_weights.append(weights)
    
    # normalize the all_weights array, in the same way get_digit_matrix does
    all_weights = np.array(all_weights)
    all_weights = array([all_weights[i,:]/(norm(all_weights[i,:] + 0.0001)) 
                        for i in range(all_weights.shape[0])])
    
    # will hold the weights relating to the input image
    image_weights = []
    
    # iterate through all k eigenfaces, computing the dot product of each eigenface
    # with the normalized input image (i.e. mean_im is subtracted)
    for i in range(k):
        image_weights.append(np.dot(proj_matrix[i], image - mean_im))
    
    # normalize the input image's k weights 
    #image_weights = np.array(image_weights)
    image_weights = array(image_weights / (norm(image_weights) + 0.0001))
    
    # initialize the variables that will find the smallest distance between
    # images and the index of those images
    min_euclidean_distance = float("inf")
    index_of_min = 0
    
    # loop through all weights and check, using normalized cross correlation, 
    # the euclidean distance between that weight and the input image's weights. 
    # if this distance is smaller than the previous minimum, update so that this
    # is the minimum (and also store the relevant index)
    # this gives us the closest image match, and where to find that image
    for i in range(len(all_weights)):
        curr_dist = np.linalg.norm(all_weights[i] - image_weights)
        if min_euclidean_distance > curr_dist:
            min_euclidean_distance = curr_dist
            index_of_min = i
    
    # return where to find the closest matched image
    return index_of_min
    

def face_recognition(k):
    '''
    Computes facial recognition. 
    '''
    # get digit matrix for all files in TRAINING folder, and normalize
    im_matrix, im_shape, im_files = get_digit_matrix('training/')
    for i in range(im_matrix.shape[0]):
        im_matrix[i,:] = im_matrix[i,:]/255.0
    
    proj_matrix, variance, mean_im = pca(im_matrix)
    
    # get digit matrix for all files in TEST folder, and normalize
    im_matrix1, im_shape1, im_files1 = get_digit_matrix('test/')
    for i in range(im_matrix1.shape[0]):
        im_matrix1[i,:] = im_matrix1[i,:]/255.0
    
    # initialize total number of guesses and number of correct guesses.
    # to be used to calcualte success percentages
    guesses_correct = 0
    guesses_all = 0
    
    #loop through all images in TEST folder
    for i in range(len(im_matrix1)):
        # compute where to find the closest matched image
        index_of_min = closest_match(im_matrix, proj_matrix, im_matrix1[i, :], mean_im, k)
        
        # variable to denote index of first number in filename. for splicing.
        first_num = 0
        
        # find first occurrence of number in filename, for splicing
        first_num = re.search('\d', im_files1[i])
        
        # splice so only actor/actress name is left (no numbers, no filepaths, etc.)
        current_actor = im_files1[i][:first_num.start()]
        current_actor = current_actor[current_actor.rfind('/') + 1:]
        
        # reinitialize first_num to 0
        first_num = 0
        
        # find first occurrence of number in closest matched file; for splicing
        first_num = re.search('\d', im_files[index_of_min])
        
        # splice so only actor/actress name is left (no numbers, no filepaths, etc.)
        actor_guess = im_files[index_of_min][:first_num.start()]
        actor_guess = actor_guess[actor_guess.rfind('/') + 1:]
        
        # check if actor is the same as guessed actor. if yes, recognition is correct.
        # if no, recognition is false. increment guesses appropriately
        if current_actor == actor_guess:
            print('Correct guess')
            guesses_correct += 1
            guesses_all += 1
        else:
            print('Incorrect guess')	
            guesses_all += 1
        
        print ("Total number of guesses: " + str(guesses_all) +
                "\nNumber of correct guesses: " + str(guesses_correct) + "\n")
            
    # return percentage of successful actor recognitions
    return ("Percentage of successful guesses: " + str(float(float(guesses_correct)/float(guesses_all))*100) + "%")


def gender_classification(k):
    '''
    Computes gender classification. 
    '''
    # get digit matrix for all files in TRAINING folder, and normalize
    im_matrix, im_shape, im_files = get_digit_matrix('training/')
    for i in range(im_matrix.shape[0]):
        im_matrix[i,:] = im_matrix[i,:]/255.0
    
    proj_matrix, variance, mean_im = pca(im_matrix)
    
    # get digit matrix for all files in TEST folder, and normalize
    im_matrix1, im_shape1, im_files1 = get_digit_matrix('test/')
    for i in range(im_matrix1.shape[0]):
        im_matrix1[i,:] = im_matrix1[i,:]/255.0
    
    # initialize total number of guesses and number of correct guesses.
    # to be used to calcualte success percentages
    guesses_correct = 0
    guesses_all = 0
    
    #loop through all images in TEST folder
    for i in range(len(im_matrix1)):
        # compute the index of the closest matched image
        index_of_min = closest_match(im_matrix, proj_matrix, im_matrix1[i, :], mean_im, k)
        
        # variable to denote index of first number in filename. for splicing.
        first_num = 0
        
        # find first occurrence of number in filename, for splicing
        first_num = re.search('\d', im_files1[i])
        
        # splice so only actor/actress name is left (no numbers, no filepaths, etc.)
        current_actor = im_files1[i][:first_num.start()]
        current_actor = current_actor[current_actor.rfind('/') + 1:]
        
        # reinitialize first_num to 0
        first_num = 0
        
        # find first occurrence of number in closest matched file; for splicing
        first_num = re.search('\d', im_files[index_of_min])
        
        # splice so only actor/actress name is left (no numbers, no filepaths, etc.)
        actor_guess = im_files[index_of_min][:first_num.start()]
        actor_guess = actor_guess[actor_guess.rfind('/') + 1:]
                
        # check if actor and guessed actor are both male. if yes, correct,
        # increment number of correct guesses and total guesses
        if current_actor in ['eckhart', 'sandler', 'brody'] and actor_guess in ['eckhart', 'sandler', 'brody']:
            print('Correct guess')
            guesses_correct += 1
            guesses_all += 1
            
        # check if actor and guessed actor are both female. if yes, correct,
        # increment number of correct guesses and total guesses
        elif (current_actor in ['anders', 'benson', 'applegate', 'agron', 'anderson'] 
                and actor_guess in ['anders', 'benson', 'applegate', 'agron', 'anderson']):
            print('Correct guess')
            guesses_correct += 1
            guesses_all += 1
        
        # else, gender of actor and guessed actor are different. then, wrong, 
        # increment only total number of guesses
        else:
            print('Incorrect guess')
            guesses_all += 1
        
        print ("Total number of guesses: " + str(guesses_all) +
                "\nNumber of correct guesses: " + str(guesses_correct) + "\n")
            
    # return percentage of successful gender classifications
    return ("Percentage of successful guesses: " + str(float(float(guesses_correct)/float(guesses_all))*100) + "%")