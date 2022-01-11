import numpy as np
import scipy.io as sio

# SimHash works as follows:
# You generate a random plane, image vectors at an angle less than |90 degrees| from the plane will
# have a positive cosine (i.e. cos(angle(plane, vector)) > 0) while image vectors at an angle greater
# than |90 degrees| will have a negative cosine. It will almost never happen by vectors pointing in exactly
# the same direction will have a 0 cosine
# The angle between a vector and a plane is 90 - (angle between vector and normal vector of plane)
# You generate r random planes for each vector pair and the respective signed cosines must be the same for all r planes
# You do this t times and if the are similar in one of the t runs then the two vectors are similar
# Thus the probability of a false positive, (given angle theta between two particular vectors and theta is small) is
# probability(vectors are not similar|theta) = (1 - (1-(theta/pi))^r)^t

# load .mat file
file_mat = sio.loadmat('mnist.mat')
# file_mat is a dictionary with key "testX"
# values for that key are a list of lists of pixel values for 28 x 28 images (784 images)
# there are 60,000 images in the full set so we will use the test set for the sake of proof of concept
# load the test images into textX variable
testX = file_mat["testX"]
# convert list of lists of numbers to numpy array of numpy arrays of numbers
# ensure image vectors are normalized (their length is 1 - unit)
testX = np.array([np.true_divide(np.array(img), np.linalg.norm(img)) for img in testX])
# convert numpy array matrix to numpy matrix type
testX = np.matrix(testX)
# get the transpose (now columns are image vectors (i.e. pixel values for each image))
testX_transpose = np.transpose(testX)
# calculate dot product between the two. Gives matrix where position (i, j) is the dot product between
# image vector i and image vector j
dot_product = np.dot(testX, testX_transpose)
# since cos(theta) = <i, j>/(len(i) * len(j)), and len(i) for all i is just 1, then we know
# cos(theta) = dot products found in the dot_product matrix
# Thus to get the angles between each image we find the arcosine (upper bound
# values by 1 to avoid rounding errors messing up the arcosine function)
angles_matrix = np.arccos(np.minimum(dot_product, 1))
# get the probability any two images will collide (be on the same side of a random hyperplane)
probability_images_similar = 1 - (np.true_divide(angles_matrix, np.pi))
# what we need to set r to to ensure that if two images have cosine similarity > 0.95 (that is cos(theta) >= 0.95)
# then the false negative rate will less than 2 percent (assume r is between 1 and 35)
print(probability_images_similar)
