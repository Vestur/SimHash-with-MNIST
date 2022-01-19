import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd

# SimHash works as follows:
# You generate a random plane, image vectors at an angle less than |90 degrees| from the plane will
# have a positive cosine (i.e. cos(angle(plane, vector)) > 0) while image vectors at an angle greater
# than |90 degrees| will have a negative cosine. It will almost never happen by vectors pointing in exactly
# the same direction will have a 0 cosine
# The angle between a vector and a plane is 90 - (angle between vector and normal vector of plane)
# You generate r random planes (length of signature) for each vector pair and the respective signed cosines must be the same for all r planes
# You do this t times and if the are similar in one of the t runs then the two vectors are similar
# Thus the probability of a false positive, (given angle theta between two particular vectors and theta is small) is
# probability(vectors are not similar|theta) = (1 - (1-(theta/pi))^r)^t
def SimHash(r, t, file_name, num_to_get):
    # load .mat file
    file_mat = sio.loadmat(file_name)
    # file_mat is a dictionary with key "testX"
    # values for that key are a list of lists of pixel values for 28 x 28 images (784 images)
    # there are 60,000 images in the full set so we will use the test set for the sake of proof of concept
    # load the test images into X variable
    testX = file_mat["testX"]
    X = file_mat["trainX"]
    # convert list of lists of numbers to numpy array of numpy arrays of numbers
    # ensure image vectors are normalized (their length is 1 - unit)
    X = np.array([np.true_divide(np.array(img), np.linalg.norm(img)) for img in X])
    testX = np.array([np.true_divide(np.array(img), np.linalg.norm(img)) for img in testX])
    # convert numpy array matrix to numpy matrix type
    X = np.matrix(X)
    testX = np.matrix(testX)
    testY = file_mat["testY"]
    # # get the transpose (now columns are image vectors (i.e. pixel values for each image))
    # testX_transpose = np.transpose(testX)
    # # calculate dot product between the two. Gives matrix where position (i, j) is the dot product between
    # # image vector i and image vector j
    # dot_product = np.dot(testX, testX_transpose)
    # # since cos(theta) = <i, j>/(len(i) * len(j)), and len(i) for all i is just 1, then we know
    # # cos(theta) = dot products found in the dot_product matrix
    # # Thus to get the angles between each image we find the arcosine (upper bound
    # # values by 1 to avoid rounding errors messing up the arcosine function)
    # angles_matrix = np.arccos(np.minimum(dot_product, 1))
    # t times see if two images return the same signature at least once
    # signature length is r
    valid_images = []
    for index in range(len(testY[0])):
        if testY[0][index] == num_to_get:
            valid_images.append(testX[index])
    images = []
    # print(len(valid_images))
    max_images = 20
    for i in range(t):
        for img in valid_images:
            if max_images <= 0:
                break
            signature = np.random.normal(0, 1, size=(784, r))
            # i'th rows are signature of i'th image
            data_signature = X.dot(signature)
            signed_signature = np.sign(data_signature)
            images_hash_values = ((signed_signature + 1)/2).dot(2 ** (np.arange(r).reshape(r, t)))
            hash_to_get = ((np.sign(img.dot(signature)) + 1)/2).dot(2 ** (np.arange(r).reshape(r, t)))
            valid_index = None
            images_hash_values = np.around(images_hash_values, decimals=3)
            found = False
            for index in range(len(images_hash_values)):
                if images_hash_values[index] == hash_to_get:
                    found = True
                    valid_index = index
                    break
            if found:
                images.append(X[valid_index])
            max_images -= 1
    return images



# what we need to set r and t to to ensure that if two images have cosine similarity > 0.95
# (that is cos(theta) >= 0.95)
# then the false negative rate will less than 3 percent (assume r is integer between 1 and 30)
# we also want false positive rate to be lower than 3 % if two images have cosine similarity < 0.6
def get_r_t(cosine_similarity_high = 0.95, false_negative_max = 0.03, cosine_similarity_low = 0.6, false_positive_max = 0.03):
    theta_low = np.arccos(cosine_similarity_low)
    theta_high = np.arccos(cosine_similarity_high)
    # print(theta_high)
    # print(1 - (theta_high/np.pi))
    # print(false_negative_max)
    found_r_t = []
    # assumes r can be found in range {1,...,30}
    # assumes t can be found in range {1,...24}
    for r in range(1, 31):
        for t in range(1, 100):
            if ((1 - (1 - (theta_high/np.pi))**r)**t < false_negative_max) and ((1 - (1 - (1 - (theta_low/np.pi))**r)**t) <= false_positive_max):
                # print(r, t)
                found_r_t.append((r, t))
                break
    return found_r_t

def plot_probability_collision(r, t):
    plt.xlim(0,1)
    plt.ylim(0,1)
    x = np.arange(0, 1, 0.01)
    y = [1 - ((1 - ((1-(np.arccos(x)/np.pi))**r))**t) for x in np.arange(0, 1, 0.01)]
    plt.plot(x, y)
    plt.show()

def plot(images):
    for image in images:
        image = image.reshape((28, 28))
        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.show()

def main():
    # what we need to set r and t to to ensure that if two images have cosine similarity > 0.95
    # (that is cos(theta) >= 0.95)
    # then the false negative rate will less than 3 percent (assume r is integer between 1 and 30)
    # we also want false positive rate to be lower than 3 % if two images have cosine similarity < 0.6
    r_t = get_r_t(0.95, 0.03, 0.60, 0.03)
    if len(r_t) == 0:
        print("No suitable r, t pair found")
        return -1
    else:
        r, t = r_t[0]
        plot_probability_collision(r, t)
    num_to_find = 1
    found_images = SimHash(25, 1, 'mnist.mat', num_to_find)
    plot(found_images)
    print(len(found_images))
    return 0

main()
if "__name__" == "main":
    main()