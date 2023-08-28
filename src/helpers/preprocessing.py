'''
    This file contains functions for preprocessing images.
    These are the functions meant to be used:

        - gmms_preprocess_image: This function preprocesses an image using a Gaussian Mixture Model with extra sensitivity for material.
        - gmm_preprocess_image: This function preprocesses an image using a Gaussian Mixture Model.
        - resize_image: This function simply rescales images to a square of size x size pixels. Meant to work with square images.
        - pad_image: This function pads images to a square of size x size pixels. Meant to work with square images.
        - simplify: This function simplifies the image by emphasizing edges and reducing noise through color simplification based on pixel color variation.
        - preprocess_image: This function preprocesses an image by removing noise and emphasizing edges.
        - augment_dataset: This function augments a dataset by applying random rotations and color jitter to images.
        - convert_to_grayscale_recursive: This function converts all images in a folder to grayscale.
        - gmm_parameters: This function fits a Gaussian Mixture Model to an image and returns the means and variances of each cluster.
        - create_bar_chart: This function creates a bar chart with the pixel intensities on the x-axis and the counts on the y-axis.
        - create_bar_chart_with_gmm: This function creates a bar chart with the pixel intensities on the x-axis and the counts on the y-axis. It also adds a red line at means and blue lines at means +/- standard deviations.
'''
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture

'''
Gaussian Mixture Model Preprocessing with Extra Sensitivty for Material

    Params:
        img_path: path to image
        num_components: number of clusters to fit
    
    Returns:
        data: A 2d numpy array. Simplified image split into num_components+1 homogneous 
        regions, with a focus on the recently extruded material
'''
def gmms_preprocess_image(img, num_components):
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    original_shape = img.shape
    img = flatten(img)
    
    # Get means and variances assuming 3 clusters
    means, stdvs = gmm_parameters(img, 3)    

    # Get information of the cluster with the highest mean
    means = np.sort(means)
    stdvs = np.sqrt(np.sort(stdvs))
    max_mean = means[-1]
    max_stdv = stdvs[-1] * 2
    if(max_stdv > 40): max_stdv = 50
    
    # This is the range of intensities we believe make up the recently extruded material
    material_range = (max_mean - max_stdv, max_mean + max_stdv)
    material_indices = get_indices_within_range(img, material_range)
    material_data = img[material_indices]
    
    # Split the material data into num_components clusters
    material_means, material_stdvs = gmm_parameters(material_data, num_components)
    material_means = np.sort(material_means)
    material_stdvs = np.sqrt(np.sort(material_stdvs))
    
    # Combine information from first cluster, and the material clusters
    total_means = np.insert(material_means, 0, means[0])
    total_stdvs = np.insert(material_stdvs, 0, stdvs[0])
        
    # Given the means and standard deviations, split the range 0-255 into 4 intervals
    ranges = get_ranges(total_means, total_stdvs)
    
    # Create a list of colors to use for each range
    colors = []
    for i in range(num_components+1):
        colors.append((255/(num_components+1))*i)
    
    # Replace all values in the data that fall within the ranges with the corresponding color
    for i in range(num_components+1):
        img = replace_values_within_range(img, ranges[i], colors[i])
    
    img = unflatten(img, original_shape)
    return img


# Simply rescales images to a square of size x size pixels. Meant to work with square images.
def resize_image(input_path, output_path, size):
    with Image.open(input_path) as img:
        img_resized = img.resize((size, size))
        img_resized.save(output_path)
        
        
# Pads images to a square of size x size pixels. Meant to work with square images.
def pad_image(input_path, output_path, final_size):
    with Image.open(input_path) as img:
        width, height = img.size

        new_width = final_size
        new_height = final_size

        left_padding = (new_width - width) // 2
        top_padding = (new_height - height) // 2
        right_padding = new_width - width - left_padding
        bottom_padding = new_height - height - top_padding

        img_with_border = ImageOps.expand(img, (left_padding, top_padding, right_padding, bottom_padding), fill='black')
        img_with_border.save(output_path)


# The method simplifies the image by emphasizing edges and reducing noise through color simplification based on pixel color variation.
def simplify(input_path: str):
    img = cv2.imread(input_path)[:,:,0]
    # Get standard deviation across all pixels in image
    x = np.std(img)
    
    # Compute the 'range' threshold based on the standard deviation
    threshold = 2.04023 * x - 4.78237
    
    if(x < 20): threshold = 5
    threshold = 5
    
    # Find the maximum pixel intensity in the image
    whitestPixel = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel): whitestPixel = img[i][j]
            
    # Set all pixels with intensities greater than 'whitestPixel - threshold' to 255
        for j in range(len(img[0])):
            if(img[i][j] > whitestPixel - threshold): img[i][j] = 255
            
    # Cutoff is the minimum pixel intensity we have already simplified
    cutoff = 255 - threshold
    
    # Loop until all pixels have been categorized
    while(True):    
        whitestPixel = 0
        
        # Find the maximum pixel intensity that's less than 'cutoff'
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] < cutoff and img[i][j] > whitestPixel): whitestPixel = img[i][j]
                
        # Break the loop if no such pixel is found
        if whitestPixel == 0: break
        
        # Set all pixels with intensities greater than 'whitestPixel - threshold' and less than 'cutoff' to 'whitestPixel'
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j] > whitestPixel - threshold and img[i][j] < cutoff): img[i][j] = whitestPixel
                
        # Update cutoff
        cutoff = whitestPixel - threshold
    return img


# Given an image path, removes noise and emphasizes edges. Returns the processed image as a numpy array.
def preprocess_image(img_path):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

    # Denoise
    img = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    # Convert to PIL Image for edge enhancement
    img = Image.fromarray(img)

    # Sharpen edges using Unsharp Mask
    img = img.filter(ImageFilter.UnsharpMask(radius=8, percent=100))

    return np.array(img)



'''
Gaussian Mixture Model

    Params: 
        img_path: path to image
        num_components: number of clusters to fit
        
    Returns:
        means: list of means for each cluster
        variances: list of variances for each cluster
'''
def gmm_parameters(data, num_components):
    # Reshape data to fit the GMM input requirements (should be 2D)
    data = data.reshape(-1, 1)

    # Initialize Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_components, random_state=0)

    # Fit the GMM to the data
    gmm.fit(data)

    # Extract means and variances
    means = gmm.means_.flatten()  # Flatten to convert to 1D
    variances = gmm.covariances_.flatten()

    # Return the parameters as a list
    return list(means), list(variances)

   
# Return a list of indices in the array that fall within the provided range.
def get_indices_within_range(array, range):
    lower, upper = range
    indices = np.where((array >= lower) & (array <= upper))
    return indices[0].tolist()

# Replace all values in the array that fall within the provided range with the given number.
def replace_values_within_range(array, range, replacement_value):
    lower, upper = range
    array[(array >= lower) & (array <= upper)] = replacement_value
    return array

# Splits the range 0-255 into intervals based on the provided means and standard deviations.
def get_ranges(means, std_devs):
    # Calculate the raw ranges
    raw_ranges = list(zip(means - std_devs, means + std_devs))
    
    # Fix overlaps and extend to 0 and 255
    ranges = []
    for i, (low, high) in enumerate(raw_ranges):
        # If this is the first range, extend the lower bound to 0
        if i == 0:
            low = 0
        # Otherwise, adjust the lower bound to be the midpoint between this range's lower bound and the previous range's upper bound
        else:
            low = (low + ranges[-1][1]) / 2

        # If this is the last range, extend the upper bound to 255
        if i == len(raw_ranges) - 1:
            high = 255
        # Otherwise, adjust the upper bound to be the midpoint between this range's upper bound and the next range's lower bound
        else:
            high = (high + raw_ranges[i+1][0]) / 2
        
        ranges.append((low, high))
    
    return ranges
    
# Turns 2d matrix into 1d vector
def flatten(matrix):
    # Use numpy's ravel function to flatten the matrix
    return matrix.ravel()

# Turns 1d vector into 2d matrix
def unflatten(vector, original_shape):
    # Use numpy's reshape function to convert the vector back to the original matrix shape
    return vector.reshape(original_shape)