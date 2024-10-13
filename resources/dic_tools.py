# Local
from resources.dic_Class import DicClass

# Math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# Data
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the sets of Gradient-Processed Correlation to try
combinations = [
    [False, False, False, False, False, False],
    [True, False, False, True, False, False], # I
    [True, False, False, False, True, False], # G
    # [True, False, False, False, False, True], # H
    # [True, False, False, True, True, False], # IG
    # [True, False, False, True, False, True], # IH
    # [True, False, False, False, True, True], # GH
    # [True, False, False, True, True, True], # IGH

    # [False, True, False, True, False, False], # G
    # [False, True, False, False, True, False], # GG
    # [False, True, False, False, False, True], # GH
    # [False, True, False, True, True, False], # IGG
    # [False, True, False, True, False, True], # IGH
    # [False, True, False, False, True, True], # GGH
    # [False, True, False, True, True, True], # IGGH
    # [False, False, True, True, False, False], # H
    # [False, False, True, False, True, False], # GH
    # [False, False, True, False, False, True], # HH
    # [False, False, True, True, True, False], # IGH
    # [False, False, True, True, False, True], # IHH
    # [False, False, True, False, True, True], # GHH
    # [False, False, True, True, True, True], #IGHH

    # [True, True, False, True, False, False], # IG
    # [True, True, False, False, True, False], # IGG
    # [True, True, False, False, False, True], # IGH
    # [True, True, False, True, True, False], # IIGG
    # [True, True, False, True, False, True], # IIGH
    # [True, True, False, False, True, True], # IGGH
    # [True, True, False, True, True, True], #IIGGH

    # [False, True, True, True, False, False], # GH
    # [False, True, True, False, True, False], # GGH
    # [False, True, True, False, False, True], # GHH
    # [False, True, True, True, True, False], # IGGH
    # [False, True, True, True, False, True], 
    # [False, True, True, False, True, True], 
    # [False, True, True, True, True, True], 

    # [True, False, True, True, False, False],
    # [True, False, True, False, True, False], 
    # [True, False, True, False, False, True],
    # [True, False, True, True, True, False],
    # [True, False, True, True, False, True], 
    # [True, False, True, False, True, True], 
    # [True, False, True, True, True, True], 

    # [True, True, True, True, False, False],
    # [True, True, True, False, True, False], 
    # [True, True, True, False, False, True],
    # [True, True, True, True, True, False],
    # [True, True, True, True, False, True], # IIGH
    # [True, True, True, False, True, True], # IGGHH
    # [True, True, True, True, True, True], # IIGGHH
]

def padItUp(image1, image2, padType='edge'):
    """
    Pads two images to the same dimensions.

    Parameters:
    image1 (numpy.ndarray): The first input image.
    image2 (numpy.ndarray): The second input image.
    padType (str): The padding type to use ('edge' or 'constant').

    Returns:
    tuple: A tuple containing the two padded images.
    """
    # Determine the maximum dimensions of the input images
    max_height = np.max([image1.shape[0], image2.shape[0]])
    max_width = np.max([image1.shape[1], image2.shape[1]])
    target_shape = (max_height, max_width)

    # Pad both images to the target shape
    image1_padded = padImage(image1, target_shape, padType)
    image2_padded = padImage(image2, target_shape, padType)

    return image1_padded, image2_padded

def padImage(image, target_shape, padType='edge'):
    """
    Pads an image to the target shape.

    Parameters:
    image (numpy.ndarray): The input image to be padded.
    target_shape (tuple): The desired shape after padding (height, width).
    padType (str): The padding type to use ('edge' or 'constant').

    Returns:
    numpy.ndarray: The padded image.
    """
    # Calculate padding amounts for each dimension
    x_diff = target_shape[0] - image.shape[0]
    y_diff = target_shape[1] - image.shape[1]
    
    # Determine the amount of padding needed on each side of the image
    if x_diff % 2 == 0:
        x_pad_before = x_diff // 2
        x_pad_after = x_diff // 2
    else:
        x_pad_before = x_diff // 2
        x_pad_after = x_diff // 2 + 1

    if y_diff % 2 == 0:
        y_pad_before = y_diff // 2
        y_pad_after = y_diff // 2
    else:
        y_pad_before = y_diff // 2
        y_pad_after = y_diff // 2 + 1

    # Determine padding mode based on padType
    if padType == 'edge':
        pad_mode = 'edge'
    else:
        pad_mode = 'constant'

    # Pad the image symmetrically around its center using the specified padding mode
    if len(image.shape) == 3:
        padded_image = np.pad(image, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (0, 0)), mode=pad_mode)
    else:
        padded_image = np.pad(image, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after)), mode=pad_mode)

    return padded_image

def MakeGrayScale(image):
    """
    Converts a color image to grayscale.

    Parameters:
    Image1 (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The grayscale image.
    """
    if (len(image.shape) >= 3):
            image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    return image

def sobelItUpXY(Image):
    """
    Computes the horizontal and vertical gradients of an image using the Sobel operator.

    Parameters:
    Image (numpy.ndarray): The input image.

    Returns:
    tuple: The horizontal and vertical gradients.
    """
    Image_bool = Image != 0
    sobel_h_full = np.zeros_like(Image)
    sobel_v_full = np.zeros_like(Image)
    if (np.any(Image_bool)):
            sobel_h = cv.Sobel(Image[np.where(np.any(Image_bool, axis=1))[0][0]:, np.where(np.any(Image_bool, axis=0))[0][0]:], ddepth=cv.CV_64F, ksize=-1, dx=1, dy=0).astype('float64') # horizontal gradient
            sobel_v = cv.Sobel(Image[np.where(np.any(Image_bool, axis=1))[0][0]:, np.where(np.any(Image_bool, axis=0))[0][0]:], ddepth=cv.CV_64F, ksize=-1, dx=0, dy=1).astype('float64') # vertical gradient
            sobel_h_full[np.where(np.any(Image_bool, axis=1))[0][0]:, np.where(np.any(Image_bool, axis=0))[0][0]:] = sobel_h
            sobel_v_full[np.where(np.any(Image_bool, axis=1))[0][0]:, np.where(np.any(Image_bool, axis=0))[0][0]:] = sobel_v
    return sobel_h_full, sobel_v_full

def sobelItUp(Image):
    """
    Computes the gradient magnitude of an image using the Sobel operator.

    Parameters:
    Image (numpy.ndarray): The input image.

    Returns:
    numpy.ndarray: The gradient magnitude.
    """
    sobel_h, sobel_v = sobelItUpXY(Image)
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    if ~np.any(magnitude):
            return np.zeros_like(Image)
    return magnitude/np.max(magnitude)

def doPreprocessing(image, calculateWithGrad, calculateWithHess, calculateWithImage, showResult=False):
    """
    Preprocesses an image with optional gradient and Hessian computations.

    Parameters:
    image (numpy.ndarray): The input image.
    calculateWithGrad (bool): Whether to calculate the gradient.
    calculateWithHess (bool): Whether to calculate the Hessian.
    calculateWithImage (bool): Whether to use the original image in the final result.
    showResult (bool): Whether to display the result.

    Returns:
    numpy.ndarray: The preprocessed image.
    """
    # Convert images to grayscale
    image = MakeGrayScale(image)
    image = image.astype(np.float64)
    image /= np.max(image)

    imageCopy = image.copy()

    if calculateWithGrad:
        imageGrad = sobelItUp(image)
    else:
        imageGrad = np.ones_like(image)
    if calculateWithHess:
        dx, dy = sobelItUpXY(image) # Compute gradients
        imageHess = np.stack((*sobelItUpXY(dx), *sobelItUpXY(dy)), axis=-1)
        imageHess = np.sqrt(np.sum(imageHess**2, axis=-1))
        imageHess /= np.max(imageHess)
    else:
        imageHess = np.ones_like(image)

    if not calculateWithImage:
        image = np.ones_like(image)

    if showResult:
        _ ,axs = plt.subplots(4, 1, figsize=(20, 30))
        axs[0].set_title('Pure Images')
        axs[0].imshow(imageCopy)
        axs[1].set_title('Grad Images')
        axs[1].imshow(imageGrad)
        axs[2].set_title('Hess Images')
        axs[2].imshow(imageHess)
        axs[3].set_title('Image to be correlated')
        axs[3].imshow(image)
            
    return image*(imageGrad*imageHess)

def doCorrAndBringBack(Image1, Image2):
    """
    Computes the normalized correlation between two images and reverses the correlation.

    Parameters:
    Image1 (numpy.ndarray): The first input image.
    Image2 (numpy.ndarray): The second input image.

    Returns:
    numpy.ndarray: The correlation result.
    """
    norm_corr = doZeroMeanNormCorr(Image1, Image2)
    return reverseCorr(norm_corr)

def doPaddingAndGetCopies(Image1, Image2, padType): 
    """
    Pads two images and returns padded copies.

    Parameters:
    Image1 (numpy.ndarray): The first input image.
    Image2 (numpy.ndarray): The second input image.
    padType (str): The padding type to use ('edge' or 'constant').

    Returns:
    tuple: The padded images.
    """
    Image1, Image2 = padItUp(Image1, Image2, padType)
    Image1 = padImage(Image1, (Image1.shape[0]*2, Image1.shape[1]*2), padType)
    Image2 = padImage(Image2, (Image2.shape[0]*2, Image2.shape[1]*2), padType)
    return Image1, Image2

def doDIC(Image1Pure:np.ndarray, Image2Pure:np.ndarray, showCorrelation=False, 
           showResult=False, divPosx=-1, divPosy=-1, circVal=-1, padError=0,
             showPadError=False, save_images=False, useGradCorrelation=False, 
             useHessCorrelation=False, useImageCorrelation=False, useImageAsPreProcess=True, useGradAsPreProcess=False, 
             useHessAsPreProcess=False, padType='edge'):
    """
    Performs Digital Image Correlation (DIC) on two images.

    Parameters:
    Image1Pure (numpy.ndarray): The first input image.
    Image2Pure (numpy.ndarray): The second input image.
    showCorrelation (bool): Whether to display the correlation map.
    showResult (bool): Whether to display the result.
    divPosx (int): X-division position for DIC.
    divPosy (int): Y-division position for DIC.
    circVal (int): Circular value for the DIC.
    padError (int): Padding error margin.
    showPadError (bool): Whether to show padding error.
    save_images (bool): Whether to save intermediate images.
    useGradCorrelation (bool): Whether to calculate with gradient.
    useHessCorrelation (bool): Whether to calculate with Hessian.
    useImageCorrelation (bool): Whether to calculate with Hessian.
    useImageAsPreProcess (bool): Whether to use the original image in the final result.
    useGradAsPreProcess (bool): Whether to merge image with gradient.
    useHessAsPreProcess (bool): Whether to merge image with Hessian.
    padType (str): The padding type to use ('edge' or 'constant').

    Returns:
    tuple: Displacement coordinate, X and Y
    displacement statistics
    merged image
    figure
    """

    Image1copy = MakeGrayScale(Image1Pure.copy())
    Image2copy = MakeGrayScale(Image2Pure.copy())
    iterations = 1

    if showResult:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle('Initial Images')
            ax1.imshow(Image1Pure)
            ax2.imshow(Image2Pure)


    if (len(Image1Pure.shape) == 3):
            iterations = 3
    for i in range(iterations):
            Image1 = Image1Pure[:,:,i].copy().astype(np.float64)
            Image2 = Image2Pure[:,:,i].copy().astype(np.float64)

            Image1 /= np.max(Image1)
            Image2 /= np.max(Image2)
            
            # Check if padError is provided as an array
            showPadError = showPadError and type(padError) != type(0)
            
            Image1 = doPreprocessing(Image1, useGradAsPreProcess, useHessAsPreProcess, useImageAsPreProcess, False)
            Image2 = doPreprocessing(Image2, useGradAsPreProcess, useHessAsPreProcess, useImageAsPreProcess, False)

            Image1_grad = doPreprocessing(Image1,True,False,False,False)
            Image2_grad = doPreprocessing(Image2,True,False,False,False)

            Image1_hess = doPreprocessing(Image1,False,True,False,False)
            Image2_hess = doPreprocessing(Image2,False,True,False,False)

            Image1, Image2 = doPaddingAndGetCopies(Image1, Image2, padType)
            Image1, Image1copy = padItUp(Image1, Image1copy, padType)
            Image2, Image2copy = padItUp(Image2, Image2copy, padType)
            Image1_grad, Image2_grad = doPaddingAndGetCopies(Image1_grad, Image2_grad,padType)
            Image1_hess, Image2_hess = doPaddingAndGetCopies(Image1_hess, Image2_hess,padType)

            if i == 0:
                real_corr = np.ones_like(Image1.copy()).astype(np.complex128)

            # Compute normalized correlations
            if useImageCorrelation:
                real_corr *= doCorrAndBringBack(Image1, Image2)
            if useGradCorrelation:
                real_corr *= doCorrAndBringBack(Image1_grad, Image2_grad)
            if useHessCorrelation:
                real_corr *= doCorrAndBringBack(Image1_hess, Image2_hess)
    
    real_corr = np.abs(real_corr)
    (delta_x, delta_y), findmax, merged, fig = plot_dic_plots(Image1, Image1copy, Image2copy, real_corr, showCorrelation, showResult, showPadError, divPosx, divPosy, circVal, padError, save_images)

    # Return displacement, displacement statistics, merged image, and figure (if required)
    return (-delta_x, -delta_y), np.max(np.abs(findmax)), findmax, merged, fig if (showCorrelation or showResult) else None


def plot_dic_plots(Image1, Image1copy, Image2copy, real_corr, showCorrelation, showResult, showPadError, divPosx, divPosy, circVal, padError, save_images=False):
    """
    Generate and display DIC (Digital Image Correlation) plots including correlation maps,
    displacement calculation, and merged images with optional display configurations.

    Parameters:
    - Image1 (numpy.ndarray): Reference image for correlation.
    - Image1copy (numpy.ndarray): Copy of the reference image.
    - Image2copy (numpy.ndarray): Copy of the target image for correlation.
    - real_corr (numpy.ndarray): 2D array representing the correlation coefficients.
    - showCorrelation (bool): Flag to display the correlation map.
    - showResult (bool): Flag to display the resulting merged image.
    - showPadError (bool): Flag to display padding errors.
    - divPosx (int): X-coordinate position for correlation search.
    - divPosy (int): Y-coordinate position for correlation search.
    - circVal (int): Radius of the circular area around the maximum correlation point.
    - padError (numpy.ndarray): Padding error array for visualization.
    - save_images (bool): Flag to save the plots instead of displaying them.

    Returns:
    - Tuple containing:
      - (delta_x, delta_y) (tuple of ints): Displacement in x and y directions.
      - findmax (numpy.ndarray): Array representing the highest correlation region.
      - merged (numpy.ndarray): Merged image showing Image1 and Image2 positions.
      - fig (matplotlib.figure.Figure or None): Figure object if plots were generated, None otherwise.
    """
    # Find maximum correlation
    shouldShowWholeCorrelation = divPosx < 0 or divPosy < 0 or circVal < 0
    corrXleft = np.max([divPosx - circVal, 0])
    corrXright = np.min([divPosx + circVal, real_corr.shape[0]])
    corrYdown = np.max([divPosy - circVal, 0])
    corrYup = np.min([divPosy + circVal, real_corr.shape[1]])

    findmax = np.zeros(real_corr.shape).astype(np.cdouble)
    if shouldShowWholeCorrelation:
        findmax = real_corr
    else:
        findmax[real_corr.shape[0] - corrXright:real_corr.shape[0] - corrXleft, corrYdown:corrYup] = 1
        findmax = findmax
        findmax *= real_corr

    DIC_pos = np.unravel_index(np.argmax(findmax, axis=None), findmax.shape)
    
    # Plotting setup
    if showCorrelation:
        if showResult and showPadError:
            print(DIC_pos)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(Image1.shape[0] / 85, Image1.shape[1] / 85))
        elif showResult or showPadError:
            print(DIC_pos)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 20))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(Image1.shape[0] / 85, Image1.shape[1] / 85))
    elif showResult:
        print(DIC_pos)
        fig, ax1 = plt.subplots(1, 1, figsize=(Image1.shape[0] / 85, Image1.shape[1] / 85))

    # Plot correlation map
    if showCorrelation:
        ax1.set_title('Corr map, log\'ed')
        pcm = ax1.pcolormesh(np.log(np.abs(real_corr[:,::-1])))
        fig.colorbar(pcm, ax=ax1)
        ax1.plot(real_corr.shape[1] - DIC_pos[1], DIC_pos[0], 'r.')
        if not shouldShowWholeCorrelation:
            ax1.plot(divPosy, real_corr.shape[0] - divPosx, 'r*')
            ax1.plot(corrYdown, real_corr.shape[0] - corrXright, 'ko')
            ax1.plot(corrYdown, real_corr.shape[0] - corrXleft, 'bo')
            ax1.plot(corrYup, real_corr.shape[0] - corrXright, 'go')
            ax1.plot(corrYup, real_corr.shape[0] - corrXleft, 'yo')
        if showPadError:
            ax2.set_title('Pad_error')
            real_pad_error = np.fft.ifftshift(np.fft.ifft2(padError))
            pcm2 = ax2.pcolormesh(np.abs(real_pad_error[:, ::-1]))
            fig.colorbar(pcm2, ax=ax2)

    # Calculate displacement in x and y directions
    delta_x = DIC_pos[0] - real_corr.shape[0] // 2
    delta_y = real_corr.shape[1]// 2 - DIC_pos[1]

    # Create merged image
    merged = np.zeros((real_corr.shape[0] + 2 * abs(delta_x), real_corr.shape[1] + 2 * abs(delta_y)))

    # Copy Image2 to appropriate position in the merged image
    merged[abs(delta_x):abs(delta_x)+real_corr.shape[0], abs(delta_y):abs(delta_y)+real_corr.shape[1]] = Image2copy

    # Adjust position and add Image1 to the merged image
    merged[abs(delta_x)-delta_x:abs(delta_x)-delta_x+real_corr.shape[0],
           abs(delta_y)+delta_y:abs(delta_y)+delta_y+real_corr.shape[1]] -= padImage(Image1copy, real_corr.shape, 'constant')
    merged = np.abs(merged)

    # Plot merged image
    if showResult:
        print((delta_x, delta_y))
        if showCorrelation and showPadError:
            ax3.set_title('Merged image')
            ax3.imshow(merged)
            ax3.plot([real_corr.shape[1] / 2 + abs(delta_y) + delta_y], [real_corr.shape[0] / 2 + abs(delta_x) - delta_x], 'bo')
            ax3.plot([real_corr.shape[1] / 2 + abs(delta_y)], [real_corr.shape[0] / 2 + abs(delta_x)], 'ro')
        elif showCorrelation:
            ax2.set_title('Merged image')
            ax2.imshow(merged)
            line1, =ax2.plot([real_corr.shape[1] / 2 + abs(delta_y) + delta_y], [real_corr.shape[0] / 2 + abs(delta_x) - delta_x], 'bo', label='Image 1')
            line2, =ax2.plot([real_corr.shape[1] / 2 + abs(delta_y)], [real_corr.shape[0] / 2 + abs(delta_x)], 'ro', label='Image 2')
            ax2.legend(handles=[line1, line2])
        else:
            ax1.set_title('Merged image')
            ax1.imshow(merged)
            line1, = ax1.plot([real_corr.shape[1] / 2 + abs(delta_y) + delta_y], [real_corr.shape[0] / 2 + abs(delta_x) - delta_x], 'bo', label='Image 1')
            line2, = ax1.plot([real_corr.shape[1] / 2 + abs(delta_y)], [real_corr.shape[0] / 2 + abs(delta_x)], 'ro', label='Image 2')
            ax1.legend(handles=[line1, line2])

    # Show or save the plot
    if (showCorrelation or showResult) and not save_images:
        plt.show()
    else:
        plt.close()
    
    return (delta_x, delta_y), findmax, merged, fig if (showCorrelation or showResult) else None

def doZeroMeanNormCorr(Image1, Image2):
    """
    Computes the zero mean normalized cross-correlation of two images.

    Parameters:
    image1 (numpy.ndarray): The first input image.
    image2 (numpy.ndarray): The second input image.

    Returns:
    numpy.ndarray: The normalized correlation result.
    """
    Image1, Image2 = padItUp(Image1, Image2)

    ftImage1 = np.fft.fftshift(np.fft.fftn(Image1))
    ftImage2 = np.fft.fftshift(np.fft.fftn(Image2))

    ftImage1 -= np.mean(ftImage1)
    ftImage2 -= np.mean(ftImage2)

    corr = ftImage1 * np.conj(ftImage2)
    corr /= np.max(np.abs(corr))
    return (corr/np.abs(corr))

def overlapRegion(image1, image2, delta_x, delta_y):
    """
    Returns the overlap region of two images.
    
    Parameters:
        image1 (2D array): The first input image.
        image2 (2D array): The second input image.
        delta_x (int): The horizontal displacement between the two images.
        delta_y (int): The vertical displacement between the two images.
        
    Returns:
        overlap1 (2D array): Overlapping region of image1.
        overlap2 (2D array): Overlapping region of image2.
    """         
    image1, image2 = padItUp(image1+1, image2+1, 'constant')
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    # Calculate the overlap region coordinates
    start_x = np.max([0, delta_x])
    end_x = np.min([width1, width2 + delta_x])
    start_y = np.max([0, delta_y])
    end_y = np.min([height1, height2 + delta_y])
    
    # Extract the overlap region from both images
    overlap1 = image1[start_y:end_y,start_x:end_x]
    overlap2 = image2[start_y-delta_y:end_y-delta_y,start_x-delta_x:end_x-delta_x]

    bool1 = overlap1 > 0
    bool2 = overlap2 > 0

    if (np.where(bool1)[0].size < np.where(bool2)[0].size and np.where(bool1)[0].size > 0):
        overlap1 = overlap1[np.any(bool1[:,:,0], axis=1),:,:][:,np.any(bool1[:,:,0], axis=0),:]
        overlap2 = overlap2[np.any(bool1[:,:,0], axis=1),:,:][:,np.any(bool1[:,:,0], axis=0),:]
    elif (np.where(bool2)[0].size > 0): 
        overlap1 = overlap1[np.any(bool2[:,:,0], axis=1),:,:][:,np.any(bool2[:,:,0], axis=0),:]
        overlap2 = overlap2[np.any(bool2[:,:,0], axis=1),:,:][:,np.any(bool2[:,:,0], axis=0),:]  
    
    return overlap1-1, overlap2-1

def reverseCorr(corr):
    return np.fft.ifftshift(np.fft.ifftn(corr))

def interpolateQuiver(X, Y, UV, order=3):
       x = X.ravel()
       y = Y.ravel()

       U = UV[1]
       V = UV[0]

       basis = getBasis(y, x, order)
       A = np.vstack(np.array([basis])).T
       
       sUV = np.linalg.lstsq(A, np.array([U.ravel(),V.ravel()]).T, rcond=None)[0]
       return sUV,basis

def getBasis(x, y, max_order=4):
    """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
    basis = []
    for i in range(max_order+1):
        for j in range(max_order - i +1):
            basis.append(x**j * y**i)
    return basis

def getFit(X, Y, poly:np.ndarray, basis=None, o=3):
    """
    Fits a polynomial model to the given data using a basis function.

    Parameters:
    X (numpy.ndarray): The input data for the independent variable.
    Y (numpy.ndarray): The input data for the dependent variable.
    poly (numpy.ndarray): Coefficients of the polynomial to be fitted, where each row corresponds to polynomial coefficients for a different term.
    basis (callable, optional): The fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc.
    o (int, optional): The order of the polynomial if `getBasis` is used. Default is 3.

    Returns:
    tuple: A tuple containing:
        - U (numpy.ndarray): The fitted U values reshaped to the shape of X.
        - V (numpy.ndarray): The fitted V values reshaped to the shape of X.
    """
    x = X.ravel()
    y = Y.ravel()
    polyU = poly[:,0]
    polyV = poly[:,1]
    if basis is None:
        basis = np.array(getBasis(x, y, o))
    U = polyU @ basis
    V = polyV @ basis
    U = U.reshape(*X.shape)
    V = V.reshape(*X.shape)
    return U,V

def imageError(oldDis, newDis):
    """
    Computes the mean error between the old displacements and the new displacements.

    Parameters:
    oldDis (numpy.ndarray): The array of original displacements.
    newDis (numpy.ndarray): The array of new displacements.

    Returns:
    float: The mean error computed as the mean norm of the difference between old and new displacements.
    """
    return np.mean(np.linalg.norm(oldDis - newDis, axis = 0))

def warp_image(image, disp_x, disp_y):
    """
    Warp an image according to given displacement fields.

    Args:
    - image (numpy.ndarray): Input image array of shape (height, width, channels)
    - disp_x (numpy.ndarray): Displacement field for X direction, shape (height, width)
    - disp_y (numpy.ndarray): Displacement field for Y direction, shape (height, width)

    Returns:
    - warped_image (numpy.ndarray): Warped image array of shape (height, width, channels)
    """
    height, width, channels = image.shape

    # Generate grid coordinates
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Add displacement and clip to stay within image boundaries
    displaced_coords_x = np.clip(x_coords + disp_x, 0, width - 1)
    displaced_coords_y = np.clip(y_coords + disp_y, 0, height - 1)

    # Stack displaced coordinates
    displaced_coords = np.stack((displaced_coords_y, displaced_coords_x), axis=-1)

    # Interpolate image values at displaced coordinates
    warped_image = np.zeros_like(image)
    for c in range(channels):
        # Transpose and reshape the coordinate array
        coords = displaced_coords.reshape((height * width, 2)).T
        # Interpolate image values at displaced coordinates
        warped_channel = map_coordinates(image[..., c], coords, order=5, mode='constant', cval=0)
         # Reshape the interpolated values back to image shape
        warped_image[..., c] = warped_channel.reshape((height, width))

    return warped_image

def showAndCompareWarped(image1:np.ndarray, image2:np.ndarray, warpedImage):
    """
    Displays and compares the reference, warped, and original images.

    Parameters:
    image1 (numpy.ndarray): The first input image (original image).
    image2 (numpy.ndarray): The second input image (reference image).
    warpedImage (numpy.ndarray): The warped version of the first input image.

    Returns:
    Void
    """
    im1, im2 = padItUp(image2,warpedImage, padType='constant')
    plt.figure(figsize=(10, 10))
    plt.imshow(im1[:,:,0])
    plt.title('Reference Image')
    plt.figure(figsize=(10, 10))
    plt.imshow(im2[:,:,0])
    plt.title('Warped Image')
    plt.figure(figsize=(10, 10))

    im1_bool = im1 != 0
    im2_bool = im2 != 0
    use_bool = im1_bool if im2_bool.sum() > im1_bool.sum() else im2_bool
    cMax = 255 if (isinstance(im1[0], int)) else 1
    plt.title('New diff : $|I_{ref} - I_{warp}|$ = ' + str(np.mean(np.abs((im1[use_bool]/cMax)-(im2[use_bool]/cMax)))))
    newDiff = (0.5*im2+0.5*im1)/np.max(np.abs(0.5*im2+0.5*im1))
    plt.imshow(newDiff)
    plt.show()

    im3, im4 = padItUp(image1, image2, 'constant')
    im3_bool = im3 != 0
    im4_bool = im4 != 0
    use_bool = im4_bool if im3_bool.sum() > im4_bool.sum() else im3_bool
    plt.figure(figsize=(10, 10))
    plt.title('Original : $|I_{ref} - I_{org}|$ = ' + str(np.mean(np.abs((im3[use_bool]/cMax)-(im4[use_bool]/cMax)))))
    orgDiff = (0.5*im3+0.5*im4)/np.max(np.abs(0.5*im3+0.5*im4))
    plt.imshow(orgDiff)
    plt.show()

def findTriangle(number, i=[0], j=1, restart=True):
    if restart:
        i = [0]
        j = 1
    if number <= sum(i):
        return j-2
    i.append(j)
    return findTriangle(number, i, j+1, False)

def mapOldOrderToNewOrder(cOld, oldOrder, newOrder):
    y = np.array([[0,0,0],[1,1,1],[2,2,2]])
    x = np.array([[0,1,2],[0,1,2],[0,1,2]])
    basisOld = getBasis(x, y, oldOrder)
    basisNew = getBasis(x, y, newOrder)
    cNew = np.zeros((len(basisNew)))
    for i,baseOld in enumerate(basisOld):
        for j,baseNew in enumerate(basisNew):
            if np.all(baseOld == baseNew):
                cNew[j] = cOld[i]
                break 
    return cNew

def error_minimization(UV, dicClass, o, uvShape):
    """
    Minimizes the error between the given displacements and the fitted displacements.

    Parameters:
    UV (numpy.ndarray): The array of displacements to be reshaped and fitted.
    dicClass (object): An object containing displacement coordinates and internal displacement data.
    o (int): The order of the polynomial for fitting.
    basis (callable): A basis function to apply to the input data.
    uvShape (tuple): The shape to reshape the UV array into.

    Returns:
    float: The error between the internal displacement and the fitted displacement.
    """
    UV = np.reshape(UV, uvShape)
    U,V = getFit(dicClass.discplacementsCoordinates[1], dicClass.discplacementsCoordinates[0], UV, o = o)
    return (imageError(np.array([dicClass.internalDisplacement[0],dicClass.internalDisplacement[1]]), 
                       np.array([V, U])))


def error_minimizationUnaffected(UV, dicClass, o, uvShape):
    """
    Minimizes the error between the given displacements and the fitted displacements.

    Parameters:
    UV (numpy.ndarray): The array of displacements to be reshaped and fitted.
    dicClass (object): An object containing displacement coordinates and internal displacement data.
    o (int): The order of the polynomial for fitting.
    basis (callable): A basis function to apply to the input data.
    uvShape (tuple): The shape to reshape the UV array into.

    Returns:
    float: The error between the internal displacement and the fitted displacement.
    """
    UV = np.reshape(UV, uvShape)
    U,V = getFit(dicClass.discplacementsCoordinatesUnaffected[1], dicClass.discplacementsCoordinatesUnaffected[0], UV, o = o)
    return (imageError(np.array([dicClass.internalDisplacementUnaffected[0],dicClass.internalDisplacementUnaffected[1]]), 
                       np.array([V, U])))

def findBestDIC(image1, image2, a=0):
    def process_combination(i, comb):
        try:
            if not np.any(comb):
                print('Combination #%d failed' %i)
                return None, None, None
            innerDIC = DicClass(showDicResults=False, showDicCorrelation=False,
                                        useImageAsPreProcess=comb[0],
                                        useGradAsPreProcess=comb[1],
                                        useHessAsPreProcess=comb[2],
                                        useImageCorrelation=comb[3],
                                        useGradCorrelation=comb[4],
                                        useHessCorrelation=comb[5])
            innerDIC.defineImage1(image1)
            innerDIC.defineImage2(image2)
            innerDIC.doSimpleExposureAdjustment()
            innerDIC.doInternalDIC()
            innerDIC.setOverlappingImages()

            innerDIC_bool = innerDIC.overlap_quality != 0
            return (innerDIC, i, innerDIC_bool)
        except:
            print('Combination #%d failed' %i)
            return None, None, None

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_combination, i, comb) for i, comb in enumerate(combinations)]
        chosen_combo = None

        for future in as_completed(futures):
            innerDIC, i, innerDIC_bool = future.result()

            if innerDIC is None:
                continue

            if type(a) is not int:
                a_bool = a.overlap_quality != 0
            if type(a) is int: 
                overlap_quality_diff = 0
            else:
                overlap_quality_diff = np.mean(a.overlap_quality[a_bool]) - np.mean(innerDIC.overlap_quality[innerDIC_bool])
            if type(a) is int or overlap_quality_diff > 0:
                a = innerDIC
                chosen_combo = i
                abs_diff = np.abs(overlap_quality_diff)

    executor.shutdown()
    if a == 0:
        return a, combinations[0], 0

    return a, combinations[chosen_combo], abs_diff

def getInternalReport(dicObject, saveFig=False, showZeros=False):
        print('-------------total-------------')
        print('total: %d' %(dicObject.chosenComboMat.size//6))

        zerosies = np.sum(dicObject.chosenComboMat, axis=-1) == 0
        zeros = np.sum(~dicObject.chosenComboMat[zerosies,0])
        print('-------------zeros-------------')
        print('zeros: %d' %zeros)

        twosies = np.sum(dicObject.chosenComboMat, axis=-1) == 2
        
        ImWImCor = np.sum(np.array(dicObject.chosenComboMat[twosies,0].astype(int)+dicObject.chosenComboMat[twosies,3].astype(int)) == 2)
        ImWGCor =  np.sum(np.array(dicObject.chosenComboMat[twosies,0].astype(int)+dicObject.chosenComboMat[twosies,4].astype(int)) == 2)
        ImWHCor =  np.sum(np.array(dicObject.chosenComboMat[twosies,0].astype(int)+dicObject.chosenComboMat[twosies,5].astype(int)) == 2)
        GWImCor =  np.sum(np.array(dicObject.chosenComboMat[twosies,1].astype(int)+dicObject.chosenComboMat[twosies,3].astype(int)) == 2)
        GWGCor =   np.sum(np.array(dicObject.chosenComboMat[twosies,1].astype(int)+dicObject.chosenComboMat[twosies,4].astype(int)) == 2)
        GWHCor =   np.sum(np.array(dicObject.chosenComboMat[twosies,1].astype(int)+dicObject.chosenComboMat[twosies,5].astype(int)) == 2)
        HWImCor =  np.sum(np.array(dicObject.chosenComboMat[twosies,2].astype(int)+dicObject.chosenComboMat[twosies,3].astype(int)) == 2)
        HWGCor =   np.sum(np.array(dicObject.chosenComboMat[twosies,2].astype(int)+dicObject.chosenComboMat[twosies,4].astype(int)) == 2)
        HWHCor =   np.sum(np.array(dicObject.chosenComboMat[twosies,2].astype(int)+dicObject.chosenComboMat[twosies,5].astype(int)) == 2)

        print('-------------twos-------------')
        print('ImWImCor: %d' %ImWImCor)
        print('ImWGCor: %d' %ImWGCor)
        print('ImWHCor: %d' %ImWHCor)
        print('GWImCor: %d' %GWImCor)
        print('GWGCor: %d' %GWGCor)
        print('GWHCor: %d' %GWHCor)
        print('HWImCor: %d' %HWImCor)
        print('HWGCor: %d' %HWGCor)
        print('HWHCor: %d' %HWHCor)
        print('Sub total: %d' %np.sum(twosies))


        threesies = np.sum(dicObject.chosenComboMat, axis=-1) == 3
        ImWImGCorr = np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)) == 3)
        ImWImHCorr = np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        ImWGHCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        GWImGCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)) == 3)
        GWImHCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        GWGHCorr =   np.sum(np.array(dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        HWImGCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)) == 3)
        HWImHCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        HWGHCorr =   np.sum(np.array(dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        ImGWImCorr = np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)) == 3)
        ImGWGCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)) == 3)
        ImGWHCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        ImHWImCorr = np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)) == 3)
        ImHWGCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)) == 3)
        ImHWHCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,0].astype(int)+dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)
        GHWImCorr =  np.sum(np.array(dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,3].astype(int)) == 3)
        GHWGCorr =   np.sum(np.array(dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,4].astype(int)) == 3)
        GHWHCorr =   np.sum(np.array(dicObject.chosenComboMat[threesies,1].astype(int)+dicObject.chosenComboMat[threesies,2].astype(int)+dicObject.chosenComboMat[threesies,5].astype(int)) == 3)

        print('------------threes------------')
        print('ImWImGCorr: %d' %ImWImGCorr)
        print('ImWImHCorr: %d' %ImWImHCorr)
        print('ImWGHCorr: %d' %ImWGHCorr)

        print('GWImGCorr: %d' %GWImGCorr)
        print('GWImHCorr: %d' %GWImHCorr)
        print('GWGHCorr: %d' %GWGHCorr)

        print('HWImGCorr: %d' %HWImGCorr)
        print('HWImHCorr: %d' %HWImHCorr)
        print('HWGHCorr: %d' %HWGHCorr)

        print('ImGWImCorr: %d' %ImGWImCorr)
        print('ImGWGCorr: %d' %ImGWGCorr)
        print('ImGWHCorr: %d' %ImGWHCorr)

        print('ImHWImCorr: %d' %ImHWImCorr)
        print('ImHWGCorr: %d' %ImHWGCorr)
        print('ImHWHCorr: %d' %ImHWHCorr)

        print('GHWImCorr: %d' %GHWImCorr)
        print('GHWGCorr: %d' %GHWGCorr)
        print('GHWHCorr: %d' %GHWHCorr)
        
        print('Sub total: %d' %np.sum(threesies))


        foursies = np.sum(dicObject.chosenComboMat, axis=-1) == 4
        missingImNotImCorr = np.sum((~dicObject.chosenComboMat[foursies,0]).astype(int)+(~dicObject.chosenComboMat[foursies,3]).astype(int) == 2)
        missingImNotGCorr =  np.sum((~dicObject.chosenComboMat[foursies,0]).astype(int)+(~dicObject.chosenComboMat[foursies,4]).astype(int) == 2)
        missingImNotHCorr =  np.sum((~dicObject.chosenComboMat[foursies,0]).astype(int)+(~dicObject.chosenComboMat[foursies,5]).astype(int) == 2)
        missingGNotImCorr =  np.sum((~dicObject.chosenComboMat[foursies,1]).astype(int)+(~dicObject.chosenComboMat[foursies,3]).astype(int) == 2)
        missingGNotGCorr =   np.sum((~dicObject.chosenComboMat[foursies,1]).astype(int)+(~dicObject.chosenComboMat[foursies,4]).astype(int) == 2)
        missingGNotHCorr =   np.sum((~dicObject.chosenComboMat[foursies,1]).astype(int)+(~dicObject.chosenComboMat[foursies,5]).astype(int) == 2)
        missingHNotImCorr =  np.sum((~dicObject.chosenComboMat[foursies,2]).astype(int)+(~dicObject.chosenComboMat[foursies,3]).astype(int) == 2)
        missingHNotGCorr =   np.sum((~dicObject.chosenComboMat[foursies,2]).astype(int)+(~dicObject.chosenComboMat[foursies,4]).astype(int) == 2)
        missingHNotHCorr =   np.sum((~dicObject.chosenComboMat[foursies,2]).astype(int)+(~dicObject.chosenComboMat[foursies,5]).astype(int) == 2)

        missingHmissingG =   np.sum((~dicObject.chosenComboMat[foursies,2]).astype(int)+(~dicObject.chosenComboMat[foursies,1]).astype(int) == 2)
        missingHmissingIm =  np.sum((~dicObject.chosenComboMat[foursies,2]).astype(int)+(~dicObject.chosenComboMat[foursies,0]).astype(int) == 2)
        missingImmissingG =  np.sum((~dicObject.chosenComboMat[foursies,0]).astype(int)+(~dicObject.chosenComboMat[foursies,1]).astype(int) == 2)
        NotHCorrNotGCorr =   np.sum((~dicObject.chosenComboMat[foursies,4]).astype(int)+(~dicObject.chosenComboMat[foursies,5]).astype(int) == 2)
        NotHCorrNotImCorr =  np.sum((~dicObject.chosenComboMat[foursies,3]).astype(int)+(~dicObject.chosenComboMat[foursies,5]).astype(int) == 2)
        NotImCorrNotGCorr =  np.sum((~dicObject.chosenComboMat[foursies,3]).astype(int)+(~dicObject.chosenComboMat[foursies,4]).astype(int) == 2)

        print('------------fours-------------')
        print('missingImNotImCorr: %d' %missingImNotImCorr)
        print('missingImNotGCorr: %d' %missingImNotGCorr)
        print('missingImNotHCorr: %d' %missingImNotHCorr)

        print('missingGNotImCorr: %d' %missingGNotImCorr)
        print('missingGNotGCorr: %d' %missingGNotGCorr)
        print('missingGNotHCorr: %d' %missingGNotHCorr)

        print('missingHNotImCorr: %d' %missingHNotImCorr)
        print('missingHNotGCorr: %d' %missingHNotGCorr)
        print('missingHNotHCorr: %d' %missingHNotHCorr)

        print('missingHmissingG: %d' %missingHmissingG)
        print('missingHmissingIm: %d' %missingHmissingIm)
        print('missingImmissingG: %d' %missingImmissingG)

        print('NotHCorrNotGCorr: %d' %NotHCorrNotGCorr)
        print('NotHCorrNotImCorr: %d' %NotHCorrNotImCorr)
        print('NotImCorrNotGCorr: %d' %NotImCorrNotGCorr)
        
        print('Sub total: %d' %np.sum(foursies))
        


        fivesies = np.sum(dicObject.chosenComboMat, axis=-1) == 5
        missingIm = np.sum(~dicObject.chosenComboMat[fivesies,0])
        missingGrad = np.sum(~dicObject.chosenComboMat[fivesies,1])
        missingHess = np.sum(~dicObject.chosenComboMat[fivesies,2])
        NotImCorr =  np.sum(~dicObject.chosenComboMat[fivesies,3])
        NotGCorr =  np.sum(~dicObject.chosenComboMat[fivesies,4])
        NotHCorr =  np.sum(~dicObject.chosenComboMat[fivesies,5])

        print('------------fives-------------')
        print('missingIm: %d' %missingIm)
        print('missingGrad: %d' %missingGrad)
        print('missingHess: %d' %missingHess)
        print('NotImCorr: %d' %NotImCorr)
        print('NotGCorr: %d' %NotGCorr)
        print('NotHCorr: %d' %NotHCorr)
        
        print('Sub total: %d' %np.sum(fivesies))

        allsies = np.sum(dicObject.chosenComboMat, axis=-1) == 6
        alls = np.sum(allsies)
        print('------------alls-------------')
        print('alls: %d' %alls)

        fig, ax1 = plt.subplots(figsize=(11, 11), layout='constrained')
        values = [zeros,
                  ImWImCor,#
                  ImWGCor,#
                  ImWHCor,#
                  GWImCor,# 
                  GWGCor,#
                  GWHCor,#
                  HWImCor,# 
                  HWGCor,#
                  HWHCor,#
                  ImWImGCorr,#
                  ImWImHCorr,#
                  ImWGHCorr,#
                  GWImGCorr,#
                  GWImHCorr,#
                  GWGHCorr,#
                  HWImGCorr,# 
                  HWImHCorr,#
                  HWGHCorr,#
                  ImGWImCorr,#
                  ImGWGCorr,#
                  ImGWHCorr,#
                  ImHWImCorr,#
                  ImHWGCorr,#
                  ImHWHCorr,#
                  GHWImCorr,#
                  GHWGCorr,#
                  GHWHCorr,#
                  missingImNotImCorr,#
                  missingImNotGCorr,#
                  missingImNotHCorr,#
                  missingGNotImCorr,#
                  missingGNotGCorr,#
                  missingGNotHCorr,#
                  missingHNotImCorr,#
                  missingHNotGCorr,#
                  missingHNotHCorr,#
                  missingHmissingG,#
                  missingHmissingIm,#
                  missingImmissingG,#
                  NotHCorrNotGCorr,#
                  NotHCorrNotImCorr,#
                  NotImCorrNotGCorr,#
                  missingIm,
                  missingGrad,
                  missingHess,
                  NotImCorr,
                  NotGCorr,
                  NotHCorr,
                  alls#, 
                  ][::-1]
        labels = ['Uncomparable picture',
                 '$F(I)$', 
                 '$F(|dI/dx|)$',
                 '$F(|d^2I/dx^2|)$',
                 '$F(G)$', 
                 '$F(|dG/dx|)$',
                 '$F(|d^2G/dx^2|)$',
                 '$F(H)$', 
                 '$F(|dH/dx|)$',
                 '$F(|d^2H/dx^2|)$',
                 #
                 '$F(I) \cdot F(|dI/dx|)$',
                 '$F(I) \cdot F(|d^2I/dx^2|)$',
                 '$F(|dI/dx|) \cdot F(|d^2I/dx^2|)$',
                 '$F(G) \cdot F(|dG/dx|)$',
                 '$F(G) \cdot F(|d^2G/dx^2|)$',
                 '$F(|dG/dx|) \cdot F(|d^2G/dx^2|)$',
                 '$F(H) \cdot F(|dH/dx|)$',
                 '$F(H) \cdot F(|d^2H/dx^2|)$',
                 '$F(|dH/dx|) \cdot F(|d^2H/dx^2|)$',
                 #
                 '$F(I \cdot G)$', 
                 '$F(|d(I \cdot G)/dx|)$',
                 '$F(|d^2(I \cdot G))/dx^2|)$',
                 '$F(I \cdot H)$', 
                 '$F(|d(I \cdot H)/dx|)$',
                 '$F(|d^2(I \cdot H))/dx^2|)$',
                 '$F(G \cdot H)$', 
                 '$F(|d(G \cdot H)/dx|)$',
                 '$F(|d^2(G \cdot H))/dx^2|)$',
                 #
                 '$F(|d(G \cdot H)/dx|) \cdot F(|d^2(G \cdot H))/dx^2|)$',
                 '$F(G \cdot H) \cdot F(|d^2(G \cdot H))/dx^2|)$', 
                 '$F(G \cdot H) \cdot F(|d(G \cdot H)/dx|)$',
                 '$F(|d(I \cdot H)/dx|) \cdot F(|d^2(I \cdot H))/dx^2|)$',
                 '$F(I \cdot H) \cdot F(|d^2(I \cdot H))/dx^2|)$', 
                 '$F(I \cdot H) \cdot F(|d(I \cdot H)/dx|)$',
                 '$F(|d(I \cdot G)/dx|) \cdot F(|d^2(I \cdot G))/dx^2|)$',
                 '$F(I \cdot G) \cdot F(|d^2(I \cdot G))/dx^2|)$', 
                 '$F(I \cdot G) \cdot F(|d(I \cdot G)/dx|)$',
                 #
                 '$F(I) \cdot F(|dI/dx|) \cdot F(|d^2I/dx^2|)$',
                 '$F(G) \cdot F(|dG/dx|) \cdot F(|d^2G/dx^2|)$',
                 '$F(H) \cdot F(|dH/dx|) \cdot F(|d^2H/dx^2|)$',
                 '$F(I \cdot G \cdot H)$', 
                 '$F(|d(I \cdot G \cdot H)/dx|)$',
                 '$F(|d^2(I \cdot G \cdot H)/dx^2|)$',
                 '$F(G \cdot H) \cdot F(|d(G \cdot H)/dx|) \cdot F(|d^2(G \cdot H))/dx^2|)$',
                 '$F(I \cdot H) \cdot F(|d(I \cdot H)/dx|) \cdot F(|d^2(I \cdot H))/dx^2|)$',
                 '$F(I \cdot G) \cdot F(|d(I \cdot G)/dx|) \cdot F(|d^2(I \cdot G))/dx^2|)$',
                 '$F(|d(I \cdot G \cdot H)/dx|) \cdot F(|d^2(I \cdot G \cdot H)/dx^2|)$',
                 '$F(I \cdot G \cdot H) \cdot F(|d^2(I \cdot G \cdot H)/dx^2|)$',
                 '$F(I \cdot G \cdot H) \cdot F(|d(I \cdot G \cdot H)/dx|)$',
                 '$F(I \cdot G \cdot H) \cdot F(|d(I \cdot G \cdot H)/dx|) \cdot F(|d^2(I \cdot G \cdot H)/dx^2|)$'#,
                 #'Total'
                 ][::-1]
        
        if not showZeros:
            values.pop(-1)
            labels.pop(-1)

        nonZeroValues = np.array(values).nonzero()[0].astype(int).tolist()
        realValues = [values[i] for i in nonZeroValues]
        rects = ax1.barh([labels[i] for i in nonZeroValues], realValues)
        ax1.xaxis.grid(True)
        ax1.bar_label(rects, [str(i) for i in realValues])
        if saveFig:
            fig.savefig('Figures/Internnal Report %s.pdf' %dicObject.timestamp)

        plt.show()

def showInternalResults(dicObject):
        dicObject.showInternalQuiver()
        plt.figure()
        plt.title('Correlation value')
        plt.imshow(dicObject.CValue)
        plt.colorbar()
        plt.figure()
        plt.title('U(x,y)')
        plt.imshow(dicObject.internalDisplacement[0])
        plt.colorbar()
        plt.figure()
        plt.title('V(x,y)')
        plt.imshow(dicObject.internalDisplacement[1])
        plt.colorbar()
        plt.figure()
        plt.title('$|F(x,y)|$')
        plt.imshow(np.sqrt(dicObject.internalDisplacement[1]**2+dicObject.internalDisplacement[0]**2))
        plt.colorbar()
        plt.figure()
        plt.title('Overlap Diff')
        plt.imshow(dicObject.overlapDiff)
        plt.colorbar()

        plt.figure()
        plt.title('Sum of chosen DIC combinations')
        sumofComb = np.sum(dicObject.chosenComboMat,axis = -1)
        plt.imshow(sumofComb)
        plt.colorbar()
        
        # Create a figure
        combNr = np.zeros(dicObject.chosenComboMat.shape[:2])
        for i in range(dicObject.chosenComboMat.shape[0]):
            for j in range(dicObject.chosenComboMat.shape[1]):
                combNr[i, j] = combinations.index(list(dicObject.chosenComboMat[i, j]))

        fig, axs = plt.subplots(1, 1)
        fig.suptitle('Chosen DIC combinations')

        # Plot the image and quiver
        axs.imshow(dicObject.image1)
        axs.quiver(dicObject.discplacementsCoordinates[1], dicObject.discplacementsCoordinates[0], 
                   dicObject.internalDisplacement[1], dicObject.internalDisplacement[0], color='r')

        # Set axis limits and invert y-axis
        axs.axis([np.min(dicObject.internalDisplacement[1]), dicObject.image1.shape[1]+10, 
                  np.min(dicObject.internalDisplacement[0]), dicObject.image1.shape[0]+10])
        axs.invert_yaxis()

        # Add text with a rectangle (bbox) around the index for visibility
        nonZero = np.nonzero(sumofComb)
        coords_x = dicObject.discplacementsCoordinates[1][nonZero]
        coords_y = dicObject.discplacementsCoordinates[0][nonZero]
        nonZeroCombNumbers = combNr[nonZero]

        for i in range(coords_x.shape[0]):
                axs.text(coords_x[i], coords_y[i], int(nonZeroCombNumbers[i]), ha='center', va='center', color='b', fontsize=12,
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))

        plt.figure()
        plt.title('Sum of chosen DIC combinations')
        plt.imshow(dicObject.comboDiffMat)
        plt.colorbar()
