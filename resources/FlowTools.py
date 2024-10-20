# Local
from resources import Tools
from resources.DIC import DicClass

# Math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.ndimage import map_coordinates

# Data
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the sets of Gradient-Processed Correlation (GPC) to try
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
cBarFrac = .045

"""
Pads two images to the same dimensions.

Parameters:
    img1 (numpy.ndarray): The first input image.
    img2 (numpy.ndarray): The second input image.
    padType (str): The padding type to use ('edge' or 'constant').

Returns:
    tuple: The two padded images.
"""
def padImages(img1, img2, padType='edge'):
    # Determine the maximum dimensions of the input images
    max_height = np.max([img1.shape[0], img2.shape[0]])
    max_width = np.max([img1.shape[1], img2.shape[1]])
    target_shape = (max_height, max_width)

    # Pad both images to the target shape
    img1_padded = padImage(img1, target_shape, padType)
    img2_padded = padImage(img2, target_shape, padType)

    return img1_padded, img2_padded

"""
Pads an image to the target shape.

Parameters:
    img (numpy.ndarray): The input image to be padded.
    target_shape (tuple): The desired shape after padding (height, width).
    padType (str): The padding type to use ('edge' or 'constant').

Returns:
    numpy.ndarray: The padded image.
"""
def padImage(img, target_shape, padType='edge'):
    # Calculate padding amounts for each dimension
    x_diff = target_shape[0] - img.shape[0]
    y_diff = target_shape[1] - img.shape[1]
    
    # Determine the amount of padding needed on each side of the image
    x_pad = (x_diff // 2, x_diff // 2 + (x_diff % 2 != 0))
    y_pad = (y_diff // 2, y_diff // 2 + (y_diff % 2 != 0))

    # Pad the image symmetrically around its center using the specified padding mode
    if len(img.shape) == 3:
        return np.pad(img, (x_pad, y_pad, (0,0)), mode=padType)
    else:
        return np.pad(img, (x_pad, y_pad), mode=padType)

"""
Converts a color image to grayscale.

Parameters:
    img1 (numpy.ndarray): The input image.

Returns:
    numpy.ndarray: The grayscale image.
"""
def grayScale(img):
    if (len(img.shape) >= 3):
        img = .299*img[:,:,0] + .587*img[:,:,1] + .114*img[:,:,2]
    return img

"""
Computes the horizontal and vertical gradients of an image using the Sobel operator.

Parameters:
    img (numpy.ndarray): The input image.

Returns:
    tuple: The horizontal and vertical gradients.
"""
def sobelXY(img):
    nonZeroRows, nonZeroCols = np.where(np.any(img, axis=0))[0], np.where(np.any(img, axis=1))[0]
    sobel_h_full, sobel_v_full = np.zeros_like(img), np.zeros_like(img)
    if (len(nonZeroRows) > 0):
        sobel_h = cv.Sobel(img[nonZeroCols[0]:,nonZeroRows[0]:], ddepth=cv.CV_64F, ksize=-1, dx=1, dy=0).astype('float64') # horizontal gradient
        sobel_v = cv.Sobel(img[nonZeroCols[0]:,nonZeroRows[0]:], ddepth=cv.CV_64F, ksize=-1, dx=0, dy=1).astype('float64') # vertical gradient
        sobel_h_full[nonZeroCols[0]:,nonZeroRows[0]:] = sobel_h
        sobel_v_full[nonZeroCols[0]:,nonZeroRows[0]:] = sobel_v
    return sobel_h_full, sobel_v_full

"""
Preprocesses an image with optional gradient and Hessian computations.

Parameters:
    img (numpy.ndarray): The input image.
    calculateWithGrad (bool): Whether to calculate the gradient.
    calculateWithHess (bool): Whether to calculate the Hessian.
    calculateWithImage (bool): Whether to use the original image in the final result.
    showResult (bool): Whether to display the result.

Returns:
    numpy.ndarray: The preprocessed image.
"""
def preprocess(img, calculateWithGrad, calculateWithHess, calculateWithImage, showResult=False):
    # Convert images to grayscale
    img = grayScale(img)
    img = img.astype(np.float64)
    img /= np.max(img)

    imageCopy = img.copy()

    if calculateWithGrad:
        sobel_h, sobel_v = sobelXY(img)
        magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        if ~np.any(magnitude):
            imageGrad = np.zeros_like(img)
        imageGrad = magnitude/np.max(magnitude)
    else:
        imageGrad = np.ones_like(img)
    if calculateWithHess:
        dx, dy = sobelXY(img) # Compute gradients
        imageHess = np.stack((*sobelXY(dx), *sobelXY(dy)), axis=-1)
        imageHess = np.sqrt(np.sum(imageHess**2, axis=-1))
        imageHess /= np.max(imageHess)
    else:
        imageHess = np.ones_like(img)

    if not calculateWithImage:
        img = np.ones_like(img)

    if showResult:
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 30))
        ax1.set_title('Pure Images')
        ax1.imshow(imageCopy)
        ax2.set_title('Grad Images')
        ax2.imshow(imageGrad)
        ax3.set_title('Hess Images')
        ax3.imshow(imageHess)
        ax4.set_title('Image to be correlated')
        ax4.imshow(img)
            
    return img*(imageGrad*imageHess)

"""
Pads two images and returns padded copies.

Parameters:
    img1 (numpy.ndarray): The first input image.
    img2 (numpy.ndarray): The second input image.
    padType (str): The padding type to use ('edge' or 'constant').

Returns:
    tuple: The padded images.
"""
def getPaddedCopies(img1, img2, padType):
    img1, img2 = padImages(img1, img2, padType)
    img1 = padImage(img1, (img1.shape[0]*2, img1.shape[1]*2), padType)
    img2 = padImage(img2, (img2.shape[0]*2, img2.shape[1]*2), padType)
    return img1, img2

"""
Performs Digital Image Correlation (DIC) on two images.

Parameters:
    img1Pure (numpy.ndarray): The first input image.
    img2Pure (numpy.ndarray): The second input image.
    divPosx (int): X-division position for DIC.
    divPosy (int): Y-division position for DIC.
    circVal (int): Circular value for the DIC.
    gradCorr (bool): Whether to calculate with gradient.
    hessCorr (bool): Whether to calculate with Hessian.
    imgCorr (bool): Whether to calculate with Hessian.
    imgProd (bool): Whether to use the original image in the final result.
    gradProd (bool): Whether to merge image with gradient.
    hessProd (bool): Whether to merge image with Hessian.
    padType (str): The padding type to use ('edge' or 'constant').

Returns:
    tuple: Displacement coordinate, X and Y
    displacement statistics
    merged image
    figure
"""
def computeDIC(img1Pure:np.ndarray, img2Pure:np.ndarray, gradCorr=False, hessCorr=False, imgCorr=False, imgProd=True, gradProd=False, hessProd=False, padType='edge', divPosx=-1, divPosy=-1, circVal=-1):
    #####################################################################################################
    #                                           Calculations                                            #
    #####################################################################################################

    img1Gray = grayScale(img1Pure.copy())
    img2Gray = grayScale(img2Pure.copy())
    iterations = 1

    if (len(img1Pure.shape) == 3):
        iterations = 3
    for i in range(iterations):
        img1 = img1Pure[:,:,i].copy().astype(np.float64)
        img2 = img2Pure[:,:,i].copy().astype(np.float64)

        img1 /= np.max(img1)
        img2 /= np.max(img2)
        
        img1 = preprocess(img1, gradProd, hessProd, imgProd, False)
        img2 = preprocess(img2, gradProd, hessProd, imgProd, False)

        img1Grad = preprocess(img1, True, False, False, False)
        img2Grad = preprocess(img2, True, False, False, False)

        img1Hess = preprocess(img1, False, True, False, False)
        img2Hess = preprocess(img2, False, True, False, False)

        img1, img2 = getPaddedCopies(img1, img2, padType)
        img1, img1Gray = padImages(img1, img1Gray, padType)
        img2, img2Gray = padImages(img2, img2Gray, padType)
        img1Grad, img2Grad = getPaddedCopies(img1Grad, img2Grad, padType)
        img1Hess, img2Hess = getPaddedCopies(img1Hess, img2Hess, padType)

        if i == 0:
            real_corr = np.ones_like(img1.copy()).astype(np.complex128)
        
        # Compute normalized correlations
        reverseCorrelation = lambda img1, img2: np.fft.ifftshift(np.fft.ifftn(ZONCC(img1, img2)))

        # Compute normalized correlations
        if imgCorr:
            real_corr *= reverseCorrelation(img1, img2)
        if gradCorr:
            real_corr *= reverseCorrelation(img1Grad, img2Grad)
        if hessCorr:
            real_corr *= reverseCorrelation(img1Hess, img2Hess)
    real_corr = np.abs(real_corr)

    #####################################################################################################
    #                                             Plotting                                              #
    #####################################################################################################
    
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
    
    # Calculate displacement in x and y directions
    delta_x = DIC_pos[0] - real_corr.shape[0] // 2
    delta_y = real_corr.shape[1]// 2 - DIC_pos[1]

    # Create merged image
    merged = np.zeros((real_corr.shape[0] + 2 * abs(delta_x), real_corr.shape[1] + 2 * abs(delta_y)))

    # Copy image 2 to appropriate position in the merged image
    merged[abs(delta_x):abs(delta_x)+real_corr.shape[0], abs(delta_y):abs(delta_y)+real_corr.shape[1]] = img2Gray

    # Adjust position and add image 1 to the merged image
    merged[abs(delta_x)-delta_x:abs(delta_x)-delta_x+real_corr.shape[0],
           abs(delta_y)+delta_y:abs(delta_y)+delta_y+real_corr.shape[1]] -= padImage(img1Gray, real_corr.shape, 'constant')
    merged = np.abs(merged)

    # Return displacement, displacement statistics, merged image, and figure (if required)
    return (-delta_x, -delta_y), findmax, merged

"""
Computes the zero mean normalized cross-correlation of two images.

Parameters:
    img1 (numpy.ndarray): The first input image.
    img2 (numpy.ndarray): The second input image.

Returns:
    numpy.ndarray: The normalized correlation result.
"""
def ZONCC(img1, img2):
    img1, img2 = padImages(img1, img2)

    ftImage1 = np.fft.fftshift(np.fft.fftn(img1))
    ftImage2 = np.fft.fftshift(np.fft.fftn(img2))

    ftImage1 -= np.mean(ftImage1)
    ftImage2 -= np.mean(ftImage2)

    corr = ftImage1 * np.conj(ftImage2)
    corr /= np.max(np.abs(corr))
    return (corr/np.abs(corr))

"""
Returns the overlap region of two images.

Parameters:
    img1 (2D array): The first input image.
    img2 (2D array): The second input image.
    delta_x (int): The horizontal displacement between the two images.
    delta_y (int): The vertical displacement between the two images.
    
Returns:
    overlap1 (2D array): Overlapping region of img1.
    overlap2 (2D array): Overlapping region of img2.
"""   
def getOverlap(img1, img2, delta_x, delta_y):      
    img1, img2 = padImages(img1+1, img2+1, 'constant')
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    
    # Calculate the overlap region coordinates
    start_x = np.max([0, delta_x])
    end_x = np.min([width1, width2+delta_x])
    start_y = np.max([0, delta_y])
    end_y = np.min([height1, height2+delta_y])
    
    # Extract the overlap region from both images
    overlap1 = img1[start_y:end_y,start_x:end_x]
    overlap2 = img2[start_y-delta_y:end_y-delta_y,start_x-delta_x:end_x-delta_x]

    bool1 = overlap1 > 0
    bool2 = overlap2 > 0

    if (np.where(bool1)[0].size < np.where(bool2)[0].size and np.where(bool1)[0].size > 0):
        overlap1 = overlap1[np.any(bool1[:,:,0], axis=1),:,:][:,np.any(bool1[:,:,0], axis=0),:]
        overlap2 = overlap2[np.any(bool1[:,:,0], axis=1),:,:][:,np.any(bool1[:,:,0], axis=0),:]
    elif (np.where(bool2)[0].size > 0): 
        overlap1 = overlap1[np.any(bool2[:,:,0], axis=1),:,:][:,np.any(bool2[:,:,0], axis=0),:]
        overlap2 = overlap2[np.any(bool2[:,:,0], axis=1),:,:][:,np.any(bool2[:,:,0], axis=0),:]  
    
    return overlap1-1, overlap2-1

def interpolateQuiver(X, Y, UV, order=3): # TODO: Integrate into class if unaffected is removed
    basis = Tools.polyBasis2D(Y.ravel(), X.ravel(), order)
    sUV = np.linalg.lstsq(np.vstack(np.array([basis])).T, np.array([UV[1].ravel(),UV[0].ravel()]).T, rcond=None)[0]
    return sUV

"""
Warp an image according to given displacement fields using piecewise polynomials (splines).

Parameters:
    img (numpy.ndarray): Input image array of shape (height, width, channels)
    disp_x (numpy.ndarray): Displacement field for X direction, shape (height, width)
    disp_y (numpy.ndarray): Displacement field for Y direction, shape (height, width)

Returns:
    warped_image (numpy.ndarray): Warped image array of shape (height, width, channels)
"""
def warpToFieldPiecewise(img, disp_x, disp_y):
    height, width, channels = img.shape

    # Generate grid coordinates
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Add displacement and clip to stay within image boundaries
    displaced_coords_x = np.clip(x_coords+disp_x, 0, width-1)
    displaced_coords_y = np.clip(y_coords+disp_y, 0, height-1)

    # Stack displaced coordinates
    displaced_coords = np.stack((displaced_coords_y, displaced_coords_x), axis=-1)

    # Interpolate image values at displaced coordinates
    warped_image = np.zeros_like(img)
    for c in range(channels):
        # Transpose and reshape the coordinate array
        coords = displaced_coords.reshape((height * width, 2)).T
        # Interpolate image values at displaced coordinates using Scipy's spline-based interpolator
        warped_channel = map_coordinates(img[..., c], coords, order=5, mode='constant', cval=0)
         # Reshape the interpolated values back to image shape
        warped_image[..., c] = warped_channel.reshape((height, width))

    return warped_image

"""
Displays and compares the reference, warped, and original images.

Parameters:
    img1 (numpy.ndarray): The first input image (original image).
    img2 (numpy.ndarray): The second input image (reference image).
    warpedImage (numpy.ndarray): The warped version of the first input image.
"""
def showWarpComparison(img1:np.ndarray, img2:np.ndarray, warpedImage):
    im1, im2 = padImages(img2, warpedImage, 'constant')
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    ax1.imshow(im1[:,:,0]); ax1.set_axis_off(); ax1.set_title('Reference Image')
    ax2.imshow(im2[:,:,0]); ax2.set_axis_off(); ax2.set_title('Warped Image')

    im1_bool = im1 != 0
    im2_bool = im2 != 0
    use_bool = im1_bool if im2_bool.sum() > im1_bool.sum() else im2_bool
    maxC = Tools.maxC if (isinstance(im1[0], int)) else 1
    newDiff = (im1+im2) / np.max(np.abs(im1+im2))
    ax3.imshow(newDiff); ax3.set_axis_off()
    ax3.set_title('New diff: $|I_{ref} - I_{warp}|$ = %.5f' % np.mean(np.abs((im1[use_bool]/maxC)-(im2[use_bool]/maxC))))

    im3, im4 = padImages(img1, img2, 'constant')
    im3_bool, im4_bool = im3 != 0, im4 != 0
    use_bool = im4_bool if im3_bool.sum() > im4_bool.sum() else im3_bool
    orgDiff = (im3+im4) / np.max(np.abs(im3+im4))
    ax4.imshow(orgDiff); ax4.set_axis_off()
    ax4.set_title('Original: $|I_{ref} - I_{org}|$ = %.5f' % np.mean(np.abs((im3[use_bool]/maxC)-(im4[use_bool]/maxC))))

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
    basisOld = Tools.polyBasis2D(x, y, oldOrder)
    basisNew = Tools.polyBasis2D(x, y, newOrder)
    cNew = np.zeros(len(basisNew))
    for i,baseOld in enumerate(basisOld):
        for j,baseNew in enumerate(basisNew):
            if np.all(baseOld == baseNew):
                cNew[j] = cOld[i]
                break
    return cNew

"""
Minimizes the error between the given displacements and the fitted displacements.

Parameters:
    UV (numpy.ndarray): The array of displacements to be reshaped and fitted.
    dicClass (object): An object containing displacement coordinates and internal displacement data.
    order (int): The order of the polynomial for fitting.
    basis (callable): A basis function to apply to the input data.
    uvShape (tuple): The shape to reshape the UV array into.

Returns:
    float: The error between the internal displacement and the fitted displacement.
"""
def error_minimization(UV, dicClass, order, uvShape):
    UV = np.reshape(UV, uvShape)
    U, V = Tools.polyFit2D(dicClass.discplacementsCoordinates[1], dicClass.discplacementsCoordinates[0], UV, order=order)
    return np.mean(np.linalg.norm(np.array([dicClass.internalDisplacement[0],dicClass.internalDisplacement[1]])-np.array([V, U]), axis=0))

"""
Minimizes the error between the given displacements and the fitted displacements.

Parameters:
    UV (numpy.ndarray): The array of displacements to be reshaped and fitted.
    dicClass (object): An object containing displacement coordinates and internal displacement data.
    order (int): The order of the polynomial for fitting.
    basis (callable): A basis function to apply to the input data.
    uvShape (tuple): The shape to reshape the UV array into.

Returns:
    float: The error between the internal displacement and the fitted displacement.
"""
def error_minimizationUnaffected(UV, dicClass, order, uvShape):
    UV = np.reshape(UV, uvShape)
    U, V = Tools.polyFit2D(dicClass.discplacementsCoordinatesUnaffected[1], dicClass.discplacementsCoordinatesUnaffected[0], UV, order=order)
    return np.mean(np.linalg.norm(np.array([dicClass.internalDisplacementUnaffected[0],dicClass.internalDisplacementUnaffected[1]])-np.array([V, U]), axis=0))

def findBestGPC(img1, img2, best=None):
    # Define Gradient-Processed Correlation (GPC)
    def GPC(iteration, combination):
        try:
            if not np.any(combination):
                return None, None, None
            dic = DicClass(combination)
            dic.defineImages(img1, img2)
            dic.adjustExposure()
            dic.calculate()
            dic.findOverlap()
            return dic, iteration, dic.overlap_quality != 0
        except:
            print('Combination #%d failed\n' % iteration)
            return None, None, None

    with ThreadPoolExecutor() as executor:
        best_i, abs_diff = 0, 0

        for f in as_completed([executor.submit(GPC, i, c) for i,c in enumerate(combinations)]):
            dic, i, dic_bool = f.result()

            if dic is not None:
                if best is None:
                    quality_diff = 0
                else:
                    quality_diff = np.mean(best.overlap_quality[best.overlap_quality != 0]) - np.mean(dic.overlap_quality[dic_bool])

                if best is None or quality_diff > 1e-4:
                    best = dic
                    best_i = i
                    abs_diff = quality_diff

        executor.shutdown()
    return best, combinations[best_i], abs_diff

def getInternalReport(dicObject, showZeros=False):
    zeros = np.sum(~dicObject.chosenComboMat[np.sum(dicObject.chosenComboMat, axis=-1) == 0,0])
    print('Total Combinations=%d' % (dicObject.chosenComboMat.size // 6))
    print('Zeros  | NoGradientProcessing=%d' % zeros)

    twosies = np.sum(dicObject.chosenComboMat, axis=-1) == 2
    twos = tuple([np.sum(dicObject.chosenComboMat[twosies,i].astype(int)+dicObject.chosenComboMat[twosies,j].astype(int) == 2) for i in range(3) for j in range(3, 6)])
    print('Twos   | ImWImCor=%-2d ImWGCor=%-2d ImWHCor=%-2d GWImCor=%-2d GWGCor=%-2d GWHCor=%-2d HWImCor=%-2d HWGCor=%-2d HWHCor=%d' % twos)
    print('       | Total=%d' % sum(twos))

    threesies = np.sum(dicObject.chosenComboMat, axis=-1) == 3
    threes = tuple([np.sum(dicObject.chosenComboMat[threesies,i].astype(int)+dicObject.chosenComboMat[threesies,j].astype(int)+dicObject.chosenComboMat[threesies,k].astype(int) == 3)
                    for i,j,k in zip([0]*3+[1]*3+[2]*3+[0]*6+[1]*3, [3,3,4]*3+[1]*3+[2]*6, [4,5,5]*3+[3,4,5]*3)])
    print('Threes | ImWImGCorr=%-2d ImWImHCorr=%-2d ImWGHCorr=%-2d GWImGCorr=%-2d  GWImHCorr=%-2d GWGHCorr=%-2d  HWImGCorr=%-2d HWImHCorr=%-2d HWGHCorr=%d' % threes[:9])
    print('       | ImGWImCorr=%-2d ImGWGCorr=%-2d  ImGWHCorr=%-2d ImHWImCorr=%-2d ImHWGCorr=%-2d ImHWHCorr=%-2d GHWImCorr=%-2d GHWGCorr=%-2d  GHWHCorr=%d' % threes[9:])
    print('       | Total=%d' % sum(threes))

    foursies = np.sum(dicObject.chosenComboMat, axis=-1) == 4
    fours = tuple([np.sum((~dicObject.chosenComboMat[foursies,i]).astype(int)+(~dicObject.chosenComboMat[foursies,j]).astype(int) == 2)
                   for i,j in zip([0]*3+[1]*3+[2]*3+[2,2,0,4,3,3], [3,4,5]*3+[1,0,1,5,5,4])])
    print('Fours  | missingImNotImCorr=%-2d missingImNotGCorr=%-2d missingImNotHCorr=%-2d missingGNotImCorr=%-2d missingGNotGCorr=%-2d  missingGNotHCorr=%d' % fours[:6])
    print('       | missingHNotImCorr=%-2d  missingHNotGCorr=%-2d  missingHNotHCorr=%-2d  missingHmissingG=%-2d  missingHmissingIm=%-2d missingImmissingG=%d' % fours[6:12])
    print('       | NotHCorrNotGCorr=%-2d   NotHCorrNotImCorr=%-2d NotImCorrNotGCorr=%d' % fours[12:])
    print('       | Total=%d' % sum(fours))

    fivesies = np.sum(dicObject.chosenComboMat, axis=-1) == 5
    fives = tuple([np.sum(~dicObject.chosenComboMat[fivesies,i]) for i in range(6)])
    print('Fives  | missingIm=%-2d missingGrad=%-2d missingHess=%-2d NotImCorr=%-2d NotGCorr=%-2d NotHCorr=%d' % fives)
    print('       | Total=%d' % sum(fives))

    full = np.sum(np.sum(dicObject.chosenComboMat, axis=-1) == 6)
    print('Full   | AllGradientProcessing=%d' % full)

    _, ax = plt.subplots(figsize=(15, 8), layout='constrained')
    values = [zeros, *twos, *threes, *fours, *fives, full][::-1]
    labels = ['Uncomparable picture',
              '$F(I)$', '$F(|dI/dx|)$', '$F(|d^2I/dx^2|)$', '$F(G)$', '$F(|dG/dx|)$', '$F(|d^2G/dx^2|)$', '$F(H)$', '$F(|dH/dx|)$', '$F(|d^2H/dx^2|)$',
              '$F(I) \cdot F(|dI/dx|)$', '$F(I) \cdot F(|d^2I/dx^2|)$', '$F(|dI/dx|) \cdot F(|d^2I/dx^2|)$',
              '$F(G) \cdot F(|dG/dx|)$', '$F(G) \cdot F(|d^2G/dx^2|)$', '$F(|dG/dx|) \cdot F(|d^2G/dx^2|)$',
              '$F(H) \cdot F(|dH/dx|)$', '$F(H) \cdot F(|d^2H/dx^2|)$', '$F(|dH/dx|) \cdot F(|d^2H/dx^2|)$',
              '$F(I \cdot G)$', '$F(|d(I \cdot G)/dx|)$', '$F(|d^2(I \cdot G))/dx^2|)$',
              '$F(I \cdot H)$', '$F(|d(I \cdot H)/dx|)$', '$F(|d^2(I \cdot H))/dx^2|)$',
              '$F(G \cdot H)$', '$F(|d(G \cdot H)/dx|)$', '$F(|d^2(G \cdot H))/dx^2|)$',
              '$F(|d(G \cdot H)/dx|) \cdot F(|d^2(G \cdot H))/dx^2|)$', '$F(G \cdot H) \cdot F(|d^2(G \cdot H))/dx^2|)$', '$F(G \cdot H) \cdot F(|d(G \cdot H)/dx|)$',
              '$F(|d(I \cdot H)/dx|) \cdot F(|d^2(I \cdot H))/dx^2|)$', '$F(I \cdot H) \cdot F(|d^2(I \cdot H))/dx^2|)$', '$F(I \cdot H) \cdot F(|d(I \cdot H)/dx|)$',
              '$F(|d(I \cdot G)/dx|) \cdot F(|d^2(I \cdot G))/dx^2|)$', '$F(I \cdot G) \cdot F(|d^2(I \cdot G))/dx^2|)$', '$F(I \cdot G) \cdot F(|d(I \cdot G)/dx|)$',
              '$F(I) \cdot F(|dI/dx|) \cdot F(|d^2I/dx^2|)$', '$F(G) \cdot F(|dG/dx|) \cdot F(|d^2G/dx^2|)$', '$F(H) \cdot F(|dH/dx|) \cdot F(|d^2H/dx^2|)$',
              '$F(I \cdot G \cdot H)$', '$F(|d(I \cdot G \cdot H)/dx|)$', '$F(|d^2(I \cdot G \cdot H)/dx^2|)$', '$F(G \cdot H) \cdot F(|d(G \cdot H)/dx|) \cdot F(|d^2(G \cdot H))/dx^2|)$',
              '$F(I \cdot H) \cdot F(|d(I \cdot H)/dx|) \cdot F(|d^2(I \cdot H))/dx^2|)$', '$F(I \cdot G) \cdot F(|d(I \cdot G)/dx|) \cdot F(|d^2(I \cdot G))/dx^2|)$',
              '$F(|d(I \cdot G \cdot H)/dx|) \cdot F(|d^2(I \cdot G \cdot H)/dx^2|)$', '$F(I \cdot G \cdot H) \cdot F(|d^2(I \cdot G \cdot H)/dx^2|)$',
              '$F(I \cdot G \cdot H) \cdot F(|d(I \cdot G \cdot H)/dx|)$', '$F(I \cdot G \cdot H) \cdot F(|d(I \cdot G \cdot H)/dx|) \cdot F(|d^2(I \cdot G \cdot H)/dx^2|)$'][::-1]
    
    if not showZeros:
        values.pop(-1)
        labels.pop(-1)

    nonZeroValues = np.array(values).nonzero()[0].astype(int).tolist()
    realValues = [values[i] for i in nonZeroValues]
    rects = ax.barh([labels[i] for i in nonZeroValues], realValues)
    ax.xaxis.grid(True)
    ax.bar_label(rects, [str(i) for i in realValues])

def showInternalResults(dicObject):
    dicObject.showQuiver()
    plt.figure(figsize=(20, 8))
    
    # Plot the image and quiver
    plt.imshow(dicObject.img1)
    plt.quiver(dicObject.discplacementsCoordinates[1], dicObject.discplacementsCoordinates[0], 
               dicObject.internalDisplacement[1], dicObject.internalDisplacement[0], color='r')

    # Set axis limits and invert y-axis
    plt.axis([np.min(dicObject.internalDisplacement[1]), dicObject.img1.shape[1]+10, 
              np.min(dicObject.internalDisplacement[0]), dicObject.img1.shape[0]+10])
    plt.invert_yaxis()

    # Add text with a rectangle (bbox) around the index for visibility
    x = dicObject.discplacementsCoordinates[1].ravel()
    y = dicObject.discplacementsCoordinates[0].ravel()
    l = [combinations.index(list(c)) for c in dicObject.chosenComboMat.reshape(int(np.prod(dicObject.chosenComboMat.shape)/6), 6)]
    if len(x) < 100:
        bbox = dict(fc='w', ec='k', boxstyle='round,pad=.3', alpha=.5)
        fs = 12
    else:
        bbox = dict(fc='w', ec='k', boxstyle='round,pad=.2', alpha=.5)
        fs = 8
    for i in range(len(x)):
        plt.text(x[i], y[i], l[i], ha='center', va='center', c='b', size=fs, bbox=bbox)
    plt.title('Chosen DIC combinations')

def get_polynomial_order(num_coeffs):
    """
    Determine the order of the polynomial based on the number of coefficients.
    """
    order = 0
    while (order + 1) * (order + 2) // 2 <= num_coeffs:
        order += 1
    return order - 1

def transform_polynomial(coeffs, h, k):
    """
    Transform the polynomial coefficients when the origin is shifted by (h, k).
    
    :param coeffs: 1D numpy array where each element is a coefficient for terms x^i y^j.
    :param h: Shift in x direction.
    :param k: Shift in y direction.
    :return: 1D numpy array of the transformed coefficients.
    """
    
    # Calculate the order of the polynomial
    num_coeffs = len(coeffs)
    order = get_polynomial_order(num_coeffs)
    
    # Initialize the new coefficients array
    new_coeffs = np.zeros_like(coeffs)
    
    # Create a mapping from index to (i, j) pairs
    index_to_ij = []
    idx = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            index_to_ij.append((j, i))
            idx += 1
    
    # Map the input coefficients to a structured 2D array
    poly_coeffs = np.zeros((order + 1, order + 1))
    for idx, (i, j) in enumerate(index_to_ij):
        poly_coeffs[i, j] = coeffs[idx]

    # Iterate through all possible terms in the original polynomial
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if poly_coeffs[i, j] != 0:
                for m in range(i + 1):
                    for l in range(j + 1):
                        new_coeffs_idx = index_to_ij.index((m, l))
                        new_coeffs[new_coeffs_idx] += (
                            poly_coeffs[i, j] * comb(i, m) * h**(i-m) * comb(j, l) * k**(j-l)
                        )
    
    return new_coeffs
