# Math
import numpy as np
import matplotlib as mlab
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Data
import cv2 as cv
from pathlib import Path
from datetime import datetime
from operator import attrgetter as attr

# General Declarations
timestamp = lambda: datetime.now().strftime('%Y-%m-%d %H.%M.%S')
maxC = np.iinfo(np.uint8).max # Max color value in standard cases
ch = 3 # Number of color channels in RGB

""" Set display settings for kernel """
def displaySettings():
    np.set_printoptions(3, linewidth=200)
    mlab.rc('legend', fontsize=12)
    mlab.rcParams['axes.formatter.use_mathtext'] = True
    plt.rc('font', family='serif', serif='cmr10', size=plt.rcParams['legend.fontsize']+4)
    plt.rc('axes', unicode_minus=False)

""" Save image according to set standard and type """
def save(name, img, stamp=timestamp, path='Figures/', compression=50, imgExt='.jpg'):
    Path(path).mkdir(exist_ok=True)
    if callable(stamp):
        stamp = stamp()
    if isinstance(img, plt.Figure):
        img.savefig(path+name+' '+stamp+'.pdf', bbox_inches='tight')
    elif isinstance(img, np.ndarray):
        if imgExt == '.jpg':
            if img.shape[-1] > 3:
                img[img[:,:,-1] == 0] = maxC # Set alpha to max to prevent zero-opacity being interpreted as black
            cv.imwrite(path+name+' '+stamp+imgExt, img, [cv.IMWRITE_JPEG_QUALITY, 100-compression])
        elif imgExt == '.png':
            cv.imwrite(path+name+' '+stamp+imgExt, img, [cv.IMWRITE_PNG_COMPRESSION, int(np.round(9*compression/100))])

""" Unpacks an OpenCV DMatch array into two 2-by-n Numpy Matrices, for example 'p1, p2 = unpackDMatchPoints(matches)' where 'p1 = [[x1, x2, .., xn], [y1, y2, .., yn]]' """
getMatchPoints = lambda matches, kp1, kp2: [np.array([[kp[idx(m)].pt[dim] for m in matches] for dim in (0,1)]) for (kp, idx) in ((kp1, attr('queryIdx')), (kp2, attr('trainIdx')))]

""" Rotation matrix to matrix-pre-multiply coordinates in the format [[x₁, x₂, ..., xₙ], [y₁, y₂, ..., yₙ]] """
rotMat = lambda angle: np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

""" The number of terms in a n:th order 2D-polynomial """
polyTerms2D = lambda order: (order+1)**2 - (order**2+order)//2 # Sqaure - Triangle

""" Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc. """
def polyBasis2D(x, y, order=2):
    basis = []
    for i in range(order+1):
        for j in range(order+1-i):
            basis.append(x**j * y**i)
    return basis

"""
Fits a polynomial model to the given data using a basis function.

Parameters:
    X (numpy.ndarray): The input data for the independent variable.
    Y (numpy.ndarray): The input data for the dependent variable.
    poly (numpy.ndarray): Coefficients of the polynomial to be fitted, where each row corresponds to polynomial coefficients for a different term.
    basis (callable, optional): The fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc.
    order (int, optional): The order of the polynomial if `polyBasis2D` is used. Default is 3.

Returns:
    tuple:
        U (numpy.ndarray): The fitted U values reshaped to the shape of X.
        V (numpy.ndarray): The fitted V values reshaped to the shape of X.
"""
def polyFit2D(X, Y, poly:np.ndarray, basis=None, order=3):
    x = X.ravel()
    y = Y.ravel()
    polyU = poly[:,0]
    polyV = poly[:,1]
    if basis is None:
        basis = np.array(polyBasis2D(x, y, order))
    U = polyU @ basis
    V = polyV @ basis
    U = U.reshape(*X.shape)
    V = V.reshape(*X.shape)
    return U, V

""" Minimization function of a polynomial based transform """
def minF(c, p, goal, order, shape):
    fit = polyFit2D(*p, c.reshape(shape), order=order)
    error = np.linalg.norm(goal-fit, axis=0)
    return np.mean(error)

""" Perform global translation and rotation, then calculate the remaining error (distance) """
def globalAlignment(p1, p2, translation=None, rotation=None, center=None, angle=None):
    if center is None:
        center = [np.mean(p, axis=1).reshape(2, 1) for p in (p1, p2)]
    if rotation is None:
        if angle is None:
            angle = [np.mean(np.arctan2(y, x)) for x,y in (p1-center[0], p2-center[1])]
        rotation = [rotMat(sign*(angle[0]-angle[1])/2) for sign in (1,-1)]
    if translation is None:
        translation = [sign*(center[1]-center[0])/2 for sign in (1,-1)]

    p1, p2 = [rotation[i] @ (p - center[i]) + center[i] + translation[i] for i,p in enumerate((p1, p2))]
    return p1, p2, translation, rotation, center, angle

""" Find polynom for warping and calculate warp-vectors, then calculate the remaining error (distance) """
def localAlignment(p1, p2, c1, c2, order, algorithm=None, tol=None, options=dict(maxiter=1e12)):
    if algorithm is not None:
        shape = c1.shape
        o1 = minimize(minF, c1.ravel(), (p1, (p2-p1)/2, order, shape), algorithm, tol=tol, options=options)
        o2 = minimize(minF, c2.ravel(), (p2, (p1-p2)/2, order, shape), algorithm, tol=tol, options=options)
        c1, c2 = o1.x.reshape(shape), o2.x.reshape(shape)
    
    l1 = polyFit2D(*p1, c1, order=order)
    l2 = polyFit2D(*p2, c2, order=order)
    
    if algorithm is not None:
        errorC = np.mean(np.abs(np.vstack(((p2-p1)/2-l1, (p1-p2)/2-l2))), axis=1)
        print('Mean warp-error-to-goal per coefficient set:   e1x=%.3f,   e1y=%.3f,   e2x=%.3f,   e2y=%.3f' % tuple(errorC))
    return p1+l1, p2+l2, c1, c2

""" Overlay two images based on index points before and after transformation """
def overlayImages(p0, p1, p2, imgs):
    overlap, notover, newImg, overlapIdxs = classifyOverlap(p0, p1, p2, imgs)

    diffImg = np.tile(np.abs(np.sum(newImg[0]-newImg[1], axis=2, keepdims=True) / ch), (1,1,4))
    diffImg[np.any(diffImg, axis=2),-1] = maxC

    newImg = np.mean(overlap, axis=0) + np.sum(notover, axis=0)
    newImg = np.pad(newImg, ((0,0),(0,0),(0,1)), constant_values=0)
    newImg[np.any(newImg, axis=2),-1] = maxC
    
    percent = 200*np.sum(overlapIdxs)/np.prod(np.array(imgs).shape[:-1])
    return newImg.astype(np.uint8), diffImg.astype(np.uint8), percent, overlap

""" Defines and returns the overlap for the transformed points """
def classifyOverlap(p0, p1, p2, imgs):
    minX, maxX, minY, maxY = [int(f(np.append(p1[dim], p2[dim]))) for dim in (0,1) for f in (min, max)]
    newImg = np.zeros((2, maxY-minY+1, maxX-minX+1, ch))

    for i, p in enumerate((p1, p2)):
        imgIdx = p.astype(int) - [[minX], [minY]]
        newImg[i][*imgIdx[::-1]] = imgs[i][*p0[::-1]]

    overlapIdxs = np.all(np.any(newImg, axis=-1), axis=0)
    overlap, notover = np.copy(newImg), np.copy(newImg)
    
    overlap[:,~overlapIdxs,:] = 0
    notover[:,overlapIdxs,:] = 0
    return overlap, notover, newImg, overlapIdxs

""" Interpolates all zero pixels adjascent to non-zero pixels in the ovelap """
def refineOverlap(overlap, s=1):
    overlap = overlap.copy()  # Create a copy of the overlap image to avoid modifying the original
    overlapIdxs = overlap != 0 * np.isnan(overlap)  # Identify non-zero and non-NaN pixels in the overlap
    dim = len(overlap.shape)  # Determine the number of dimensions in the overlap image

    # Adjust overlap indices based on the dimensions of the image
    if dim == 4:
        overlapIdxs = np.any(overlapIdxs, axis=(0,-1))
    else:
        overlapIdxs = np.any(overlapIdxs, axis=-1)

    # Find the minimum and maximum x and y coordinates of the overlap region
    y_min = np.min(np.where(np.any(overlapIdxs, axis=1)))
    y_max = np.max(np.where(np.any(overlapIdxs, axis=1)))
    x_min = np.min(np.where(np.any(overlapIdxs, axis=0)))
    x_max = np.max(np.where(np.any(overlapIdxs, axis=0)))

    # Check if there are no valid overlap indices
    if ~np.any(overlapIdxs):
        print('All is wrong')
        return overlap  # Return the unmodified overlap if no valid indices are found

    # Iterate over all non-overlap pixels
    for idx in np.array(np.where(~overlapIdxs)).T:
        # Skip pixels that are on the edge of the overlap region
        if idx[0] <= y_min or idx[0] >= y_max or idx[1] <= x_min or idx[1] >= x_max:
            continue
        
        n_t = []  # List to store neighboring pixels' values

        # Iterate over the neighborhood of the current pixel
        for i in range(s * 2):
            for j in range(s * 2):
                if j == s and i == s:
                    continue  # Skip the current pixel itself
                if dim == 4:
                    # Collect neighboring pixels for 4D image
                    n_t.append(overlap[:,np.max([np.min([idx[0]+s-j, overlap.shape[1]]), 0]), np.max([np.min([idx[1]+s-i, overlap.shape[2]]), 0]),:])
                else:
                    # Collect neighboring pixels for lower-dimensional image
                    n_t.append(overlap[np.max([np.min([idx[0]+s-j, overlap.shape[1]]), 0]), np.max([np.min([idx[1]+s-i, overlap.shape[2]]), 0]),:])

        n_t = np.array(n_t)  # Convert the list of neighbors to a NumPy array

        # Skip if there are no valid neighbors
        if ~np.any(n_t):
            continue
        
        if dim == 4:
            # Calculate and assign mean values for the pixel in a 4D image
            overlap[:,*idx,:] = np.array([[np.mean(n_t[np.any(n_t, axis=2)][0::2], axis=0)], [np.mean(n_t[np.any(n_t, axis=2)][1::2], axis=0)]]).reshape(2,3)
        else:
            # Calculate and assign mean values for the pixel in a lower-dimensional image
            overlap[*idx,:] = np.mean(n_t[np.any(n_t, axis=1)][0::2], axis=0).reshape(1,4)
    
    return overlap  # Return the refined overlap image

""" Perform local and global alignment for given c values """
def mergeAndWarpImages(p01, p02, c1, c2, order, translation, rotation, center, imgs):
    # Apply warping to the images
    p1, p2 = globalAlignment(p01, p02, translation, rotation, center)[:2]
    p1, p2 = localAlignment(p1, p2, c1, c2, order)[:2]
    img, diff, _, overlap = overlayImages(p01, p1, p2, imgs)
    return img, diff, overlap, p1, p2

""" Returns a side by side image of the refined overlaps """
def refineAndConcatenateOverlaps(p0, p1, p2, imgs):
    overlap, notover, _, _ = classifyOverlap(p0, p1, p2, imgs)
    overlap = refineOverlap(overlap)

    overlap_bool = overlap != 0
    notover[overlap_bool] = 0
    overlap_bool = np.any(np.any(overlap_bool, axis=0), axis=-1)

    y_min = np.min(np.where(np.any(overlap_bool, axis=1)))
    y_max = np.max(np.where(np.any(overlap_bool, axis=1)))
    x_min = np.min(np.where(np.any(overlap_bool, axis=0)))
    x_max = np.max(np.where(np.any(overlap_bool, axis=0)))

    overlap1 = overlap[0,y_min:y_max,x_min:x_max,:] + notover[0,y_min:y_max,x_min:x_max,:]
    overlap2 = overlap[1,y_min:y_max,x_min:x_max,:] + notover[1,y_min:y_max,x_min:x_max,:]
    return np.concatenate([overlap1, overlap2], axis=1), y_min, y_max, x_min, x_max

""" Applies the c functions to both images with a given inital overlap """
def opticalFlowInterpolator(img1, img2, c_dis, c_no_dis, order, overlapDisplacement, imgs, showTranslation=False):
    p01_Y = np.arange(img1.shape[0])
    p01_X = np.arange(img1.shape[1])

    p01_y, p01_x = np.meshgrid(p01_Y, p01_X)

    p01 = np.array([p01_x.ravel(), p01_y.ravel()])

    p02_Y = np.arange(img2.shape[0])
    p02_X = np.arange(img2.shape[1])
    p02_y, p02_x = np.meshgrid(p02_Y, p02_X)

    p02 = np.array([p02_x.ravel(), p02_y.ravel()])

    # Apply affine transformation to the images
    _, _, translation, rotation, center, angle = globalAlignment(overlapDisplacement, np.array([[0],[0]]))
    p1, p2, _, _, _, _ = globalAlignment(p01, p02, translation, rotation, center, angle)

    if showTranslation:
        img_trans = overlayImages(p01, p1, p2, imgs)[0]
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow((img_trans)/np.max((img_trans)))
        plt.show()
    p1, p2, _, _ = localAlignment(p01, p02, c_no_dis, c_dis, order)
    img, _, _, _ = overlayImages(p01, p1, p2 + overlapDisplacement, imgs)

    return p01, p1, p2, img