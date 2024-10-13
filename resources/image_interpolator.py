# Math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Data
import cv2 as cv
from pathlib import Path
from datetime import datetime

# General Declarations
timestamp = lambda: datetime.now().strftime('%Y-%m-%d %H.%M.%S')
maxCVal = np.iinfo(np.uint8).max

# Save image according to set standard and type
def save(name, img, stamp=timestamp, path='Figures/', compression=50, imgExt='.jpg'):
    Path(path).mkdir(exist_ok=True)
    if callable(stamp):
        stamp = stamp()
    if isinstance(img, plt.Figure):
        img.savefig(path+name+' '+stamp+'.pdf', bbox_inches='tight')
    elif isinstance(img, np.ndarray):
        if imgExt == '.jpg':
            if img.shape[-1] > 3:
                img[img[:,:,-1] == 0] = maxCVal # Set alpha to max to prevent zero-opacity being interpreted as black
            cv.imwrite(path+name+' '+stamp+imgExt, img, [cv.IMWRITE_JPEG_QUALITY, 100-compression])
        elif imgExt == '.png':
            cv.imwrite(path+name+' '+stamp+imgExt, img, [cv.IMWRITE_PNG_COMPRESSION, int(np.round(9*compression/100))])

# Rotation matrix to matrix-pre-multiply coordinates in the format [[x₁, x₂, ..., xₙ], [y₁, y₂, ..., yₙ]]
rotMat = lambda angle: np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

# Lists the terms of a n:th degree 2D-polynomial [c₀, c₁x, c₂x², ..., cₙy, cₙ₊₁xy, ..., cₙₓₙyⁿ]
#   expand: The polynomial can be combinatorially expanded by m such that terms cxᵃyᵇ where n+m ≥ a+b will be included (n ≥ a and n ≥ b still applies)
def polyVal2D(x, y, c, degree, expand=0):
    terms = []
    for i in range(degree+1):
        for j in range(degree+min(expand-i,-i)+1):
            terms.append(c[len(terms)] * (x**j * y**i))
    return np.array(terms)

# The number of terms in a n:th degree 2D-polynomial, expanded by m such that terms cxᵃyᵇ where n+m ≥ a+b will be included (n ≥ a and n ≥ b still applies)
def polyTerms2D(degree, expand=0):
    n = max(degree-expand, 0)
    return (degree+1)**2 - (n**2+n)//2 # Sqaure - Triangle

# Minimization function of a polynomial based transform
def minF(c, p, goal, degree):
    fit = np.sum([polyVal2D(*p, cxy, *degree) for cxy in np.split(c, 2)], axis=1)
    error = np.linalg.norm(goal-fit, axis=0)
    return np.mean(error)

# Perform global translation and rotation, then calculate the remaining error (distance)
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

# Find polynom for warping and calculate warp-vectors, then calculate the remaining error (distance)
def localAlignment(p1, p2, c1, c2, degree, algorithm=None, tol=None, options=dict(maxiter=1e12)):
    if algorithm is not None:
        o1 = minimize(minF, c1.ravel(), (p1, (p2-p1)/2, degree), algorithm, tol=tol, options=options)
        o2 = minimize(minF, c2.ravel(), (p2, (p1-p2)/2, degree), algorithm, tol=tol, options=options)
        c1, c2 = np.split(o1.x, 2), np.split(o2.x, 2)
    
    l1 = np.sum([polyVal2D(*p1, c1[dim], *degree) for dim in (0,1)], axis=1)
    l2 = np.sum([polyVal2D(*p2, c2[dim], *degree) for dim in (0,1)], axis=1)
    errorC = np.mean(np.abs(np.vstack(((p2-p1)/2-l1, (p1-p2)/2-l2))), axis=1)

    print('Mean warp-error-to-goal per coefficient set:   e1x=%.3f,   e1y=%.3f,   e2x=%.3f,   e2y=%.3f' % tuple(errorC))
    return p1+l1, p2+l2, c1, c2

# Overlay two images based on index points before and after transformation
def overlayImages(p0, p1, p2, imgC, imgs):
    overlap, notover, diffImg, overlapIdxs = getOverlapAndNotover(p0, p1, p2, imgC, imgs)

    newImg = np.mean(overlap, axis=0) + np.sum(notover, axis=0)
    newImg = np.pad(newImg, ((0,0),(0,0),(0,1)), constant_values=0)
    newImg[np.any(newImg, axis=2),-1] = maxCVal
    
    percent = 200*np.sum(overlapIdxs)/np.prod(np.array(imgs).shape[:-1])
    return newImg.astype(np.uint8), diffImg.astype(np.uint8), percent, overlap

# Defines and returns the overlap for the transformed points
def getOverlapAndNotover(p0, p1, p2, imgC, imgs):
    minX, maxX, minY, maxY = [int(f(np.append(p1[dim], p2[dim]))) for dim in (0,1) for f in (min, max)]
    newImg = np.zeros((2, maxY-minY+1, maxX-minX+1, imgC))

    for i, p in enumerate((p1, p2)):
        imgIdx = p.astype(int) - [[minX], [minY]]
        newImg[i][*imgIdx[::-1]] = imgs[i][*p0[::-1]]

    diffImg = np.tile(np.abs(np.sum(newImg[0]-newImg[1], axis=2, keepdims=True) / imgC), (1,1,4))
    diffImg[np.any(diffImg, axis=2),-1] = maxCVal
    
    overlapIdxs = np.all(np.any(newImg, axis=-1), axis=0)
    overlap, notover = np.copy(newImg), np.copy(newImg)
    
    overlap[:,~overlapIdxs,:] = 0
    notover[:,overlapIdxs,:] = 0
    return overlap, notover, diffImg, overlapIdxs

# Interpolates all zero pixels adjascent to non-zero pixels in the ovelap
def refineOverlap(overlap, s=1):
    overlap = overlap.copy()  # Create a copy of the overlap image to avoid modifying the original
    overlapIdxs = overlap != 0 * np.isnan(overlap)  # Identify non-zero and non-NaN pixels in the overlap
    dim = len(overlap.shape)  # Determine the number of dimensions in the overlap image

    # Adjust overlap indices based on the dimensions of the image
    if dim == 4:
        overlapIdxs = np.any(np.any(overlapIdxs, axis=0), axis=-1)
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
                    n_t.append(overlap[:, np.max([np.min([idx[0] + s - j, overlap.shape[1]]), 0]), np.max([np.min([idx[1] + s - i, overlap.shape[2]]), 0]), :])
                else:
                    # Collect neighboring pixels for lower-dimensional image
                    n_t.append(overlap[np.max([np.min([idx[0] + s - j, overlap.shape[1]]), 0]), np.max([np.min([idx[1] + s - i, overlap.shape[2]]), 0]), :])

        n_t = np.array(n_t)  # Convert the list of neighbors to a NumPy array

        # Skip if there are no valid neighbors
        if ~np.any(n_t):
            continue
        
        if dim == 4:
            # Calculate and assign mean values for the pixel in a 4D image
            overlap[:, *idx, :] = np.array([[np.mean(n_t[np.any(n_t, axis=2)][0::2], axis=0)], [np.mean(n_t[np.any(n_t, axis=2)][1::2], axis=0)]]).reshape((2, 3))
        else:
            # Calculate and assign mean values for the pixel in a lower-dimensional image
            overlap[*idx, :] = np.mean(n_t[np.any(n_t, axis=1)][0::2], axis=0).reshape((1, 4))
    
    return overlap  # Return the refined overlap image

# Perform local and global alignment for given c values
def mergeAndWarpImages(p01, p02, c1, c2, degree, translation, rotation, center, imgC, imgs):
    # Apply warping to the images
    p1, p2 = globalAlignment(p01, p02, translation, rotation, center)[:2]
    p1, p2 = localAlignment(p1, p2, c1, c2, degree)[:2]
    img, diff, pct, overlap = overlayImages(p01, p1, p2, imgC, imgs)
    return img, diff, overlap, pct, p1, p2

# Returns a side by side image of the refined overlaps
def refineAndConcatenateOverlaps(p0, p1, p2, imgC, imgs):
    overlap, notover, _, _ = getOverlapAndNotover(p0, p1, p2, imgC, imgs)
    overlap2 = refineOverlap(overlap)

    overlap_bool = overlap2 != 0
    notover[overlap_bool] = 0
    overlap_bool = np.any(np.any(overlap_bool, axis=0), axis=-1)

    y_min = np.min(np.where(np.any(overlap_bool, axis=1)))
    y_max = np.max(np.where(np.any(overlap_bool, axis=1)))
    x_min = np.min(np.where(np.any(overlap_bool, axis=0)))
    x_max = np.max(np.where(np.any(overlap_bool, axis=0)))

    overlap1 = overlap2[0,y_min:y_max,x_min:x_max,:] + notover[0,y_min:y_max,x_min:x_max,:]
    overlap2 = overlap2[1,y_min:y_max,x_min:x_max,:] + notover[1,y_min:y_max,x_min:x_max,:]
    return np.concatenate([overlap1,overlap2], axis=1)

#Applies the c functions to both images with a given inital overlap
def opticalFlowInterpolator(image1, image2, c_dis, c_no_dis, order, overlapDisplacement, imgC, imgs, showTranslation=False):
    p01_Y = np.arange(image1.shape[0])
    p01_X = np.arange(image1.shape[1])

    p01_y, p01_x = np.meshgrid(p01_Y, p01_X)

    p01 = np.array([p01_x.ravel(),p01_y.ravel()])

    p02_Y = np.arange(image2.shape[0])
    p02_X = np.arange(image2.shape[1])
    p02_y, p02_x = np.meshgrid(p02_Y, p02_X)

    p02 = np.array([p02_x.ravel(),p02_y.ravel()])

    # Apply affine transformation to the images
    _, _, translation, rotation, center, angle = globalAlignment(overlapDisplacement, np.array([[0],[0]]))
    p1, p2 = globalAlignment(p01, p02, translation, rotation, center, angle)[:2]

    if showTranslation:
        img_trans = overlayImages(p01, p1, p2, imgC, imgs)[0]
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow((img_trans)/np.max((img_trans)))
        plt.show()
    p1, p2 = localAlignment(p01, p02, c_no_dis, c_dis,[order])[:2]

    print('Image aligned')
    img, diff, pct, overlap = overlayImages(p01, p1, p2 + overlapDisplacement, imgC, imgs)

    print('p1 = ' + np.array_str(p1))
    print('p2 = ' + np.array_str(p2))
    return p01, p1, p2, img, diff, pct, overlap