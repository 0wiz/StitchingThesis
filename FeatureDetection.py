# Math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlab

# Data processing
import cv2 as cv

# Data analysis
from pathlib import Path
from datetime import datetime
from operator import attrgetter as attr

#Local libraries
import resources.image_interpolator as localWarper # type: ignore

# Macros
timestamp = lambda: datetime.now().strftime('%Y-%m-%d %H.%M.%S')

# General
savePath = 'Figures/'
imgExt = '.png'
Path(savePath).mkdir(exist_ok=True)
np.set_printoptions(3, linewidth=200)
kp1, des1 = (None, None)
kp2, des2 = (None, None)

# OpenCV
maxCVal = np.iinfo(np.uint8).max
cvImg1, cvImg2 = (None, None)
imgH, imgW, imgC = (None, None, None)
downscale = None

# Matplotlib
mlab.rc('legend', fontsize=12)
plt.rc('font', family='serif', serif='cmr10', size=plt.rcParams['legend.fontsize']+4)
plt.rc('axes', unicode_minus=False)
legendBBox = dict(boxstyle='round,pad=.4,rounding_size=.2', fc='white', ec=(.8,.8,.8), alpha=.8)
cBarPad = .01
scatMS = 1

#        ----- Global function declarations -----        #

# Save image according to set standard and type
def save(name, img, path=savePath, stamp=timestamp(), compression=10, imgExt = imgExt):
    if True:
        if isinstance(img, plt.Figure):
            img.savefig(path+name+' '+stamp+'.pdf', bbox_inches='tight')
        elif isinstance(img, np.ndarray):
            if imgExt == '.jpg':
                if img.shape[-1] > imgC:
                    img[img[:,:,-1] == 0] = maxCVal
                cv.imwrite(path+name+' '+stamp+imgExt, img, [cv.IMWRITE_JPEG_QUALITY, 100-compression])
            elif imgExt == '.png':
                cv.imwrite(path+name+' '+stamp+imgExt, img, [cv.IMWRITE_PNG_COMPRESSION, int(np.round(9*compression/100))])

# Draw box(es) with text(s) on an OpenCV image
def cvTextBox(img, text, loc=(20,20), margin=(3,6), alpha=.4, font=cv.FONT_HERSHEY_DUPLEX, size=1, fc=(0,0,0), bc=maxCVal, thickness=2, lineType=cv.LINE_AA):
    if isinstance(text, str):
        text = [text]
    if img.shape[-1] > imgC:
        fc = (*fc, maxCVal)
    size *= img.shape[1] / 2000
    margin = (size*margin[0], size*margin[1])
    thickness = max(int(size*thickness), 1)

    # Calculate positions and add background boxes
    textLoc = [(0,0)]*len(text)
    l, t = loc # Left, Top
    for i, label in enumerate(text):
        w, h = cv.getTextSize(label, font, size, thickness)[0] # Width, Height
        r, b = int(l+w+2*margin[0]), int(t+h+2*margin[1]+2) # Right, Bottom
        sub = img[t:b,l:r,:imgC].astype(np.float32)
        img[t:b,l:r,:imgC] = (sub*(1-alpha)+np.ones_like(sub)*bc*alpha).astype(img.dtype) # cv.AddWeighted inconsistently throws "int() argument must be a string" (memory bug in CV's Python-C++ interface?)
        textLoc[i] = [int(l+margin[0]), int(t+h+margin[1]-1)]
        t = b

    # Loop again to gaurantee that labels are in front
    for i, label in enumerate(text):
        cv.putText(img, label, textLoc[i], font, size, fc, thickness, lineType)

# Draw an (anchored) box with text on a plt axes
def pltTextBox(ax, text, loc, bbox=legendBBox, size=plt.rcParams['legend.fontsize']):
    parambox = mlab.offsetbox.AnchoredText(text, loc, prop=dict(bbox=bbox, size=size))
    parambox.patch.set_alpha(0)
    ax.add_artist(parambox)

# Calculate the distances and angles between each pair in two lists of points
def points2DistAng(p1, p2):
    vectors = p1-p2
    distances = np.linalg.norm(vectors, axis=0)
    angles = np.degrees(np.arctan(vectors[1]/vectors[0]))
    return distances, angles

##########################################################
#       ----- Functions related to filtering -----       #
##########################################################

# Unpacks an OpenCV DMatch array into two 2-by-n Numpy Matrices, for example 'p1, p2 = unpackDMatchPoints(matches)' where 'p1 = [[x1, x2, .., xn], [y1, y2, .., yn]]'
match2Points = lambda matches: [np.array([[kp[idx(m)].pt[dim] for m in matches] for dim in (0,1)]) for (kp,idx) in ((kp1, attr('queryIdx')), (kp2, attr('trainIdx')))]

# Filter out matches lower than median similarity
def MedSim(matches, inStr='m'):
    filtered = sorted(matches, key=lambda m: m.distance)[:len(matches)//2]
    return filtered, 'MedSim(%s)' % inStr

# Filter with RANSAC (t: Maximum allowed reprojection error to treat a point pair as an inlier)
def RANSAC(matches, inStr='m', t=30):
    p1, p2 = match2Points(matches)
    _, labels = cv.findHomography(p1.T, p2.T, cv.RANSAC, t)
    filtered = [matches[i] for i,l in enumerate(labels) if l == 1]
    return filtered, 'RANSAC(%s, t=%d)' % (inStr, t)

def FeatureDetection(image1, image2, ransac_tol = 30, order = 2):
    global cvImg1, cvImg2, imgH, imgW, imgC, downscale, kp1, des1, kp2, des2

    cvImg1, cvImg2 = (image1, image2)
    imgH, imgW, imgC = *cvImg1.shape[:2], 3
    downscale = imgH/20

    # Find keypoints in images
    featureFinder = cv.SIFT_create()
    kp1, des1 = featureFinder.detectAndCompute(cv.cvtColor(image1, cv.COLOR_BGR2GRAY), None)
    kp2, des2 = featureFinder.detectAndCompute(cv.cvtColor(image2, cv.COLOR_BGR2GRAY), None)
    # Match keypoints & evaluate filtering
    matcher = cv.BFMatcher(crossCheck=True)
    matches = matcher.match(des1, des2)
    # Filter
    filterMatches, _ = RANSAC(*MedSim(matches), t = ransac_tol)
    degree = [order, 0]; minAlgo = 'Powell'; tol = 1e-6
    c1 = c2 = np.zeros((2, localWarper.polyTerms2D(*degree)))
    p1, p2 = match2Points(filterMatches)
    p1G, p2G, translation, rotation, center, angle = localWarper.globalAlignment(p1, p2)
    p1L, p2L, c1, c2 = localWarper.localAlignment(p1G, p2G, c1, c2, degree, minAlgo, tol)
    return p1L, p2L, c1, c2, translation, rotation, center, angle