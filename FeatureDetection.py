# Local
import resources.image_interpolator as tools # type: ignore

# Math
import numpy as np

# Data
import cv2 as cv

# Filter out matches lower than median similarity
def MedSim(matches):
    return sorted(matches, key=lambda m: m.distance)[:len(matches)//2]

# Filter with RANSAC (t: Maximum allowed reprojection error to treat a point pair as an inlier)
def RANSAC(matches, t=30):
    p1, p2 = tools.getMatchPoints(matches, kp1, kp2)
    _, labels = cv.findHomography(p1.T, p2.T, cv.RANSAC, t)
    return [matches[i] for i,l in enumerate(labels) if l == 1]

def FeatureDetection(img1, img2, ransac_tol=30, order=2):
    global kp1, kp2

    # Find keypoints in images
    featureFinder = cv.SIFT_create()
    kp1, des1 = featureFinder.detectAndCompute(cv.cvtColor(img1[:,:,::-1], cv.COLOR_BGR2GRAY), None)
    kp2, des2 = featureFinder.detectAndCompute(cv.cvtColor(img2[:,:,::-1], cv.COLOR_BGR2GRAY), None)
    
    # Match keypoints & evaluate filtering
    matcher = cv.BFMatcher(crossCheck=True)
    matches = matcher.match(des1, des2)

    # Filter
    filterMatches = RANSAC(MedSim(matches), t=ransac_tol)
    minAlgo = 'Powell'; tol = 1e-6
    c1 = c2 = np.zeros((tools.polyTerms2D(order), 2))
    p1, p2 = tools.getMatchPoints(filterMatches, kp1, kp2)
    p1G, p2G, translation, rotation, center, angle = tools.globalAlignment(p1, p2)
    p1L, p2L, c1, c2 = tools.localAlignment(p1G, p2G, c1, c2, order, minAlgo, tol)
    return p1L, p2L, c1, c2, translation, rotation, center, angle