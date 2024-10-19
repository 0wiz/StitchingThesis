# Local
from resources import Tools # type: ignore
from resources import FlowTools # type: ignore

# Math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Data
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor

class DicClass:
    """ Class for performing Digital Image Correlation (DIC) operations on images. """
    def __init__ (self, gpcCombination=[True,False,False,True,False,False], padType='constant'):
        self.img1 = None                                    # First image for comparison.
        self.img2 = None                                    # Second image for comparison.
        self.img1_overlap = None                            # Overlapping region of the first image.
        self.img2_overlap = None                            # Overlapping region of the second image.
        self.overlappingPoint = None                        # Coordinates of the overlapping point.
        self.corrValue = None                               # Correlation value between img1 and img2.
        self.gpcCombination = gpcCombination                # Combination of Gradient-Processed Correlation (GPC) to use
        self.padType = padType                              # Type of padding for images ('constant' by default).
        self.internalDisplacement = None                    # Internal displacements of image 1.
        self.CValue = None                                  # Correlation values across displacements.
        self.discplacementsCoordinates = None               # Coordinates of displacements.
        self.sUV0 = None                                    # Initial displacement field for optimization. 
        self.OptimizeResult = None
        self.bigDisplacement = None
        self.sUV0Unaffected = None # Is unaffected actually presented anywhere?

    """
    Defines the images for DIC.
        
    Parameters:
        img1 (numpy.ndarray): The first input image.
        img2 (numpy.ndarray): The second input image.
    """
    def defineImages(self, img1:np.ndarray, img2:np.ndarray):
        self.img1, self.img2 = img1.copy(), img2.copy()

    """ Performs DIC computation. """
    def calculate(self):
        self.overlappingPoint, self.corrMap, self.merged = FlowTools.computeDIC(self.img1, self.img2, *self.gpcCombination, self.padType)
        self.corrValue = np.max(self.corrMap) / np.sum(self.corrMap)

    """ Finds the overlapping regions of image 1 and image 2. Saves them and the overlap quality to the class. """
    def findOverlap(self):
        self.img2_overlap, self.img1_overlap = FlowTools.getOverlap(self.img2, self.img1, -self.overlappingPoint[1], self.overlappingPoint[0])
        self.overlap_quality = np.sqrt(np.abs(np.mean(self.img2_overlap, axis=-1)**2 - np.mean(self.img1_overlap, axis=-1)**2))

    """ Performs simple exposure adjustment on both images. """
    def adjustExposure(self):
       maxval = np.max([np.max(self.img1), np.max(self.img2)])
       dtype = self.img1.dtype
       self.img1 = self.img1.astype('float64')
       self.img2 = self.img2.astype('float64')
       self.img1 /= np.max(self.img1)
       self.img2 /= np.max(self.img2)
       self.img1 *= maxval
       self.img2 *= maxval
       self.img1 = self.img1.astype(dtype)
       self.img2 = self.img2.astype(dtype)
    
    """ Displays overlapping regions and their quality difference. """
    def showOverlap(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        if (self.img1_overlap.size > 0):
            ax1.imshow(self.img1_overlap); ax1.set_axis_off(); ax1.set_title('Image 1 Overlap')
        if (self.img2_overlap.size > 0):
            ax2.imshow(self.img2_overlap); ax2.set_axis_off(); ax2.set_title('Image 2 Overlap')
        if (self.img1_overlap.size > 0 and self.img2_overlap.size > 0):
            scale = ax3.imshow(self.overlap_quality, cmap='Greys_r'); ax3.set_axis_off()
            ax3.set_title('Overlap Difference: %.5f' % np.mean(self.overlap_quality))
            fig.colorbar(scale, ax=ax3)

    """ Displays both images. """
    def showImages(self):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        ax1.imshow(self.img1); ax1.set_axis_off(); ax1.set_title('Image 1')
        ax2.imshow(self.img2); ax2.set_axis_off(); ax2.set_title('Image 2')

    """ Displays quiver plot of internal displacement. """
    def showQuiver(self):
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 13), constrained_layout=True)

        ax1.imshow(self.img1); ax1.set_axis_off(); ax1.set_title('Image to be compared')

        ax2.imshow(self.img1); ax2.set_axis_off(); ax2.set_title('Quiver')
        ax2.quiver(self.discplacementsCoordinates[1], self.discplacementsCoordinates[0], self.internalDisplacement[1], self.internalDisplacement[0], color='r')
        ax2.axis([np.min(self.internalDisplacement[1]), self.img1.shape[1]+10, np.min(self.internalDisplacement[0]), self.img1.shape[0]+10])
        ax2.invert_yaxis()

        ax3.imshow(self.img2); ax3.set_axis_off(); ax3.set_title('Reference Image')
    
    """
    Performs image warping based on internal displacement.
        
    Parameters:
        order: The order of the polynomial used in minimization.
        optimized: Whether or not to optimize... something
    """
    def warp(self, order=1, optimized=False):
        warpSetting = DicClass(self.gpcCombination)
        warpSetting.defineImages(self.img1, self.img2)
        warpSetting.windowSizeX = self.windowSizeX
        warpSetting.windowSizeY = self.windowSizeY
        warpSetting.internalDisplacement = self.internalDisplacement
        warpSetting.discplacementsCoordinates = self.discplacementsCoordinates
        sUV = None
        if not optimized:
            new_x = np.linspace(0, warpSetting.img1.shape[1], warpSetting.img1.shape[1])
            new_y = np.linspace(0, warpSetting.img1.shape[0], warpSetting.img1.shape[0])
            new_X, new_Y = np.meshgrid(new_x, new_y)

            sUV = FlowTools.interpolateQuiver(*warpSetting.discplacementsCoordinates, warpSetting.internalDisplacement, order)
            basis = np.array(Tools.polyBasis2D(new_X.ravel(), new_Y.ravel(), order))
            fitU, fitV = Tools.polyFit2D(new_X, new_Y, sUV, basis, order)
            if self.bigDisplacement is not None:
                dicX, dicY = [warpSetting.discplacementsCoordinates[i].copy()+self.bigDisplacement[i]/2 for i in (1,0)]
                displacedX, displacedY = [warpSetting.internalDisplacement[i].copy() for i in (0,1)]

                sUVUnaffected = FlowTools.interpolateQuiver(dicX, dicY, (displacedX, displacedY), order)
                self.internalDisplacementUnaffected = np.array([displacedX, displacedY])
                self.discplacementsCoordinatesUnaffected = np.array([dicY, dicX])
                self.sUV0Unaffected = sUVUnaffected
        else: 
            fitU, fitV = self.newInternalDisplacement
            new_Y, new_X = self.newDiscplacementsCoordinates
            sUV = self.sUV0
            basis = self.fitBasis

        warpSetting.internalDisplacement = np.array([fitU, fitV])
        warpSetting.discplacementsCoordinates = np.array([new_Y, new_X])

        warpSetting.windowSizeX = int(warpSetting.discplacementsCoordinates[1][1,0] - warpSetting.discplacementsCoordinates[1][0,0])
        warpSetting.windowSizeY = int(warpSetting.discplacementsCoordinates[0][0,1] - warpSetting.discplacementsCoordinates[0][0,0])
        self.warpedImage = FlowTools.warpToFieldPiecewise(warpSetting.img1, -fitU, fitV)

        warpSetting.defineImages(self.warpedImage, self.img2_overlap)
        warpSetting.calculate()
        warpSetting.findOverlap()
        if not optimized:
            warpSetting.internalDisplacement = np.array([fitU, fitV])
            self.newInternalDisplacement = np.array([fitU, fitV])
            warpSetting.discplacementsCoordinates = np.array([new_Y, new_X])
            self.newDiscplacementsCoordinates = np.array([new_Y, new_X])
            self.warped_img1_overlap = warpSetting.img1_overlap
            self.warped_img2_overlap = warpSetting.img2_overlap
            self.sUV0 = sUV
            self.fitBasis = basis

    """
    Optimizes image warping using error minimization.

    Parameters:
        order: The order of the polynomial used in minimization.
    """
    def optimizeWarping(self, order=1):
        basis = Tools.polyBasis2D(self.discplacementsCoordinates[1], self.discplacementsCoordinates[0], order=order)
        uvShape = (len(basis), 2)
        if (self.sUV0 is None or len(self.sUV0.ravel()) != uvShape[0]*uvShape[1]):
            UV = np.zeros(uvShape)
        else:
            UV = np.reshape(self.sUV0.ravel(), uvShape)
        error1 = FlowTools.error_minimization(UV, self, order, uvShape)
        result = minimize(FlowTools.error_minimization, UV.ravel(), (self, order, uvShape), 'Powell', tol=1e-18, options={"maxiter":1e12})
        error2 = FlowTools.error_minimization(result.x, self, order, uvShape)
        if (not result.success or error1 < error2):
            print('Did not finish')
        self.OptimizeResult = result
        self.newInternalDisplacement = np.array(Tools.polyFit2D(self.newDiscplacementsCoordinates[1], self.newDiscplacementsCoordinates[0], np.reshape(result.x, uvShape), order=order))

    """
    Rolls window and performs DIC calculation to find best GPC combination.

    Parameters:
        stepLength: Step-size per window roll vertically and horizontally.
        windowSize: The dimensions of the rolling window.
        extraWiggle: Additional room per window roll for DIC to search in.
    """
    def rollWindowAndFindBestGPC(self, stepLength, windowSize, extraWiggle=0):
        if type(windowSize) == int:
            windowsizeX = windowSize
            windowsizeY = windowSize
        else:
            windowsizeX, windowsizeY = windowSize
        
        # Pad images here as before
        im1 = FlowTools.padImage(self.img1, (self.img1.shape[0]+np.ceil(windowsizeY).astype(int),
                                           self.img1.shape[1]+np.ceil(windowsizeX).astype(int)), 'constant')
        im2 = FlowTools.padImage(self.img2, (self.img2.shape[0]+np.ceil(windowsizeY).astype(int) + np.ceil(extraWiggle).astype(int),
                                           self.img2.shape[1]+np.ceil(windowsizeX).astype(int)+np.ceil(extraWiggle).astype(int)), 'constant')
        
        y_positions = list(range(0, im1.shape[0] - windowsizeY, stepLength))
        x_positions = list(range(0, im1.shape[1] - windowsizeX, stepLength))
        results = []

        with ThreadPoolExecutor() as executor:
            for y_chunk, x_chunk in chunk_positions(y_positions, x_positions, 10):  # Adjust chunk_size as needed
                for y_s in y_chunk:
                    for x_s in x_chunk:
                        results.append(executor.submit(process_patch, y_s, x_s, im1, im2, windowsizeY, windowsizeX, extraWiggle))
            
            processed_results = [future.result() for future in results]
            executor.shutdown()

        # Extracting results and organizing into the necessary outputs as before
        displacement, Cvalue, coordinates, overlapDiff, chosen_combo_mat, combo_diff_mat = [], [], [], [], [], []
        for result in processed_results:
            displacement.append(result['displacement'])
            Cvalue.append(result['Cvalue'])
            coordinates.append(result['coordinates'])
            overlapDiff.append(result['overlapDiff'])
            chosen_combo_mat.append(result['chosen_combo'])
            combo_diff_mat.append(result['combo_diff'])

        self.discplacementsCoordinates = np.array([
            np.reshape(np.array(coordinates)[:,0], (len(y_positions), len(x_positions))),
            np.reshape(np.array(coordinates)[:,1], (len(y_positions), len(x_positions)))
        ])
        self.internalDisplacement = np.array([
            -np.reshape(np.array(displacement)[:,0], (len(y_positions), len(x_positions))),
            -np.reshape(np.array(displacement)[:,1], (len(y_positions), len(x_positions)))
        ])
        self.CValue = np.reshape(np.array(Cvalue), (len(y_positions), len(x_positions)))
        self.chosenComboMat = np.reshape(np.array(chosen_combo_mat), (len(y_positions), len(x_positions), 6))
        self.overlapDiff = np.reshape(np.array(overlapDiff), (len(y_positions), len(x_positions)))
        self.windowSizeX, self.windowSizeY = windowsizeX, windowsizeY
        self.comboDiffMat = np.reshape(np.array(combo_diff_mat), (len(y_positions), len(x_positions)))

    def filterMissMatched(self, self_like=None, ransac_tol=30):
        if self_like is None:
            self_like = self
            
        vec1 = np.array([self_like.discplacementsCoordinates[0].ravel(),
                         self_like.discplacementsCoordinates[1].ravel()])
        vec2 = np.array([self_like.discplacementsCoordinates[0].ravel()+self_like.internalDisplacement[0].ravel(),
                         self_like.discplacementsCoordinates[1].ravel()+self_like.internalDisplacement[1].ravel()])
        labels = cv.findHomography(vec1.T, vec2.T, cv.RANSAC, ransac_tol)[1].reshape(self_like.discplacementsCoordinates[0].shape)
        overlap_bool = labels == 1 * ~np.isnan(self_like.CValue)

        self.internalDisplacement = np.array([self_like.internalDisplacement[i][overlap_bool] for i in (0,1)])
        self.discplacementsCoordinates = np.array([self_like.discplacementsCoordinates[i][overlap_bool] for i in (0,1)])
        self.chosenComboMat = self_like.chosenComboMat[overlap_bool,:]
        
        self.comboDiffMat = np.zeros_like(self_like.comboDiffMat)
        self.comboDiffMat[overlap_bool] = self_like.comboDiffMat[overlap_bool]
        
        self.CValue = np.zeros_like(self_like.CValue)
        self.CValue[overlap_bool] = self_like.CValue[overlap_bool]
        
        self.overlapDiff = np.zeros_like(self_like.overlapDiff)
        self.overlapDiff[overlap_bool] = self_like.overlapDiff[overlap_bool]

def chunk_positions(y_positions, x_positions, chunk_size):
    for i in range(0, len(y_positions), chunk_size):
        for j in range(0, len(x_positions), chunk_size):
            yield y_positions[i:i+chunk_size], x_positions[j:j+chunk_size]

def process_patch(y_s, x_s, im1, im2, windowsizeY, windowsizeX, extraWiggle):
    subim1 = im1[y_s:y_s+windowsizeY,x_s:x_s+windowsizeX]
    subim2 = im2[y_s:y_s+windowsizeY+extraWiggle,x_s:x_s+windowsizeX+extraWiggle]

    if np.all(subim1 == 0) or np.all(subim2 == 0):
        return {
            'displacement': (0, 0),
            'Cvalue': np.nan,
            'coordinates': [y_s, x_s],
            'overlapDiff': np.nan,
            'chosen_combo': [False, False, False, False, False, False],
            'combo_diff' : 0
        }
    
    subDic, chosen_combo, combo_diff = FlowTools.findBestGPC(subim1, subim2)
    
    return {
        'displacement': [subDic.overlappingPoint[0], subDic.overlappingPoint[1]],
        'Cvalue': subDic.corrValue,
        'coordinates': [y_s, x_s],
        'overlapDiff': np.mean(np.mean(subDic.img2_overlap, axis=-1) - np.mean(subDic.img1_overlap, axis=-1)),
        'chosen_combo': chosen_combo,
        'combo_diff' : combo_diff
    }
