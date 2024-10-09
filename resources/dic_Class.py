import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy as sc
from datetime import datetime
import math as mt
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import resources.dic_tools as dicTool # type: ignore




##SINGLES
#combinations.append([True, False, False, True, False, False]) #Image: CI
#combinations.append([False, True, False, True, False, False]) #Gradient: CI
#combinations.append([False, False, True, True, False, False]) #Hessian: CI
#
#combinations.append([True, False, False, False, True, False]) #Image: CG
#combinations.append([False, True, False, False, True, False]) #Gradient: CG
#combinations.append([False, False, True, False, True, False]) #Hessian: CG
#
#combinations.append([True, False, False, False, False, True]) #Image: CH
#combinations.append([False, True, False, False, False, True]) #Gradient: CH
#combinations.append([False, False, True, False, False, True]) #Hessian: CH
#combinations.append([True, False, False, True, True, False]) #Image: CIG
#combinations.append([False, True, False, True, True, False]) #Gradient: CIG
#combinations.append([False, False, True, True, True, False]) #Hessian: CIG
#combinations.append([True, False, False, True, False, True]) #Image: CIH
#combinations.append([False, True, False, True, False, True]) #Gradient: CIH
#combinations.append([False, False, True, True, False, True]) #Hessian: CIH
#combinations.append([True, False, False, False, True, True]) #Image: CGH
#combinations.append([False, True, False, False, True, True]) #Gradient: CGH
#combinations.append([False, False, True, False, True, True]) #Hessian: CGH
#combinations.append([True, False, False, True, True, True]) #Image: CIGH
#combinations.append([False, True, False, True, True, True]) #Gradient: CIGH
#combinations.append([False, False, True, True, True, True]) #Hessian: CIGH
#
##DOUBLES
#combinations.append([True, True, False, True, False, False]) #Image x Gradient: CI
#combinations.append([False, True, True, True, False, False]) #Gradient x Hessian: CI
#combinations.append([True, False, True, True, False, False]) #Image x Hessian: CI
#combinations.append([True, True, False, False, True, False]) #Image x Gradient: CG
#combinations.append([False, True, True, False, True, False]) #Gradient x Hessian: CG
#combinations.append([True, False, True, False, True, False]) #Image x Hessian: CG
#combinations.append([True, True, False, False, False, True]) #Image x Gradient: CH
#combinations.append([False, True, True, False, False, True]) #Gradient x Hessian: CH
#combinations.append([True, False, True, False, False, True]) #Image x Hessian: CH
#combinations.append([True, True, False, True, True, False]) #Image x Gradient: CIG
#combinations.append([False, True, True, True, True, False]) #Gradient x Hessian: CIG
#combinations.append([True, False, True, True, True, False]) #Image x Hessian: CIG
#combinations.append([True, True, False, True, False, True]) #Image x Gradient: CIH
#combinations.append([False, True, True, True, False, True]) #Gradient x Hessian: CIH
#combinations.append([True, False, True, True, False, True]) #Image x Hessian: CIH
#combinations.append([True, True, False, False, True, True]) #Image x Gradient: CGH
#combinations.append([False, True, True, False, True, True]) #Gradient x Hessian: CGH
#combinations.append([True, False, True, False, True, True]) #Image x Hessian: CGH
#combinations.append([True, True, False, True, True, True])  #Image x Gradient: CIGH
#combinations.append([False, True, True, True, True, True])  #Gradient x Hessian: CIGH
#combinations.append([True, False, True, True, True, True])  #Image x Hessian: CIGH
#
##TRIPLES
#combinations.append([True, True, True, True, False, False]) #Image x Gradient x Hessian: CI
#combinations.append([True, True, True, False, True, False]) #Image x Gradient x Hessian: CG
#combinations.append([True, True, True, False, False, True]) #Image x Gradient x Hessian: CH
#combinations.append([True, True, True, True, True, False]) #Image x Gradient x Hessian: CIG
#combinations.append([True, True, True, False, True, True]) #Image x Gradient x Hessian: CGH
#combinations.append([True, True, True, True, False, True]) #Image x Gradient x Hessian: CIH
#combinations.append([True, True, True, True, True, True]) #Image x Gradient x Hessian: CIGH



class DicClass:
    """
    Class for performing Digital Image Correlation (DIC) operations on images.

    Attributes:
    ----------
    image1 : np.ndarray or None
        First image for comparison.
    image2 : np.ndarray or None
        Second image for comparison.
    image1_overlap : np.ndarray or None
        Overlapping region of the first image.
    image2_overlap : np.ndarray or None
        Overlapping region of the second image.
    overlappingPoint : tuple or None
        Coordinates of the overlapping point.
    corrValue : float or None
        Correlation value between image1 and image2.
    showDicResults : bool
        Flag to show DIC results.
    showDicCorrelation : bool
        Flag to show DIC correlation.
    showDicPadError : bool
        Flag to show DIC padding error.
    calculateDICWithImage : bool
        Flag to calculate DIC with image data.
    calculateDICWithGrad : bool
        Flag to calculate DIC with gradient data.
    calculateDICWithHess : bool
        Flag to calculate DIC with Hessian data.
    mergeImageWithGrad : bool
        Flag to merge image with gradient data.
    mergeImageWithHess : bool
        Flag to merge image with Hessian data.
    padType : str
        Type of padding for images ('constant' by default).
    internalDisplacement : np.ndarray or None
        Internal displacements of image1.
    CValue : np.ndarray or None
        Correlation values across displacements.
    discplacementsCoordinates : tuple or None
        Coordinates of displacements.
    timestamp : str
        Current timestamp in 'YYYY-MM-DD HH.MM.SS' format.
    sUV0 : np.ndarray or None
        Initial displacement field for optimization.

    Methods:
    -------
    defineImage1(image: np.ndarray) -> None:
        Defines the first image for DIC.
    
    defineImage2(image: np.ndarray) -> None:
        Defines the second image for DIC.
    
    doubleThePad(image: np.ndarray) -> np.ndarray:
        Doubles the padding of the input image.
    
    doInternalDIC() -> None:
        Performs DIC computation internally.
    
    setOverlappingImages() -> None:
        Sets overlapping regions of image1 and image2.
    
    doSimpleExposureAdjustment() -> None:
        Performs simple exposure adjustment on both images.
    
    doMoreAdvancedExposureAdjustment() -> None:
        Performs advanced exposure adjustment if overlap difference is available.
    
    showOverlappingImages() -> None:
        Displays overlapping regions and their quality difference.
    
    showImages() -> None:
        Displays both images.
    
    adjustOvelapIfBad() -> None:
        Adjusts overlap if quality is insufficient.
    
    rollWindowInternallyAndDIC(stepLength: int, *windowSize: int, extraWiggle: int = 0) -> None:
        Rolls window internally and performs DIC calculation.
    
    showInternalQuiver(saveImages: bool = False) -> None:
        Displays quiver plot of internal displacement.
    
    setWarpedImage() -> None:
        Sets the warped image based on internal displacement.
    
    setWarpedImageGPT() -> None:
        Sets the warped image using an optimized displacement field.
    
    updateWindowSize() -> None:
        Updates window size based on displacement coordinates.
    
    saveFigures() -> None:
        Saves correlation map and merged images as PNG files.
    
    makeFiguresSavable(image: np.ndarray) -> np.ndarray:
        Converts image to a format suitable for saving as PNG.
    
    doWarping(order: int = 1, optimized: bool = False, saveQuiver: bool = False) -> None:
        Performs image warping based on internal displacement.
    
    optimizeWarping(o: int = 1) -> None:
        Optimizes image warping using error minimization.
    """
    def __init__ (self, showDicResults = False, showDicCorrelation = False, 
                  showDicPadError = False,
                  useImageAsPreProcess = True,
                  useGradAsPreProcess = False,
                  useHessAsPreProcess = False,
                  useImageCorrelation = True,
                  useGradCorrelation = False,
                  useHessCorrelation = False,
                  padType = 'constant'):
        self.image1 = None
        self.image2 = None
        self.image1_overlap = None
        self.image2_overlap = None
        self.overlappingPoint = None
        self.corrValue = None
        self.showDicResults = showDicResults
        self.showDicCorrelation = showDicCorrelation
        self.showDicPadError = showDicPadError
        self.useImageAsPreProcess = useImageAsPreProcess
        self.useGradAsPreProcess = useGradAsPreProcess
        self.useHessAsPreProcess = useHessAsPreProcess
        self.useImageCorrelation = useImageCorrelation
        self.useGradCorrelation = useGradCorrelation
        self.useHessCorrelation = useHessCorrelation
        self.padType = padType
        self.internalDisplacement = None
        self.CValue = None
        self.discplacementsCoordinates = None
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        self.sUV0 = None
        self.OptimizeResult = None
        self.bigDisplacement = None
        self.sUV0Unaffected = None
        self.isFiltered = False

    #Define Image1
    def defineImage1(self, image : np.ndarray):
        self.image1 = image.copy()
    
    #Define Image2
    def defineImage2(self, image : np.ndarray):
        self.image2 = image.copy()

    def doubleThePad(self, image):
        return dicTool.padImage(image, (image.shape[0]*2,image.shape[1]*2), padType='constant')


    def doInternalDIC(self):
        self.overlappingPoint, _, self.corrMap, self.merged, _ = dicTool.doDIC(self.image1, self.image2, showCorrelation=self.showDicCorrelation, showResult=self.showDicResults,
                showPadError=self.showDicPadError, padType = self.padType,
                useImageAsPreProcess = self.useImageAsPreProcess,
                useGradAsPreProcess = self.useGradAsPreProcess,
                useHessAsPreProcess = self.useHessAsPreProcess, 
                useImageCorrelation = self.useImageCorrelation,
                useGradCorrelation = self.useGradCorrelation,
                useHessCorrelation = self.useHessCorrelation)
        self.corrValue = np.max(self.corrMap)/np.sum(self.corrMap)
        
    def setOverlappingImages(self):
        self.image2_overlap,self.image1_overlap = dicTool.overlapRegion(self.image2, self.image1,  -self.overlappingPoint[1], self.overlappingPoint[0])
        self.overlap_quality = np.sqrt(np.abs(np.mean(self.image2_overlap, axis= -1)**2 - np.mean(self.image1_overlap, axis= -1)**2))

    def doSimpleExposureAdjustment(self):
       maxval = np.max([np.max(self.image1),np.max(self.image2)])
       dtype = self.image1.dtype
       self.image1 = self.image1.astype('float64')
       self.image2 = self.image2.astype('float64')
       self.image1 /= np.max(self.image1)
       self.image2 /= np.max(self.image2)
       self.image1 *= maxval
       self.image2 *= maxval
       self.image1 = self.image1.astype(dtype)
       self.image2 = self.image2.astype(dtype)
    
    def doMoreAdvancedExposureAdjustment(self):
        if self.overlapDiff is None or len(self.image1.shape) < 3:
            return None
        
        nextSmallest = np.argsort(np.reshape(self.overlapDiff,(self.overlapDiff.shape[0]*self.overlapDiff.shape[1],)))[0]
        inxs = np.unravel_index(nextSmallest, (self.overlapDiff.shape[0],self.overlapDiff.shape[1]))
        min_im_1 = self.image1[self.discplacementsCoordinates[0][inxs[0]][inxs[1]]:self.discplacementsCoordinates[0][inxs[0]][inxs[1]]+self.windowSizeY,
                  self.discplacementsCoordinates[1][inxs[0]][inxs[1]]:self.discplacementsCoordinates[1][inxs[0]][inxs[1]]+self.windowSizeX]
        min_im_2 = self.image2[self.discplacementsCoordinates[0][inxs[0]][inxs[1]]:self.discplacementsCoordinates[0][inxs[0]][inxs[1]]+self.windowSizeY,
                  self.discplacementsCoordinates[1][inxs[0]][inxs[1]]:self.discplacementsCoordinates[1][inxs[0]][inxs[1]]+self.windowSizeX]
        for c in range(self.image1.shape[-1]):
            maxval = np.max([np.max(min_im_1[:,:,c]),np.max(min_im_2[:,:,c])])
            dtype = self.image1.dtype
            a = self.image1[:,:,c].astype('float64')
            b = self.image2[:,:,c].astype('float64')
            a /= np.max(a)
            b /= np.max(b)
            a *= maxval
            b *= maxval
            self.image1[:,:,c] = a.astype(dtype)
            self.image2[:,:,c] = b.astype(dtype)


    def showOverlappingImages(self):
        if(self.image1_overlap.size > 0):
            plt.figure()
            plt.title('Image1 overlap')
            plt.imshow(self.image1_overlap)
            plt.figure()
        if(self.image2_overlap.size > 0):
            plt.title('Image2 overlap')
            plt.imshow(self.image2_overlap)
            plt.figure()
        if(self.image1_overlap.size > 0 and self.image2_overlap.size > 0):
            plt.title('Overlap difference: ' + str(np.mean(self.overlap_quality)))
            plt.imshow(self.overlap_quality, cmap = 'Greys_r')
            plt.colorbar()
        plt.show()
        
    def showImages(self):
        plt.figure()
        plt.title('Image1')
        plt.imshow(self.image1)
        plt.figure()
        plt.title('Image2')
        plt.imshow(self.image2)
        plt.show()

    def adjustOvelapIfBad(self):
        im1 = self.image1[:,self.image1.shape[1]//4:]
        im2 = self.image2[:,:self.image2.shape[1]*3//4]
        overlapFinding = DicClass(
                  useImageAsPreProcess = self.useImageAsPreProcess,
                  useGradAsPreProcess = self.useGradAsPreProcess,
                  useHessAsPreProcess = self.useHessAsPreProcess,
                  useImageCorrelation = self.useImageCorrelation,
                  useGradCorrelation = self.useGradCorrelation,
                  useHessCorrelation = self.useHessCorrelation,
                                showDicCorrelation=self.showDicCorrelation, showDicResults= self.showDicResults)
        overlapFinding.defineImage1(im1)
        overlapFinding.defineImage2(im2)
        overlapFinding.showImages()
        overlapFinding.doInternalDIC()
        overlapFinding.setOverlappingImages()
        overlapFinding.showOverlappingImages()
        if (np.mean(overlapFinding.overlap_quality) < np.mean(self.overlap_quality)):
            print('Better overlap found')
            self.overlappingPoint = (overlapFinding.overlappingPoint[0], overlapFinding.overlappingPoint[1] + self.image1.shape[1]//2)

    def rollWindowInternallyAndDIC(self, stepLength, *windowSize, extraWiggle = 0):
        if len(windowSize) >= 2:
            windowsizeX = windowSize[0]
            windowsizeY = windowSize[1]
        else:
            windowsizeX = windowSize
            windowsizeY = windowSize
        displacement = []
        Cvalue = []
        coordinates = []
        overlapDiff = []
        chosen_combo_mat = []
        combo_diff_mat = []
        im1 = dicTool.padImage(self.image1, (self.image1.shape[0] + mt.ceil(windowsizeY), self.image1.shape[1] + mt.ceil(windowsizeX)), padType='constant')
        im2 = dicTool.padImage(self.image2, (self.image2.shape[0] + mt.ceil(windowsizeY) + mt.ceil(extraWiggle), self.image2.shape[1] + mt.ceil(windowsizeX) + mt.ceil(extraWiggle)), padType='constant')
        
        y_length = len(range(0, im1.shape[0]-windowsizeY, stepLength))
        x_length = len(range(0, im1.shape[1]-windowsizeX, stepLength))
        total_length = y_length*x_length
        for j,y_s in enumerate(range(0, im1.shape[0]-windowsizeY, stepLength)):
            if y_s == 0:
                print('Started at time: ' + datetime.now().strftime('%Y-%m-%d %H.%M.%S'))
            for i,x_s in enumerate(range(0, im1.shape[1]-windowsizeX, stepLength)):
                subim1 = im1[y_s:y_s+windowsizeY,x_s:x_s+windowsizeX]
                subim2 = im2[y_s:y_s+windowsizeY+extraWiggle,x_s:x_s+windowsizeX+extraWiggle]

                if np.all(subim1 == 0) or np.all(subim2 == 0):
                    displacement.append((0,0))
                    Cvalue.append(np.nan)
                    coordinates.append([y_s,x_s])
                    overlapDiff.append(np.nan)
                    chosen_combo_mat.append([False,False,False,False,False,False])
                    combo_diff_mat.append(0)
                    continue
                
                subDic, chosen_combo, combo_diff = findBestDIC(subim1,subim2)
                combo_diff_mat.append(combo_diff)
                chosen_combo_mat.append(chosen_combo)
                meandiff1 = np.mean(dicTool.sobelItUp(subDic.image1_overlap))
                if np.isnan(meandiff1):
                    meandiff1 = 1
                meandiff2 = np.mean(dicTool.sobelItUp(subDic.image2_overlap))
                if np.isnan(meandiff2):
                    meandiff2 = 1
                displacement.append([subDic.overlappingPoint[0], subDic.overlappingPoint[1]])
                Cvalue.append(subDic.corrValue)
                coordinates.append([y_s,x_s])
                overlapDiff.append(np.mean(np.linalg.norm(np.mean(subDic.image2_overlap, axis = -1)**2-np.mean(subDic.image1_overlap, axis = -1)**2))*(1+(1-meandiff1)+(1-meandiff2)))
                total_step = i+(x_length*j)
                if total_step == (total_length)//4:
                    print('Woooo, we are 25% way there!, at time: ' + datetime.now().strftime('%Y-%m-%d %H.%M.%S'))
                elif total_step == (total_length)//2:
                    print('Woooo, we are half way there!, at time: ' + datetime.now().strftime('%Y-%m-%d %H.%M.%S'))
                elif total_step == (total_length*3)//4:
                    print('Woooo, we are 75% way there!, at time: ' + datetime.now().strftime('%Y-%m-%d %H.%M.%S'))
        print('Ended at time: ' + datetime.now().strftime('%Y-%m-%d %H.%M.%S'))
        self.discplacementsCoordinates = (np.reshape(np.array(coordinates)[:,0],(y_length,x_length)),np.reshape(np.array(coordinates)[:,1],(y_length,x_length)))
        self.internalDisplacement = (-np.reshape(np.array(displacement)[:,0],(y_length,x_length)),-np.reshape(np.array(displacement)[:,1],(y_length,x_length)))
        self.CValue = np.reshape(np.array(Cvalue),(y_length,x_length))
        self.chosenComboMat = np.reshape(np.array(chosen_combo_mat),(y_length,x_length, 6))
        self.overlapDiff = np.reshape(np.array(overlapDiff),(y_length,x_length))
        self.windowSizeX = windowsizeX
        self.windowSizeY = windowsizeY
        self.comboDiffMat =  np.reshape(np.array(combo_diff_mat),(y_length,x_length))

    def showInternalQuiver(self, saveImages = False):
        fig, axs = plt.subplots(1,3,figsize=(15*3,13), constrained_layout = True)

        axs[0].set_title('Image to be compared')
        axs[0].imshow(self.image1)
        nonZero = np.nonzero(np.sum(self.chosenComboMat, axis = -1))
        axs[1].set_title('Quiver')
        axs[1].imshow(self.image1)
        axs[1].quiver(self.discplacementsCoordinates[1][nonZero], self.discplacementsCoordinates[0][nonZero], self.internalDisplacement[1][nonZero], self.internalDisplacement[0][nonZero], color='r')
        axs[1].axis([np.min(self.internalDisplacement[1][nonZero]), self.image1.shape[1]+10, np.min(self.internalDisplacement[0][nonZero]), self.image1.shape[0]+10])
        axs[1].invert_yaxis()

        axs[2].set_title('Reference Image')
        axs[2].imshow(self.image2)
        if saveImages:
            #fig.subplots_adjust(wspace = 0.005, hspace=0.005, left = 0.005, bottom=0.005)
            fig.savefig('Figures/Show_quiver %s.png' % self.timestamp)
    
    def setWarpedImage(self):
        if(self.discplacementsCoordinates is None):
            return
        maxYDisplacement = mt.ceil(np.max(np.abs(self.internalDisplacement[0])))
        maxXDisplacement = mt.ceil(np.max(np.abs(self.internalDisplacement[1])))
        newImage = np.zeros([self.image1.shape[0]+2*maxYDisplacement,
                              self.image1.shape[1]+2*maxXDisplacement,3], dtype=self.image1.dtype)
        for j,y in enumerate(self.discplacementsCoordinates[0][:,0].astype(int)):
            for i,x in enumerate(self.discplacementsCoordinates[1][0,:].astype(int)):
                newImage[maxYDisplacement + y + mt.ceil(self.internalDisplacement[0][j,i]):
                          maxYDisplacement + y + mt.ceil(self.internalDisplacement[0][j,i]) + self.windowSizeY,
                         maxXDisplacement + x + mt.ceil(self.internalDisplacement[1][j,i]):
                           maxXDisplacement + x + mt.ceil(self.internalDisplacement[1][j,i]) + self.windowSizeX,
                           :] = self.image1[y: y + self.windowSizeY,x: x + self.windowSizeX,:]
        self.warpedImage = newImage
        
    def updateWindowSize(self):
        self.windowSizeX = int(self.discplacementsCoordinates[1][1,0] - self.discplacementsCoordinates[1][0,0])
        self.windowSizeY = int(self.discplacementsCoordinates[0][0,1] - self.discplacementsCoordinates[0][0,0])

    def saveFigures(self):
        cv.imwrite('Figures/Correlation_map %s.png' % self.timestamp, self.makeFiguresSavable(self.corrMap[::-1,::-1]));
        cv.imwrite('Figures/Merged %s.png' % self.timestamp, self.makeFiguresSavable(self.merged));

    def makeFiguresSavable(self,image):
        imageMap = np.zeros((image.shape[0],image.shape[1],3))
        if len(image.shape) < 3:
            imageMap[:,:,0] = (((image-np.min([np.min(image),0]))/np.max(image-np.min([np.min(image),0])))*255).astype(int)
            imageMap[:,:,1] = (((image-np.min([np.min(image),0]))/np.max(image-np.min([np.min(image),0])))*255).astype(int)
            imageMap[:,:,2] = (((image-np.min([np.min(image),0]))/np.max(image-np.min([np.min(image),0])))*255).astype(int)
        else:
            imageMap[:,:,0] = (((image[:,:,0] -np.min([np.min(image[:,:,0] ),0]))/np.max(image[:,:,0] -np.min([np.min(image[:,:,0] ),0])))*255).astype(int)
            imageMap[:,:,1] = (((image[:,:,1] -np.min([np.min(image[:,:,1] ),0]))/np.max(image[:,:,1] -np.min([np.min(image[:,:,1] ),0])))*255).astype(int)
            imageMap[:,:,2] = (((image[:,:,2] -np.min([np.min(image[:,:,2] ),0]))/np.max(image[:,:,2] -np.min([np.min(image[:,:,2] ),0])))*255).astype(int)
        return imageMap

    def doWarping(self, order = 1, optimized = False, saveQuiver = False, showFigures = False, p2 = None):
        
        warpSetting = DicClass(useImageAsPreProcess = self.useImageAsPreProcess,
                  useGradAsPreProcess = self.useGradAsPreProcess,
                  useHessAsPreProcess = self.useHessAsPreProcess,
                  useImageCorrelation = self.useImageCorrelation,
                  useGradCorrelation = self.useGradCorrelation,
                  useHessCorrelation = self.useHessCorrelation)
        warpSetting.defineImage1(self.image1)
        warpSetting.defineImage2(self.image2)
        warpSetting.windowSizeX = self.windowSizeX
        warpSetting.windowSizeY = self.windowSizeY
        warpSetting.internalDisplacement = self.internalDisplacement
        warpSetting.discplacementsCoordinates = self.discplacementsCoordinates
        sUV = None
        if not optimized:
            new_x = np.linspace(0, warpSetting.image1.shape[1], warpSetting.image1.shape[1])#[:-1]#, smallerDIC33.image1.shape[1])
            new_y = np.linspace(0, warpSetting.image1.shape[0], warpSetting.image1.shape[0])#[:-1]#, smallerDIC3.image1.shape[0])

            new_X,new_Y = np.meshgrid(new_x,new_y)

            sUV,basis = dicTool.interpolateQuiver(*warpSetting.discplacementsCoordinates, warpSetting.internalDisplacement, order)
            basis = np.array(dicTool.getBasis(new_X.ravel(), new_Y.ravel(), order))
            fitU,fitV = dicTool.getFit(new_X,new_Y,sUV,basis, order)
            if self.bigDisplacement is not None or p2 is not None:
                if p2 is not None:
                    order_here = dicTool.findTriangle(len(p2[0]))
                    basis = dicTool.getBasis(warpSetting.discplacementsCoordinates[1],warpSetting.discplacementsCoordinates[0],order_here)
                    A = np.vstack(np.array([basis])).T
                    dicY = A@p2[1]
                    dicX = A@p2[0]
                else:
                    dicX,dicY = warpSetting.discplacementsCoordinates[1].copy() + self.bigDisplacement[1]/2, warpSetting.discplacementsCoordinates[0].copy() + self.bigDisplacement[0]/2
                displacedX,displacedY = warpSetting.internalDisplacement[0].copy(), warpSetting.internalDisplacement[1].copy()

                sUVUnaffected, _ = dicTool.interpolateQuiver(dicX, dicY, (displacedX, displacedY), order)
                self.internalDisplacementUnaffected = [displacedX,displacedY]
                self.discplacementsCoordinatesUnaffected = [dicY,dicX]
                self.sUV0Unaffected = sUVUnaffected
        else: 
            fitU,fitV = self.newInternalDisplacement
            new_Y, new_X = self.newDiscplacementsCoordinates
            sUV = self.sUV0
            basis = self.fitBasis
        if showFigures:
            fig = plt.figure(figsize=(10,10))
            plt.title('Interpolated Quiver')
            plt.imshow(warpSetting.image1)
            plt.quiver(self.discplacementsCoordinates[1], self.discplacementsCoordinates[0], fitU[*self.discplacementsCoordinates], fitV[*self.discplacementsCoordinates], color='r')

            if saveQuiver:
                fig.savefig('Figures/Interpolated quiver %s.png' % self.timestamp)
            plt.show()

        warpSetting.internalDisplacement = [fitU,fitV]
        warpSetting.discplacementsCoordinates = [new_Y,new_X]

        warpSetting.updateWindowSize()

        self.warpedImage = dicTool.warp_image(warpSetting.image1, -fitU, fitV)


        warpSetting.defineImage1(self.warpedImage)
        warpSetting.defineImage2(self.image2_overlap)
        warpSetting.doInternalDIC()
        warpSetting.setOverlappingImages()
        if not optimized:
            warpSetting.internalDisplacement = [fitU,fitV]
            self.newInternalDisplacement = [fitU,fitV]
            warpSetting.discplacementsCoordinates = [new_Y,new_X]
            self.newDiscplacementsCoordinates = [new_Y,new_X]
            self.warped_image1_overlap = warpSetting.image1_overlap
            self.warped_image2_overlap = warpSetting.image2_overlap
            self.sUV0 = sUV
            self.fitBasis = basis

    def optimizeWarping(self, o = 1):
        basis = dicTool.getBasis(*(self.discplacementsCoordinates[1], self.discplacementsCoordinates[0]),max_order=o)
        uvShape = (len(basis), 2)
        if (self.sUV0 is None or len(self.sUV0.ravel()) != uvShape[0]*uvShape[1]):
            UV = np.zeros(uvShape)
        else:
            UV = np.reshape(self.sUV0.ravel(), uvShape)
        error1 = error_minimization(UV, self, o, uvShape)
        result = sc.optimize.minimize(error_minimization, UV.ravel(), (self, o, uvShape), 'Powell', tol = 1e-18,options={"maxiter":1e12})
        error2 = error_minimization(result.x, self, o, uvShape)
        if (not result.success or error1 < error2):
            print('Did not finish')
        self.OptimizeResult = result
        self.newInternalDisplacement = dicTool.getFit(self.newDiscplacementsCoordinates[1], self.newDiscplacementsCoordinates[0], np.reshape(result.x, uvShape), o = o)

    def optimizeWarpingUnaffected(self, o = 1):
        basis = dicTool.getBasis(*(self.internalDisplacementUnaffected[1], self.internalDisplacementUnaffected[0]),max_order=o)
        uvShape = (len(basis), 2)
        if (self.sUV0Unaffected is None or len(self.sUV0Unaffected.ravel()) != uvShape[0]*uvShape[1]):
            UV = np.zeros(uvShape)
        else:
            UV = np.reshape(self.sUV0.ravel(), uvShape)
        error1 = error_minimizationUnaffected(UV, self, o, uvShape)
        result = sc.optimize.minimize(error_minimizationUnaffected, UV.ravel(), (self, o, uvShape), 'Powell', tol = 1e-18, options={"maxiter":1e12})
        error2 = error_minimizationUnaffected(result.x, self, o, uvShape)
        if (not result.success or error1 < error2):
            print('Did not finish')
        self.OptimizeResultUnaffected = result
        #self.newInternalDisplacement = dicTool.getFit(self.newDiscplacementsCoordinates[1], self.newDiscplacementsCoordinates[0], np.reshape(result.x, uvShape), o = o)

    def rollWindowInternallyAndDIC_parallel(self, stepLength, *windowSize, extraWiggle=0):
            print('Started at time: ' + datetime.now().strftime('%Y-%m-%d %H.%M.%S'))
            if len(windowSize) >= 2:
                windowsizeX = windowSize[0]
                windowsizeY = windowSize[1]
            else:
                windowsizeX = windowSize
                windowsizeY = windowSize
            
            # Pad images here as before
            im1 = dicTool.padImage(self.image1, 
                                   (self.image1.shape[0] + np.ceil(windowsizeY).astype(int),
                                     self.image1.shape[1] + np.ceil(windowsizeX).astype(int)),
                                   padType='constant')
            im2 = dicTool.padImage(self.image2,
                                   (self.image2.shape[0] + np.ceil(windowsizeY).astype(int) + np.ceil(extraWiggle).astype(int),
                                     self.image2.shape[1] + np.ceil(windowsizeX).astype(int) + np.ceil(extraWiggle).astype(int)),
                                   padType='constant')
            
            y_positions = list(range(0, im1.shape[0] - windowsizeY, stepLength))
            x_positions = list(range(0, im1.shape[1] - windowsizeX, stepLength))
            
            results = []

            with ProcessPoolExecutor() as executor:
                for y_chunk, x_chunk in chunk_positions(y_positions, x_positions, 10):  # Adjust chunk_size as needed
                    for y_s in y_chunk:
                        for x_s in x_chunk:
                            results.append(executor.submit(process_patch, y_s, x_s, im1, im2, windowsizeY, windowsizeX, extraWiggle))
                
                processed_results = [future.result() for future in results]

            # Extracting results and organizing into the necessary outputs as before
            displacement = []
            Cvalue = []
            coordinates = []
            overlapDiff = []
            chosen_combo_mat = []
            combo_diff_mat = []

            for result in processed_results:
                displacement.append(result['displacement'])
                Cvalue.append(result['Cvalue'])
                coordinates.append(result['coordinates'])
                overlapDiff.append(result['overlapDiff'])
                chosen_combo_mat.append(result['chosen_combo'])
                combo_diff_mat.append(result['combo_diff'])


            # Rest of the processing as in your original method
            self.discplacementsCoordinates = (
                np.reshape(np.array(coordinates)[:, 0], (len(y_positions), len(x_positions))),
                np.reshape(np.array(coordinates)[:, 1], (len(y_positions), len(x_positions)))
            )
            self.internalDisplacement = (
                -np.reshape(np.array(displacement)[:, 0], (len(y_positions), len(x_positions))),
                -np.reshape(np.array(displacement)[:, 1], (len(y_positions), len(x_positions)))
            )
            self.CValue = np.reshape(np.array(Cvalue), (len(y_positions), len(x_positions)))
            self.chosenComboMat = np.reshape(np.array(chosen_combo_mat), (len(y_positions), len(x_positions), 6))
            self.overlapDiff = np.reshape(np.array(overlapDiff), (len(y_positions), len(x_positions)))
            self.windowSizeX = windowsizeX
            self.windowSizeY = windowsizeY
            self.comboDiffMat = np.reshape(np.array(combo_diff_mat), (len(y_positions), len(x_positions)))
            executor.shutdown() 
            print('Ended at time: ' + datetime.now().strftime('%Y-%m-%d %H.%M.%S'))

    def filterMissMatched(self, self_like = None, ransac_tol = 30):
        if self_like is None:
            self_like = self
            
        vec1 = np.array([self_like.discplacementsCoordinates[0].ravel(),
                         self_like.discplacementsCoordinates[1].ravel()])
        vec2 = np.array([self_like.discplacementsCoordinates[0].ravel()+self_like.internalDisplacement[0].ravel(),
                         self_like.discplacementsCoordinates[1].ravel()+self_like.internalDisplacement[1].ravel()])

        labels = cv.findHomography(vec1.T, vec2.T,cv.RANSAC,ransac_tol)[1].reshape(self_like.discplacementsCoordinates[0].shape)

        overlap_bool = labels == 1 * ~np.isnan(self_like.CValue)

        self.internalDisplacement = (np.zeros_like(self_like.internalDisplacement[0]),np.zeros_like(self_like.internalDisplacement[0]))
        self.internalDisplacement[0][overlap_bool] = self_like.internalDisplacement[0][overlap_bool]
        self.internalDisplacement[1][overlap_bool] = self_like.internalDisplacement[1][overlap_bool]
        
        self.discplacementsCoordinates = (np.zeros_like(self_like.discplacementsCoordinates[0]),np.zeros_like(self_like.discplacementsCoordinates[0]))
        self.discplacementsCoordinates[0][overlap_bool] = self_like.discplacementsCoordinates[0][overlap_bool]
        self.discplacementsCoordinates[1][overlap_bool] = self_like.discplacementsCoordinates[1][overlap_bool]
        
        self.chosenComboMat = np.zeros_like(self_like.chosenComboMat)
        self.chosenComboMat[overlap_bool,:] = self_like.chosenComboMat[overlap_bool,:]
        
        self.comboDiffMat = np.zeros_like(self_like.comboDiffMat)
        self.comboDiffMat[overlap_bool] = self_like.comboDiffMat[overlap_bool]
        
        self.CValue = np.zeros_like(self_like.CValue)
        self.CValue[overlap_bool] = self_like.CValue[overlap_bool]
        
        self.overlapDiff = np.zeros_like(self_like.overlapDiff)
        self.overlapDiff[overlap_bool] = self_like.overlapDiff[overlap_bool]
        
        self.isFiltered = True

def chunk_positions(y_positions, x_positions, chunk_size):
                for i in range(0, len(y_positions), chunk_size):
                    for j in range(0, len(x_positions), chunk_size):
                        yield y_positions[i:i+chunk_size], x_positions[j:j+chunk_size]

def process_patch(y_s, x_s, im1, im2, windowsizeY, windowsizeX, extraWiggle):
                subim1 = im1[y_s:y_s + windowsizeY, x_s:x_s + windowsizeX]
                subim2 = im2[y_s:y_s + windowsizeY + extraWiggle, x_s:x_s + windowsizeX + extraWiggle]

                if np.all(subim1 == 0) or np.all(subim2 == 0):
                    return {
                        'displacement': (0, 0),
                        'Cvalue': np.nan,
                        'coordinates': [y_s, x_s],
                        'overlapDiff': np.nan,
                        'chosen_combo': [False, False, False, False, False, False],
                        'combo_diff' : 0
                    }
                
                subDic, chosen_combo, combo_diff = findBestDIC_parallel(subim1, subim2)
                
                return {
                    'displacement': [subDic.overlappingPoint[0], subDic.overlappingPoint[1]],
                    'Cvalue': subDic.corrValue,
                    'coordinates': [y_s, x_s],
                    'overlapDiff': np.mean(np.mean(subDic.image2_overlap, axis=-1) - np.mean(subDic.image1_overlap, axis=-1)),
                    'chosen_combo': chosen_combo,
                    'combo_diff' : combo_diff
                }

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
    U,V = dicTool.getFit(dicClass.discplacementsCoordinates[1], dicClass.discplacementsCoordinates[0], UV, o = o)
    return (dicTool.imageError(np.array([dicClass.internalDisplacement[0],dicClass.internalDisplacement[1]]), 
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
    U,V = dicTool.getFit(dicClass.discplacementsCoordinatesUnaffected[1], dicClass.discplacementsCoordinatesUnaffected[0], UV, o = o)
    return (dicTool.imageError(np.array([dicClass.internalDisplacementUnaffected[0],dicClass.internalDisplacementUnaffected[1]]), 
                       np.array([V, U])))

def findBestDIC(image1,image2, a = 0):
    for i,comb in enumerate(dicTool.combinations):
        innerDIC = DicClass(showDicResults=False, showDicCorrelation=False,
                                      useImageAsPreProcess = comb[0],
                                      useGradAsPreProcess = comb[1],
                                      useHessAsPreProcess = comb[2],
                                      useImageCorrelation = comb[3],
                                      useGradCorrelation = comb[4],
                                      useHessCorrelation = comb[5])
        innerDIC.defineImage1(image1)
        innerDIC.defineImage2(image2)
        innerDIC.doSimpleExposureAdjustment()
        innerDIC.doInternalDIC()
        innerDIC.setOverlappingImages()
        

        innerDIC_bool = innerDIC.overlap_quality != 0

        
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
    if a == 0:
        return a, dicTool.combinations[0],0
    
    return a, dicTool.combinations[chosen_combo],abs_diff

def findBestDIC_parallel(image1, image2, a=0):
    def process_combination(i, comb):
        try:
            if not np.any(comb):
                print('Combination #%d failed' %i)
                return (None, None, None)
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
            return (None,None,None)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_combination, i, comb) for i, comb in enumerate(dicTool.combinations)]
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
        return a, dicTool.combinations[0], 0

    return a, dicTool.combinations[chosen_combo], abs_diff