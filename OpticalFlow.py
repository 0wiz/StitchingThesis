import resources.dic_tools as dicTool # type: ignore
import numpy as np

showFlow = False
debug = False

def OpticalFlow(image1, image2, ransac_tol=30, rollingWindowStepSize=25,
                 windowSizeX=50, windowSizeY=50, extraWiggle=10, order=2):
    global showFlow, debug
    if debug:
        showFlow = True

    print('Beginning Optical Flow')
    bigPicDic, _, _ = dicTool.findBestDIC(image1, image2)
    print('Overlap Found')
    smallerDIC, _, _ = dicTool.findBestDIC(bigPicDic.image1_overlap, bigPicDic.image2_overlap)

    print('Beginning Rolling Window')
    smallerDIC.rollWindowInternallyAndDIC(rollingWindowStepSize, windowSizeX, windowSizeY, extraWiggle)
    if debug:
        dicTool.showInternalResults(smallerDIC)
    smallerDIC.filterMissMatched(None, ransac_tol)
    smallerDIC.doWarping(order, saveQuiver=False, showFigures=showFlow)
    smallerDIC.optimizeWarping(order)
    smallerDIC.doWarping(optimized=True, saveQuiver=False, showFigures=showFlow)
    c = smallerDIC.OptimizeResult.x.reshape(smallerDIC.sUV0.shape)
    c_dis = np.array([c[:,0], -c[:,1]])
    c_no_dis = np.zeros_like(c_dis)
    return c_dis, c_no_dis