import resources.dic_tools as dicTool # type: ignore
import resources.dic_Class as dicClass # type: ignore
import numpy as np

showFlow = False
debug = False

def OpticalFlow(image1, image2, ransac_tol = 30, rollingWindowStepSize = 25,
                 windowSizeX = 50, windowSizeY = 50, extraWiggle = 10, order = 2):
    global showFlow, debug
    if debug:
        showFlow = True

    print('Begining Optical Flow')
    bigPicDic, _, _ = dicClass.findBestDIC_parallel(image1, image2)
    print('Overlap Found')
    smallerDIC, _, _ = dicClass.findBestDIC_parallel(bigPicDic.image1_overlap, bigPicDic.image2_overlap)

    print('Begining Rolling window')
    smallerDIC.rollWindowInternallyAndDIC_parallel(rollingWindowStepSize, windowSizeX, windowSizeY, extraWiggle)
    if debug:
        smallerDIC.showInternalResults()
    smallerDIC.filterMissMatched(None, ransac_tol)
    smallerDIC.doWarping(order, saveQuiver = False, showFigures = showFlow)
    smallerDIC.optimizeWarping(order)
    smallerDIC.doWarping(optimized=True, saveQuiver=False, showFigures = showFlow)
    c = smallerDIC.OptimizeResult.x.reshape(smallerDIC.sUV0.shape)
    c_dis = np.array([c[:,0],-c[:,1]])
    c_no_dis = np.zeros_like(c_dis)
    return c_dis, c_no_dis