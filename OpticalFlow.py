# Local
import resources.dic_tools as flowTools # type: ignore

# Math
import numpy as np

def OpticalFlow(img1, img2, ransac_tol=30, rollingWindowStepSize=25,
                 windowSizeX=50, windowSizeY=50, extraWiggle=10, order=2):

    print('Beginning Optical Flow')
    bigPicDic, _, _ = flowTools.findBestGPC(img1, img2)
    print('Overlap Found')
    smallerDIC, _, _ = flowTools.findBestGPC(bigPicDic.img1_overlap, bigPicDic.img2_overlap)

    print('Beginning Rolling Window')
    smallerDIC.rollWindowAndFindBestGPC(rollingWindowStepSize, windowSizeX, windowSizeY, extraWiggle)
    smallerDIC.filterMissMatched(None, ransac_tol)
    smallerDIC.warp(order)
    smallerDIC.optimizeWarping(order)
    smallerDIC.warp(optimized=True)
    c = smallerDIC.OptimizeResult.x.reshape(smallerDIC.sUV0.shape)
    c_dis = np.array([c[:,0], -c[:,1]]).T
    c_no_dis = np.zeros_like(c_dis)
    return c_dis, c_no_dis