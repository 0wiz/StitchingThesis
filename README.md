## Experimentation code for thesis "Image Stitching for Video: Feature Detection & Optical Flow"

**To be added:** Publication link and doi.

### Abstract

In panoramic video surveillance, image stitching techniques are expanded to leverage the additional potential offered by the temporal element of video in static environments. This thesis explores image stitching methods relevant to video stitching without conducting direct video stitching experiments, seeking to formalize and enhance the foundational understanding of certain conventions within the field. A conventional feature detection method, Scale-Invariant Feature Transform, is compared with a model of optical flow, Digital Image Correlation, to evaluate how these methods complement each other in determining the offset between views. Using this offset data as error, the images are aligned with two warping functions, where a second-degree polynomial per axis is fit to the offsets through parameter optimization. The methods are first assessed independently and then in combination.

The results demonstrate the effectiveness of these methods in ideal conditions, validating their ability to address stitching tasks. The appendices present challenges encountered under various conditions, including scenes with regions of hardly detectable features, varying overlap sizes between fields of view, and the presence or absence of lens distortion correction.

The main conclusion of the project is that a hybrid approach, where the output from a feature detection based algorithm is finalized with a optical flow based algorithm, enhances the robustness of the stitching process, allowing for a more accurate and coherent mosaic of the panoramic scene.

<br>

### The Code

The interoperability between similar functions and coherence in code style is somewhat limited, as the experimentation of the two approaches was split and developed separately. The decision to publish the code came later. Most optical flow functions are contained within ``resources/DIC.py`` and ``resources/FlowTools.py``, while ``resources/Tools.py`` holds more general functions, as well as feature detection specifics used in the combined approach.

GitHub can fail to render the output of jupyter notebooks, so the following links can be used to view the most recent uploaded output results:

``CombinedApproach.ipynb`` https://nbviewer.org/github/0wiz/StitchingThesis/blob/a99b6f2b377a6a9b0fc152172e0f899bcb37c5e0/CombinedApproach.ipynb

``FeatureDetection.ipynb`` https://nbviewer.org/github/0wiz/StitchingThesis/blob/a99b6f2b377a6a9b0fc152172e0f899bcb37c5e0/FeatureDetection.ipynb

``OpticalFlow.ipynb`` https://nbviewer.org/github/0wiz/StitchingThesis/blob/a99b6f2b377a6a9b0fc152172e0f899bcb37c5e0/OpticalFlow.ipynb