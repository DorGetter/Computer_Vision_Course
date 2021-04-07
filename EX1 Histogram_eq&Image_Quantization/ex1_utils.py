"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List

'''
Returns my ID 
Dor Getter - 313301327
'''

def myID() -> np.int:
    return 313301327


# --------------------------------------------------------------------------------------------------------- #
# Simple utils functions :

def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    '''Reads an image, and returns the image converted as requested
    :param filename: The path to the image , :param representation: grayscale(1) or RGB(2)
    :return: The image np array.
    output image is represented by a matrix of class np.float with intensities
     (either grayscale or RGB channel intensities) normalized to the range [0, 1].'''

    if representation == LOAD_RGB:
        # Read in image
        try:
            img = cv2.imread(filename, 1)  # Open the image from path. flag = 1 to color.
        except:
            print('Could not read image at location:', filename)
        # Convert the input to an array of type float32.
        pixels = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    elif representation == LOAD_GRAY_SCALE:
        try:
            img = cv2.imread(filename, 0)  # Open the image from path. flag = 1 to grayscale.
        except:
            print('Could not read image at location:', filename)
        pixels = np.asarray(img, dtype=np.float32)  # Convert the input to an array of type float32.

    pixels /= 255.0  # Normalize the intensities.

    #                           <- checks ->
    # print('Data Type: %s' % pixels.dtype)                           # Check that pixels is in float
    # print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))    # Check intensities level.
    # cv2.imshow('img', pixels)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(type(pixels))
    return pixels





def imDisplay(filename: str, representation: int):
    '''Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: None'''
    img = imReadAndConvert(filename, representation)
    if representation == LOAD_RGB:
        plt.imshow(img)
    elif representation == LOAD_GRAY_SCALE:
        plt.imshow(img , 'gray')

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


# --------------------------------------------------------------------------------------------------------- #
# YIQ channel functions :
'''
BackGround - YIQ signal:
````````````````````````
*   YIQ color model is a color model used by American broadcasters for their signal format of NTSC.
    This is because the signals cannot be sent in RGB format as RGB to television monitors as the monitors
    uses a single composite signal, and for sending RGB format it require different signals. 
*   So by using YIQ which separates the intensity from the color information it makes the YIQ color 
    space very attractive to TV broadcasting, because it helps maintain compatibility with monochrome TV standards.

YIQ color space: 
*   Y component -  consist of the Luminescence. 
    decides how much the brightness of the particular pixel position is going to be. 
    To achieve the luminescence from RGB, the Red Green and Blue intensities are chosen to yield a standard
    luminosity curve. (Black & White televisions need only the Y component from the YIQ signal).
*   I component - consist of Orange  -> Cyan hue information. (Basically it has flesh tone shading info).
*   Q component - consist of Magenta -> Green hue information. 

The YIQ model also takes advantage of the fact that the human eye is more sensitive to changes in luminance than 
changes to hue or saturation. The Y, I, Q components are assumed to be in the [0, 1] or [0, 255] range. 

transformation (NTSC encoder/decoder):
``````````````````````````````````````
*   RGB -> YIQ :
    To achieve the luminescence from RGB, the Red Green and Blue intensities are chosen to yield a standard
    luminosity curve.  
    The RGB to YIQ conversion is defined as:
    | [Y] |        | [0.299  0.587   0.114] |     | [R] |
    | [I] |   =    | [0.596 -0.275  -0.321] |  *  | [G] |
    | [Q] |        | [0.212 -0.523   0.311] |     | [B] |

*   YIQ -> RGB :
    The YIQ to RGB conversion is defined as:
    | [R] |        | [1.000  0.956   0.620] |     | [Y] |
    | [G] |   =    | [1.000 -0.272  -0.647] |  *  | [I] |
    | [B] |        | [1.000 -1.108   1.705] |     | [Q] |

    explain --YIQ 2 RGB--
    Done by inverse matrix transformation of the formula above:
                              -1  
    | [0.299  0.587   0.114] |      | [1.000  0.956   0.620] |
    | [0.596 -0.275  -0.321] |  =   | [1.000 -0.272  -0.647] |
    | [0.212 -0.523   0.311] |      | [1.000 -1.108   1.705] |
'''

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    '''Transform an RGB image into the YIQ color space.
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space'''

    # RGB to YIQ conversion
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgRGB.shape  # Saving the input image shape to later recovery.
    # Applying matrix multiplication.
    return np.dot(imgRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(OrigShape)
    pass

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space"""


    # RGB to YIQ conversion formula:
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgYIQ.shape  # Saving the input image shape to later recovery.
    multi_invers_mat = np.linalg.inv(yiq_from_rgb)  # Compute the (multiplicative) inverse of a matrix.
    # Applying matrix multiplication.
    return np.dot(imgYIQ.reshape(-1, 3), multi_invers_mat.transpose()).reshape(OrigShape)


# --------------------------------------------------------------------------------------------------------- #
# Histogram functions :
'''
BackGround - Histogram:
```````````````````````
*   Definition - Given a digital image with intensity levels [0-L], a histogram is a 
    representation (graph or plot) of the frequency of occurrence of each pixel intensity of an image.

*   Histogram can be used for various purposes such as image enhancement and image segmentation by
    plotting the frequency and the cumulative frequency for the intensity values of an image. 

*   Histogram compose of:  
    () Y axis - The overall number of pixels.   ( size of the img matrix ).
    () X axis - The intensity values.           ( [0-255] ). 
    The histogram has a vertical bar for each integer [0-255] of the X axis and the height of the bar
    yield the number of pixels that contain the corresponding intensity.
    
Histogram Equalize Quantization : 
`````````````````````````````````
*   Histogram equalization is a method in image processing of contrast adjustment using the image's histogram.
*   usually increases the global contrast of many images, especially when the usable data of the image is represented
    by close contrast values. Through this adjustment, the intensities can be better distributed on the histogram.
    This allows for areas of lower local contrast to gain a higher contrast. Histogram equalization accomplishes 
    this by effectively spreading out the most frequent intensity values.
'''

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)"""

    # Grayscale case -
    if len(imgOrig.shape) == 2:
        img = imgOrig * 255                         # de-normalized the pixels intensity.
        # Getting the Original histogram.
        histOrig, bins_org = np.histogram(img.flatten(), 256, [0, 255])
        # Calculating the cumsum of the Original histogram. ( cdf = 0 is ignored).
        cdf = histOrig.cumsum()
        # Mask the cdf where equal to 0. (All operations will not perform masked elements).
        cdf_m = np.ma.masked_equal(cdf, 0)
        # Find the minimum histogram value (excluding 0) and apply the histogram equalization equation.
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        # masked data replaced by 0.
        cdf = np.ma.filled(cdf_m, 0)
        # Getting image after equalization.
        imgEq = cdf[img.astype('uint8')]
        # Getting the histogram after equalization.
        histEq, bins_eq = np.histogram(imgEq.flatten(), 256, [0, 256])

    # RGB case -
    # A more general approach would be transforming RGB values into another space
    # that contains a luminescence/intensity value in this case YIQ will be used,
    # apply histogram equalize only in intensity plane and perform the inverse transform.
    elif len(imgOrig.shape) == 3:
        img_yiq = transformRGB2YIQ(imgOrig)
        # de-normalize pixels intensity.
        img_yiq[:, :, 0] = img_yiq[:, :, 0] * 255
        # Getting the Original histogram.
        histOrig, bins_org = np.histogram(img_yiq[:, :, 0].flatten(), 256, [0, 255])
        cdf = histOrig.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        img_yiq[:, :, 0] = cdf[img_yiq[:, :, 0].astype('uint8')]
        histEq, bins_eq = np.histogram(img_yiq[:, :, 0].flatten(), 256, [0, 256])
        img_yiq[:, :, 0] = img_yiq[:, :, 0] / 255
        imgEq = transformYIQ2RGB(img_yiq)

    else:
        raise ValueError('Unsupported representation. only RGB or Greyscale images allowed')

    return imgEq, histOrig, histEq
    pass


# --------------------------------------------------------------------------------------------------------- #
# Quantization functions :

'''
BackGround - Quantization:
```````````````````````
*   Quantization is the process of mapping a large set of input values to output values in a countable (smaller) set.
*   Quantization process "damaged" the image quality in exchange for compression.
    The goal is to minimize this damage as much as possible.
'''
def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])"""

    # Grayscale case -
    # ````````````````
    if len(imOrig.shape) == 2:
        img = imOrig * 255              # de-normalized the pixels intensity.
                                        # Generate the histogram.
        histOrig, bins = np.histogram(img.flatten(), 256)
        bins = np.arange(0, 256)
                                        # initialize z - the borders which divide the histograms into segments.
        z_arr = np.arange(nQuant + 1)
                                        # initialize q - the values that each of the segments intensities will map to.
        q_arr = np.arange(nQuant, dtype=np.float32)
                                        # Set values to z_arr.
        for i in range(0, len(z_arr)):
            z_arr[i] = round((i / nQuant) * len(histOrig))

        img_list = []
        mse_list = []
                                        # Calculate the error for nIter times.
        for k in range(0, nIter):

            for i in range(0, nQuant):
                                        # Calculate the map for each segment.
                                        # Summing the error from z in i'th place to i+1.
                q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1] + 1], weights=histOrig[z_arr[i]:z_arr[i + 1] + 1])

            for j in range(1, nQuant):
                                        # Setting the borders.
                z_arr[j] = (q_arr[j - 1] + q_arr[j]) / 2

            quant_img = img.copy()
            for l in range(1, nQuant + 1):
                quant_img[(quant_img >= z_arr[l - 1]) & (quant_img < z_arr[l])] = q_arr[l - 1]

            img_list.append(quant_img)
                                        # Performing Mean Square Error:
                                        # Checking the distance between the original intensities to the new ones.
            mse = pow(np.power(img - quant_img, 2).sum(), 0.5) / img.size
            mse_list.append(mse)

    # RGB case -
    # ````````````````
    elif len(imOrig.shape) == 3:

        img = transformRGB2YIQ(imOrig) * 255    # de-normalized the pixels intensity & convert to YIQ.

        histOrig , bins = np.histogram(img[:, :, 0].flatten(), 256, [0, 256])

        z_arr = np.arange(nQuant + 1)
        q_arr = np.arange(nQuant)

        for i in z_arr:  # init
            z_arr[i] = round((i / nQuant) * len(histOrig - 1))

        img_list = []
        mse_list = []

        for k in range(0, nIter):

            for i in range(0, nQuant):
                q_arr[i] = np.average(bins[z_arr[i]:z_arr[i + 1]].reshape(1, -1),
                                      weights=histOrig[z_arr[i]:z_arr[i + 1]].reshape(1, -1))

            for j in range(1, nQuant):
                z_arr[j] = round((q_arr[j - 1] + q_arr[j]) / 2)

            quant_img = img.copy()
            for i in range(1, nQuant + 1):
                quant_img[:, :, 0][(quant_img[:, :, 0] > z_arr[i - 1]) & (quant_img[:, :, 0] < z_arr[i])] = q_arr[i - 1]

            quant_img = transformYIQ2RGB(quant_img) / 255
            img_list.append(quant_img)
                                            # Performing Mean Square Error:
                                            # Checking the distance between the original intensities to the new ones.
            mse=pow(np.power(imOrig-quant_img,2).sum(),0.5)/imOrig.size
            mse_list.append(mse)

    else:
        raise ValueError('Unsupported representation. only RGB or Greyscale images allowed')

    return img_list, mse_list
    pass

















































# def hsitogramEqualize_(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
#     # Checking imgOrig channels (grayScale or RGB)
#     g_scale = False
#     if len(imgOrig.shape) == 2:
#         g_scale = True
#     elif len(imgOrig.shape) == 3:
#         rgb_scale = False
#     else:
#         raise ValueError('RGB || Gray Scale image only')
#     img2 = np.copy(imgOrig)
#
#     if g_scale:
#         histogram, bin_edges = np.histogram(img2, bins=256, range=(0, 1))
#         plt.figure()
#         plt.title("grayscale Histogram")
#         plt.xlabel("grayscale value")
#         plt.ylabel("pixels")
#         plt.xlim([0.0, 1.0])
#
#         plt.plot(bin_edges[0:-1], histogram)
#         plt.show()
#
#     else:
#         colors = ("r", "g", "b")
#         channel_ids = (0, 1, 2)
#
#         # create the histogram plot, with three lines, one for
#         # each color
#         plt.xlim([0, 1.0])
#         for channel_id, c in zip(channel_ids, colors):
#             histogram, bin_edges = np.histogram(
#                 img2[:, :, channel_id], bins=256, range=(0, 1))
#
#             plt.plot(bin_edges[0:-1], histogram, color=c)
#
#         plt.xlabel("Color value")
#         plt.ylabel("Pixels")
#         plt.show()
#     pass
