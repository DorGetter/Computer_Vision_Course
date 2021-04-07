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
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np
import cv2.cv2 as cv2

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def gamma(X):
    return


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    img = cv2.imread(img_path, rep - 1)
    cv2.namedWindow('Gamma Correction')
    cv2.createTrackbar('Gamma', 'Gamma Correction', 1, 200, gamma)
    img = np.asarray(img) / 255

    cv2.imshow('Gamma correction', img)
    k = cv2.waitKey(1)

    new_gamma_img = img

    while 1:
        cv2.imshow('Gamma correction', new_gamma_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        g = cv2.getTrackbarPos('Gamma', 'Gamma correction')
        print(g / 100)
        new_gamma_img = np.power(img, g / 100)

    cv2.destroyAllWindows()
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
