# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

def mse(imageA,imageB):
    err= np.sum((imageA.astype("float")- imageB.astype("float"))**2)
    err /= float(imageA.shape[0]* imageA.shape[1])
    return err


def compare_colored_mse(imageA, imageB, imgsrc1, imgsrc2):
    b, g, r = cv2.split(imageA)
    b2, g2, r2 = cv2.split(imageB)

    # red_channel1 = imageA[:,:,2]
    m_b = mse(b, b2)
    m_r = mse(r, r2)
    m_g = mse(g, g2)

    mse_brg=((m_b + m_r + m_g) / 3)
    print("mse_brg:", mse_brg, "between:" , imgsrc1, "and", imgsrc2)

    mse_Percent = mse_brg / (65025.0)
    print("mse_Percent:", mse_Percent, "between:" , imgsrc1, "and", imgsrc2)

def compare_image_ssim(imageA, imageB,  imgsrc1, imgsrc2):

    s= ssim(imageA, imageB, multichannel=True)
    print("SSIM:" , s, "between:" , imgsrc1, "and", imgsrc2)

def compare_image_mse(imageA, imageB, imgsrc1, imgsrc2):
    regular_mse = mse(imageA,imageB)
    print("regular_mse:", regular_mse, "between:" , imgsrc1, "and", imgsrc2)

if __name__ == '__main__':
    # imgsrc0 = r'C:\Users\user\PycharmProjects\imgCompare\pistol\pistol0.png'
    # pistol0 = cv2.imread(imgsrc0)
    # pistol0 = cv2.resize(pistol0, (10000, 10000))
    # pistol0 = cv2.cvtColor(pistol0, cv2.COLOR_BGR2GRAY)

    # imgsrc1 = r'C:\Users\user\PycharmProjects\imgCompare\pistol\pistol1.png'
    # pistol1 = cv2.imread(imgsrc1)
    # pistol1 = cv2.resize(pistol1, (10000, 10000))
    # pistol1 = cv2.cvtColor(pistol1, cv2.COLOR_BGR2GRAY)

    # imgsrc2 = r'C:\Users\user\PycharmProjects\imgCompare\pistol\pistol2.png'
    # pistol2 = cv2.imread(imgsrc2)
    # pistol2 = cv2.resize(pistol2, (10000, 10000))
    # pistol2 = cv2.cvtColor(pistol2, cv2.COLOR_BGR2GRAY)

    # imgsrc3 = r'C:\Users\user\PycharmProjects\imgCompare\pistol\pistol3.png'
    # pistol3 = cv2.imread(imgsrc3)
    # pistol3 = cv2.resize(pistol3, (10000, 10000))
    # pistol3 = cv2.cvtColor(pistol3, cv2.COLOR_BGR2GRAY)

    # imgsrc4 = r'C:\Users\user\PycharmProjects\imgCompare\Cookie\cookie0.png'
    # cookie0 = cv2.imread(imgsrc4)
    # cookie0 = cv2.resize(cookie0, (10000, 10000))
    # cookie0 = cv2.cvtColor(cookie0, cv2.COLOR_BGR2GRAY)

    # imgsrc5 = r'C:\Users\user\PycharmProjects\imgCompare\Cookie\cookie1.5.png'
    # cookie1 = cv2.imread(imgsrc5)
    # cookie1 = cv2.resize(cookie1, (10000, 10000))
    # cookie1 = cv2.cvtColor(cookie1, cv2.COLOR_BGR2GRAY)

    imgsrc6 = r'C:\Users\user\PycharmProjects\imgCompare\Cookie\cookie1.png'
    cookie2 = cv2.imread(imgsrc6)
    cookie2 = cv2.resize(cookie2, (10000, 10000))
    cookie2 = cv2.cvtColor(cookie2, cv2.COLOR_BGR2GRAY)

    imgsrc7 = r'C:\Users\user\PycharmProjects\compare_Images\Cookie\cookie0.png'
    image0 = cv2.imread(imgsrc7)
    image0 = cv2.resize(image0, (10000, 10000))
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)

    # imgsrc8 = r'C:\Users\user\PycharmProjects\compare_Images\pistol\pistol4.png'
    # image1 = cv2.imread(imgsrc8)
    # image1 = cv2.resize(image1, (10000, 10000))
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    print("hello")
    compare_image_mse(image0, cookie2, imgsrc7, imgsrc6)
    # compare_colored_mse(image0, cookie2, imgsrc7, imgsrc6)
    #compare_image_ssim(image0, image1, imgsrc7, imgsrc8)
