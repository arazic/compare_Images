# from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA,imageB):
    err= np.sum((imageA.astype("float")- imageB.astype("float"))**2)
    err /= float(imageA.shape[0]* imageA.shape[1])
    return err

def compare_image(imageA, imageB, title):
    m= mse(imageA, imageB)
    s= ssim(imageA, imageB)
    fig= plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m,s))

    ax= fig.add_subplot(1,2,1)
    plt.imshow(imageA, cmap= plt.cm.gray)
    plt.axis("off")

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")

    plt.show()



if __name__ == '__main__':
    # imgsrc0 = r'C:\Users\user\PycharmProjects\imgCompare\pistol0.png'
    # pistol0 = cv2.imread(imgsrc0)
    # pistol0 = cv2.resize(pistol0, (10000, 10000))
    # pistol0 = cv2.cvtColor(pistol0, cv2.COLOR_BGR2GRAY)
    #
    # imgsrc1 = r'C:\Users\user\PycharmProjects\imgCompare\pistol\pistol1.png'
    # pistol1 = cv2.imread(imgsrc1)
    # pistol1 = cv2.resize(pistol1, (10000, 10000))
    # pistol1 = cv2.cvtColor(pistol1, cv2.COLOR_BGR2GRAY)
    #
    # imgsrc2 = r'C:\Users\user\PycharmProjects\imgCompare\pistol\pistol2.png'
    # pistol2 = cv2.imread(imgsrc2)
    # pistol2 = cv2.resize(pistol2, (10000, 10000))
    # pistol2 = cv2.cvtColor(pistol2, cv2.COLOR_BGR2GRAY)
    #
    # imgsrc3 = r'C:\Users\user\PycharmProjects\imgCompare\pistol\pistol3.png'
    # pistol3 = cv2.imread(imgsrc3)
    # pistol3 = cv2.resize(pistol3, (10000, 10000))
    # pistol3 = cv2.cvtColor(pistol3, cv2.COLOR_BGR2GRAY)

    imgsrc4 = r'C:\Users\user\PycharmProjects\imgCompare\Cookie\cookie0.png'
    cookie0 = cv2.imread(imgsrc4)
    cookie0 = cv2.resize(cookie0, (10000, 10000))
    cookie0 = cv2.cvtColor(cookie0, cv2.COLOR_BGR2GRAY)

    imgsrc5 = r'C:\Users\user\PycharmProjects\imgCompare\Cookie\cookie1.5.png'
    cookie1 = cv2.imread(imgsrc5)
    cookie1 = cv2.resize(cookie1, (10000, 10000))
    cookie1 = cv2.cvtColor(cookie1, cv2.COLOR_BGR2GRAY)

    imgsrc6 = r'C:\Users\user\PycharmProjects\imgCompare\Cookie\cookie1.png'
    cookie2 = cv2.imread(imgsrc6)
    cookie2 = cv2.resize(cookie2, (10000, 10000))
    cookie2 = cv2.cvtColor(cookie2, cv2.COLOR_BGR2GRAY)


    compare_image(cookie0, cookie2, 'COMP1')