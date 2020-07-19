#!/usr/bin/env python
# coding: utf-8

# In[6]:


print ('hello')


# In[7]:


import os


# In[8]:


import pandas as pd


# In[9]:


import numpy as np


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


from matplotlib import font_manager as fm, rcParams


# In[12]:


import matplotlib


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


import seaborn as sns


# In[15]:


sns.set(rc={'figure.figsize':(30,9)})


# In[16]:


plt.style.use('seaborn')


# In[17]:


sns.set_palette(sns.color_palette("hls", 8));


# In[18]:


sns.set()


# In[19]:


plt.rcParams['figure.figsize']=20,9
import datetime
import calendar
import itertools
from cycler import cycler
import time
import pickle
plt.rc('lines', linewidth=4)
cc = (cycler(color=list('bgrmyk')) *
cycler(linestyle=['-']))
plt.rc('axes', prop_cycle=cc)


# In[20]:


emoji = '😊'


# In[21]:


print(emoji)


# In[22]:


from PIL import Image, ImageStat


# In[23]:


from PIL import ImageChops


# In[152]:


im1SRC=r'C:\Users\user\data\emojis_ios\1.png'


# In[153]:


im1 = Image.open(im1SRC)


# In[154]:


print(im1)


# In[155]:


im2SRC=r'C:\Users\user\data\emojis_ios\2.png'


# In[156]:


im2 = Image.open(im2SRC)


# # measure preparation
# 

# ## Pil Comparison

# #####  Returns the absolute value of the pixel-by-pixel difference between the two images.
#                    .. code-block:: python
#                          out = abs(image1 - image2)
#                      :rtype: :py:class:`~PIL.Image.Image`
# 

# In[158]:


diff = ImageChops.difference(im1, im2)


# In[159]:


print(diff)


# In[160]:


def pil_comp(im1,im2):
    diff = ImageChops.difference(im1, im2)
    if diff.getbbox():
        print("images are different")
        print(_compute_manhattan_distance(diff)/100)
        return _compute_manhattan_distance(diff)/100
    else:
        print("images are the same")
        return 0


# In[161]:


pil_comp(im1,im2)


# In[162]:


pil_comp(im1,im1)


# ## Hash Comparison

# ### hash diffrence 
# 
# #####  dhash, an algorithm which compares neighbor pixels if they get darker or lighter, seems so be very accurate   and also fast. 
# 
# ##### the resulting hash won't change if the image is scaled or the aspect ratio changes. Increasing or decreasing the brightness or contrast, or even altering the colors won't dramatically change the hash value. Even complex adjustments like gamma corrections and color profiles won't impact the result. And best of all: this is FAST! Seriously -- the slowest part of the algorithm is the size reduction step.
# 
# ##### The hash values represent the relative change in brightness intensity. To compare two hashes, just count the number of bits that are different. A value of 0 indicates the same hash and likely a similar picture. A value greater than 10 is likely a different image, and a value between 1 and 10 is potentially a variation.

# In[35]:


import os
from PIL import Image
import pylab


# In[36]:



def diff(h1, h2):
    return sum([bin(int(a, 16) ^ int(b, 16)).count('1') for a, b in zip(h1, h2)])

def dhash(image, hash_size = 8):
    # scaling and grayscaling
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.ANTIALIAS)
    pixels = list(image.getdata())

    # calculate differences
    diff_map = []
    for row in range(hash_size):
        for col in range(hash_size):
            diff_map.append(image.getpixel((col, row)) > image.getpixel((col + 1, row)))
    # build hex string
    return hex(sum(2**i*b for i, b in enumerate(reversed(diff_map))))[2:-1]


# In[37]:


def hash_comp(im1, im2):
    h1 = dhash(im1)
    h2 = dhash(im2)
    dataPercent=((((diff(h1, h2))/64)*100))
#     print(dataPercent/100)
    return dataPercent/100


# In[38]:


hash_comp(im1, im1)


# In[166]:



# def diff(h1, h2):
#     return sum([bin(int(a, 16) ^ int(b, 16)).count('1') for a, b in zip(h1, h2)])

# def dhash(image, hash_size = 8):
#     # scaling and grayscaling
#     image = image.convert('L').resize((hash_size + 1, hash_size), Image.ANTIALIAS)
#     pixels = list(image.getdata())

#     # calculate differences
#     diff_map = []
#     for row in range(hash_size):
#         for col in range(hash_size):
#             diff_map.append(image.getpixel((col, row)) > image.getpixel((col + 1, row)))
#     # build hex string
#     return hex(sum(2**i*b for i, b in enumerate(reversed(diff_map))))[2:-1]


# def hash_comp(im1, im2):
#     h1 = dhash(im1)
#     h2 = dhash(im2)
#     dataPercent=((((diff(h1, h2))/64)*100))
# #     print(dataPercent/100)
#     return dataPercent/100

regular_imageSRC = r'C:\Users\user\data\emojis_ios\0.png'
upside_down_imageSRC=r'C:\Users\user\data\emojis_ios\0_upside_down.png'
regular_image= Image.open(regular_imageSRC)
upside_down_image=  Image.open(upside_down_imageSRC)

hash_comp(regular_image,upside_down_image)


# ## SSIM Comparison

# #### SSIM similarity  
# #### SSIM is a perception-based model that considers image degradation as perceived change in structural information, while also incorporating important perceptual phenomena, including both luminance masking and contrast masking terms. Structural information is the idea that the pixels have strong inter-dependencies especially when they are spatially close. These dependencies carry important information about the structure of the objects in the visual scene. 

# In[39]:


from skimage.metrics import structural_similarity as ssim


# In[40]:


import os
os.sys.path


# In[41]:


get_ipython().system('pip install opencv-python')


# In[42]:


import cv2


# In[43]:


def compare_image_ssim(imageA, imageB):

    s= ssim(imageA, imageB, multichannel=True)
    print("SSIM:" , s)


# In[44]:


def preprocessing_ssim_mse(imSRC):
    img = cv2.imread(imSRC)
    img = cv2.resize(img, (10000, 10000))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# In[45]:


def ssim_copm(im1SRC, im2SRC):
    im1= preprocessing_ssim_mse(im1SRC)
    im2= preprocessing_ssim_mse(im2SRC)
    compare_image_ssim(im1,im2)


# In[46]:


ssim_copm(im1SRC,im1SRC)


# ## MSE Comparison

# In[47]:


def mse(imageA,imageB):
    err= np.sum((imageA.astype("float")- imageB.astype("float"))**2)
    err /= float(imageA.shape[0]* imageA.shape[1])
    print(err)
    return err


# In[48]:


def compare_image_ssim(imageA, imageB,  imgsrc1, imgsrc2):

    s= ssim(imageA, imageB, multichannel=True)
    print("SSIM:" , s, "between:" , imgsrc1, "and", imgsrc2)


# In[49]:


def compare_image_mse(imageA, imageB, imgsrc1, imgsrc2):
    regular_mse = mse(imageA,imageB)
    print("regular_mse:", regular_mse)
    


# In[50]:


# imgsrc6 = r'C:\Users\user\PycharmProjects\imgCompare\Cookie\cookie1.png'
image0 = cv2.imread(im1SRC)
image0 = cv2.resize(image0, (10000, 10000))
image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)

# imgsrc7 = r'C:\Users\user\PycharmProjects\compare_Images\Cookie\cookie0.png'
image1 = cv2.imread(im2SRC)
image1 = cv2.resize(image1, (10000, 10000))
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)


# In[51]:


compare_image_mse(image0, image1, im1SRC, im2SRC)


# # Data
# 

# In[170]:


import pandas as pd
import glob


# In[171]:


paths = []
im = []
for file in glob.iglob(r"C:/Users/user/data/emojis_ios/*"):
    paths.append(file)
    im.append(Image.open(file))


# In[172]:


emojis_data={
    0: ["Face with Tears of Joy-iOS 13.3", "😂-> iOS 13.3", paths[0]],
    1: ["Face with Tears of Joy-iOS 8.3", "😂-> iOS 8.3" ,paths[1]],
    2: ["Grinning Face-iOS 13.3", "😀-> iOS 13.3", paths[2]],
    3: ["Grinning Face-iOS 8.3", "😀-> iOS 8.3",  paths[3]],
    4: ["Pistol-iOS 13.3", "🔫-> iOS 13.3", paths[4]],
    5: ["Pistol-iOS 8.3", "🔫-> iOS 8.3",  paths[5]],
    6: ["Cookie-iOS 13.3", "🍪-> iOS 13.3", paths[6]],
    7: ["Cookie-iOS 8.3", "🍪-> iOS 8.3",  paths[7]],
    8: ["Grape-iOS 13.3", "🍇-> iOS 13.3", paths[8]],
    9: ["Grape-iOS 8.3", "🍇-> iOS 8.3",  paths[9]]
    
}


# In[173]:


emojis_data


# In[174]:


emojis_names=[emojis_data[0][1],emojis_data[1][1],emojis_data[2][1],emojis_data[3][1],emojis_data[4][1],emojis_data[5][1],emojis_data[6][1],emojis_data[7][1],emojis_data[8][1],emojis_data[9][1]]
emojis_names


# # Method - Comparing the emojis
# 

# ## pil- Calculates the difference
# 

# In[180]:


regular_imageSRC = r'C:\Users\user\data\emojis_ios\0.png'
upside_down_imageSRC=r'C:\Users\user\data\0_upside_down.png'
regular_image= Image.open(regular_imageSRC)
upside_down_image=  Image.open(upside_down_imageSRC)

pil_comp(regular_image,upside_down_image)


# In[181]:


pil= []
for i in range(len(paths)):
    new=[]
    for j in range(len(paths)):
        new.append(pil_comp(im[i],im[j]))
        print(paths[i])
        print(paths[j])
    pil.append(new)


# In[182]:


emojis_names


# In[183]:


emojis_names=[emojis_data[0][1],emojis_data[1][1],emojis_data[2][1],emojis_data[3][1],emojis_data[4][1],emojis_data[5][1],emojis_data[6][1],emojis_data[7][1],emojis_data[8][1],emojis_data[9][1]]


df_pil= pd.DataFrame(pil, columns =emojis_names)
df_pil.insert(loc=0, column='emoji-> version', value=emojis_names)

df_pil


# ## hash- Calculates the difference
# 

# In[184]:


# def diff(h1, h2):
#     return sum([bin(int(a, 16) ^ int(b, 16)).count('1') for a, b in zip(h1, h2)])

# def dhash(image, hash_size = 8):
#     # scaling and grayscaling
#     image = image.convert('L').resize((hash_size + 1, hash_size), Image.ANTIALIAS)
#     pixels = list(image.getdata())

#     # calculate differences
#     diff_map = []
#     for row in range(hash_size):
#         for col in range(hash_size):
#             diff_map.append(image.getpixel((col, row)) > image.getpixel((col + 1, row)))
#     # build hex string
#     return hex(sum(2**i*b for i, b in enumerate(reversed(diff_map))))[2:-1]

# def hash_comp(im1, im2):
#     h1 = dhash(im1)
#     h2 = dhash(im2)
#     dataPercent=((((diff(h1, h2))/64)*100))
#     return dataPercent/100


regular_imageSRC = r'C:\Users\user\data\emojis_ios\0.png'
upside_down_imageSRC=r'C:\Users\user\data\0_upside_down.png'
regular_image= Image.open(regular_imageSRC)
upside_down_image=  Image.open(upside_down_imageSRC)
hash_comp(regular_image,upside_down_image)


# In[193]:


hash_= []
for i in range(len(paths)):
    new=[]
    for j in range(len(paths)):
        im1 = Image.open(paths[i])
        im2 = Image.open(paths[j])
        new.append(hash_comp(im1,im2))
        print(paths[i])
        print(paths[j])
        print(hash_comp(im1,im2))
    hash_.append(new)
    
hash_


# In[191]:



emojis_names=[emojis_data[0][1],emojis_data[1][1],emojis_data[2][1],emojis_data[3][1],emojis_data[4][1],emojis_data[5][1],emojis_data[6][1],emojis_data[7][1],emojis_data[8][1],emojis_data[9][1]]


df_hash= pd.DataFrame(hash_, columns =emojis_names)
df_hash.insert(loc=0, column='emoji-> version', value=emojis_names)

df_hash


# ## SSIM-  Calculates the similarity

# In[187]:


import cv2

def compare_image_ssim(imageA, imageB):
    s= ssim(imageA, imageB, multichannel=True)
    print("SSIM:" , s)
    
def preprocessing_ssim_mse(imSRC):
    img = cv2.imread(imSRC)
    img = cv2.resize(img, (10000, 10000))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ssim_copm(im1SRC, im2SRC):
    im1= preprocessing_ssim_mse(im1SRC)
    im2= preprocessing_ssim_mse(im2SRC)
    compare_image_ssim(im1,im2)    
    


# In[188]:


get_ipython().system('pip install imutils')
from skimage.metrics import structural_similarity
import argparse
import imutils
import cv2


# In[140]:


from skimage.measure import compare_ssim

def ssim_compare(im1SRC, im2SRC):
    #  Load the two input images
    imageA = cv2.imread(im1SRC)
    imageB = cv2.imread( im2SRC)

    #  Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    #  Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    #  You can print only the score if you want
    print("SSIM: {}".format(score))
    return score


# In[189]:



ssim_= []
for i in range(len(paths)):
    new=[]
    for j in range(len(paths)):
        new.append(ssim_compare(paths[i],paths[j]))
    ssim_.append(new)
    
ssim_


# In[190]:


df_ssim= pd.DataFrame(ssim_, columns =emojis_names)
df_ssim.insert(loc=0, column='emoji-> version', value=emojis_names)

df_ssim 
        


# # Feature detection and Feature matching-  Calculates the similarity

# #### The approach we’re going to use to find similarities if the images are not equal is Feature detection and Feature matching.        We find the features of both images.
# 
# 

# In[144]:


import  cv2
import glob

def comare_2img(imgsrc, imgsrc3, time1, time2, sift):

    showPictures= False
    img = cv2.imread(imgsrc)
    image= cv2.resize(img,(500,350))

    img3 = cv2.imread(imgsrc3)
    filterdImage= cv2.resize(img3,(500,350))

    if showPictures:
        cv2.imshow('Original',image)
        cv2.imshow('Duplicate',filterdImage)
      #  cv2.waitKey(0)
      #  cv2.destroyAllWindows()

# check if two are equals

    res= image.shape
    res2= filterdImage.shape

    # print(res) # (350, 500, 3)- 3 because 3 colors
    # print(res2) # (350, 500, 3)- 3 because 3 colors

###same sizes and chanels
    if res == res2:
        print("The image have same size and channels")
        diffrence= cv2.subtract(image,filterdImage)
        b, g, r = cv2.split(diffrence)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The image are equals !!!!!")

        print("The blue difference (in pixels):",cv2.countNonZero(b)) # number of no zero blue pixels
        print("The blue difference (in pixels):",cv2.countNonZero(g)) # number of no zero green pixels
        print("The blue difference (in pixels):", cv2.countNonZero(r)) # number of no zero red pixels

        if showPictures:
            cv2.imshow("b", b)
            cv2.imshow("g", g)
            cv2.imshow("r", r)
            cv2.waitKey(time1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        #check the similarity between the two images
        kp_1, desc_1 =  sift.detectAndCompute(image,None) # kp=key points, each decs (descripture) discribe key point
        kp_2, desc_2 =  sift.detectAndCompute(filterdImage,None)
        print("fitcher to first image:"+str(len(kp_1)))
        print("fitcher to second image:"+str(len(kp_2)))
        index_params= dict(algorithm=0, trees=5)
        search_params= dict()
        flann= cv2.FlannBasedMatcher(index_params, search_params)

        if (len(kp_1))!=0 and (len(kp_2))!=0:
            matches= flann.knnMatch(desc_1, desc_2, k=2) # if find some similar descriptors do there is a match (the fitcher on the first img similar to the fitcher in the second)
            print("matches: ",len(matches))

            good_points = []
            for m,n in matches: # choose the good points
                alpa= 0.6
                if m.distance < alpa* n.distance: # m tack from desc_1, n from desc_2
                    # the distance difine how good the matches are-> (smaller distance=better matches)
                    # alpa= 0.1 will find less matches then alpa=1. in alpa=1 we will find more matches but many false one
                    good_points.append(m)
            print("good_points: ",len(good_points))

            if good_points == None:
                result= cv2.drawMatchesKnn(image, kp_1, filterdImage, kp_2, matches, None)
                print("good_point none")
            else:
                result= cv2.drawMatches(image, kp_1, filterdImage, kp_2, good_points, None)

        if len(kp_1) <= len(kp_2):
            number_key_points= len(kp_1)
        else:
            number_key_points= len(kp_2)

        matchPercent=None
        if number_key_points != 0:
            matchPercent= (len(good_points)/number_key_points)*100
            print("how good it's the match: ", matchPercent , "file placed in:", imgsrc3)
#             cv2.imshow("result", result)
            cv2.waitKey(time2)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        else:
            print("fitcher in one of the images is:" + str(len(kp_2)))

        return matchPercent

def points_comp(img1SRC, img2SRC):
    sift = cv2.xfeatures2d.SIFT_create()

    matchPercent= comare_2img(img1SRC, img2SRC, 6000, 6000,sift)

    return matchPercent


# In[194]:


FF= [[1.0, 0.10377358490566038, 0.09803921568627452, 0.07142857142857142, 0.0, 0.0, 0.02941176470588235, 0.03431372549019608, 0.004901960784313725, 0.13043478260869565], [0.0849056603773585, 1.0, 0.0, 0.8142857142857143, 0.0, 0.0, 0.0, 0.009433962264150943, 0.009433962264150943, 0.09782608695652174], [0.09313725490196079, 0.03773584905660377, 1.0, 0.15714285714285714, 0.00819672131147541, 0.0, 0.01556420233463035, 0.003816793893129771, 0.007633587786259542, 0.2826086956521739], [0.02857142857142857, 0.7857142857142857, 0.08571428571428572, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11428571428571428], [0.004901960784313725, 0.0, 0.004098360655737705, 0.014285714285714285, 1.0, 0.0, 0.004098360655737705, 0.004098360655737705, 0.036885245901639344, 0.010869565217391304], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0588235294117647, 0.028301886792452834, 0.04669260700389105, 0.014285714285714285, 0.00819672131147541, 0.0, 0.9883268482490273, 0.03501945525291829, 0.023346303501945526, 0.06521739130434782], [0.03431372549019608, 0.009433962264150943, 0.011450381679389311, 0.0, 0.0, 0.038461538461538464, 0.011673151750972763, 1.0, 0.005037783375314861, 0.03260869565217391], [0.00980392156862745, 0.009433962264150943, 0.003816793893129771, 0.0, 0.02459016393442623, 0.0, 0.011673151750972763, 0.0025188916876574307, 1.0, 0.010869565217391304], [0.010869565217391304, 0.010869565217391304, 0.0, 0.0, 0.010869565217391304, 0.0, 0.0, 0.03260869565217391, 0.010869565217391304, 1.0]]
#   alpa= 0.6


# In[195]:


df_FF= pd.DataFrame(FF, columns =emojis_names)
df_FF.insert(loc=0, column='emoji-> version', value=emojis_names)

df_FF 
#   alpa= 0.6


# In[197]:


FF_bigerAlfa=[[1.0, 0.1981132075471698, 0.2352941176470588, 0.2714285714285714, 0.04411764705882353, 0.038461538461538464, 0.06862745098039216, 0.0784313725490196, 0.03431372549019608, 0.30434782608695654], [0.18867924528301888, 1.0, 0.05660377358490567, 0.842857142857143, 0.009433962264150943, 0.0, 0.018867924528301886, 0.04716981132075472, 0.028301886792452834, 0.17391304347826086], [0.2549019607843137, 0.169811320754717, 1.0, 0.4142857142857143, 0.06967213114754098, 0.07692307692307693, 0.054474708171206226, 0.04961832061068702, 0.061068702290076333, 0.45652173913043476], [0.12857142857142856, 0.8, 0.2, 1.0, 0.02857142857142857, 0.0, 0.02857142857142857, 0.04285714285714286, 0.02857142857142857, 0.22857142857142856], [0.06862745098039216, 0.04716981132075472, 0.02868852459016393, 0.04285714285714286, 1.0, 0.01282051282051282, 0.04918032786885246, 0.036885245901639344, 0.0942622950819672, 0.21739130434782608], [0.01282051282051282, 0.0, 0.01282051282051282, 0.0, 0.0, 1.0, 0.01282051282051282, 0.01282051282051282, 0.0, 0.0], [0.22058823529411764, 0.10377358490566038, 0.1245136186770428, 0.12857142857142856, 0.13114754098360656, 0.05128205128205128, 0.9883268482490273, 0.11673151750972761, 0.11673151750972761, 0.25], [0.10294117647058823, 0.04716981132075472, 0.04961832061068702, 0.11428571428571428, 0.04918032786885246, 0.11538461538461538, 0.038910505836575876, 1.0, 0.04030226700251889, 0.20652173913043476], [0.02450980392156863, 0.028301886792452834, 0.022900763358778622, 0.02857142857142857, 0.0860655737704918, 0.08974358974358974, 0.05836575875486381, 0.017632241813602016, 1.0, 0.07608695652173914], [0.06521739130434782, 0.03260869565217391, 0.03260869565217391, 0.04285714285714286, 0.03260869565217391, 0.02564102564102564, 0.03260869565217391, 0.06521739130434782, 0.010869565217391304, 1.0]]
#   alpa= 0.75


# In[198]:


df_FF_bigerAlfa= pd.DataFrame(FF_bigerAlfa, columns =emojis_names)
df_FF_bigerAlfa.insert(loc=0, column='emoji-> version', value=emojis_names)

df_FF_bigerAlfa 


# In[199]:


FF_bigestAlfa=[[1.0, 0.4245283018867924, 0.4362745098039216, 0.6142857142857143, 0.19607843137254904, 0.3717948717948718, 0.19607843137254904, 0.18627450980392157, 0.18137254901960784, 0.6956521739130435], [0.2830188679245283, 1.0, 0.16037735849056603, 0.9142857142857143, 0.169811320754717, 0.08974358974358974, 0.0660377358490566, 0.1509433962264151, 0.16037735849056603, 0.30434782608695654], [0.4852941176470588, 0.5849056603773585, 1.0, 0.8857142857142857, 0.22950819672131145, 0.5769230769230769, 0.16731517509727623, 0.15648854961832062, 0.24427480916030533, 0.9782608695652173], [0.2857142857142857, 0.8, 0.35714285714285715, 1.0, 0.2, 0.1, 0.11428571428571428, 0.12857142857142856, 0.22857142857142856, 0.34285714285714286], [0.18627450980392157, 0.24528301886792453, 0.14754098360655737, 0.37142857142857144, 1.0, 0.19230769230769235, 0.13934426229508196, 0.13524590163934427, 0.20901639344262296, 0.6195652173913043], [0.05128205128205128, 0.01282051282051282, 0.0641025641025641, 0.08571428571428572, 0.05128205128205128, 1.0, 0.07692307692307693, 0.10256410256410256, 0.05128205128205128, 0.05128205128205128], [0.39705882352941174, 0.3679245283018868, 0.2607003891050584, 0.47142857142857136, 0.29918032786885246, 0.41025641025641024, 0.9922178988326849, 0.25680933852140075, 0.311284046692607, 0.7065217391304348], [0.2647058823529412, 0.169811320754717, 0.21755725190839695, 0.3285714285714285, 0.22540983606557374, 0.5256410256410257, 0.16731517509727623, 1.0, 0.12090680100755667, 0.6304347826086957], [0.13725490196078433, 0.18867924528301888, 0.10687022900763359, 0.3285714285714285, 0.21721311475409835, 0.5512820512820513, 0.15953307392996108, 0.08816120906801007, 1.0, 0.3695652173913043], [0.15217391304347827, 0.09782608695652174, 0.10869565217391304, 0.12857142857142856, 0.05434782608695652, 0.1282051282051282, 0.09782608695652174, 0.15217391304347827, 0.10869565217391304, 1.0]]

#   alpa= 0.85

df_FF_bigestAlfa= pd.DataFrame(FF_bigestAlfa, columns =emojis_names)
df_FF_bigestAlfa.insert(loc=0, column='emoji-> version', value=emojis_names)

df_FF_bigestAlfa 


# In[ ]:




