import numpy as np
import  cv2
import glob

def comare_2img(imgsrc, imgsrc3, time1, time2, sift):

    showPictures= True
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
  #      print("The image have same size and channels")
        diffrence= cv2.subtract(image,filterdImage)
        b, g, r = cv2.split(diffrence)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("The image are equals !!!!!")

    #    print("The blue difference (in pixels):",cv2.countNonZero(b)) # number of no zero blue pixels
     #   print("The blue difference (in pixels):",cv2.countNonZero(g)) # number of no zero green pixels
     #   print("The blue difference (in pixels):", cv2.countNonZero(r)) # number of no zero red pixels

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
    #    print("fitcher to first image:"+str(len(kp_1)))
     #   print("fitcher to second image:"+str(len(kp_2)))
        index_params= dict(algorithm=0, trees=5)
        search_params= dict()
        flann= cv2.FlannBasedMatcher(index_params, search_params)

        if (len(kp_1))!=0 and (len(kp_2))!=0:
            matches= flann.knnMatch(desc_1, desc_2, k=2) # if find some similar descriptors do there is a match (the fitcher on the first img similar to the fitcher in the second)
    #        print("matches: ",len(matches))

            good_points = []
            for m,n in matches: # choose the good points
                alpa= 0.85
                if m.distance < alpa* n.distance: # m tack from desc_1, n from desc_2
                    # the distance difine how good the matches are-> (smaller distance=better matches)
                    # alpa= 0.1 will find less matches then alpa=1. in alpa=1 we will find more matches but many false one
                    good_points.append(m)
#            print("good_points: ",len(good_points))

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
        #    print("how good it's the match: ", matchPercent , "file placed in:", imgsrc3)
            if showPictures:
                cv2.imshow("result", result)
                cv2.waitKey(time2)
                cv2.destroyAllWindows()
                cv2.waitKey(1)
        else:
            print("fitcher in one of the images is:" + str(len(kp_2)))


        # sift2 = cv2.xfeatures2d.SIFT_create()
        # kp = sift2.detect(image, None)
        # img = cv2.drawKeypoints(image, kp, img)
        # img= cv2.resize(img,(500,350))
        #
        # cv2.imshow("SIFT", img)
        # cv2.imwrite('sift_keypoints.jpg', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return matchPercent

def sanityCheck(sift):
    imgsrc = r'C:\Users\user\PycharmProjects\imgCompare\pistol\rofl0.png'
    comare_2img(imgsrc, imgsrc, 6000,6000,sift)


def comare_many(imgsrc,sift):
    all_images_to_compare= []
    titels = []
    for file in glob.iglob(r"C:/Users/user/PycharmProjects/compare_Images/views/*"):
        titels.append(file)
        all_images_to_compare.append(file)
        # print(file)

    for imageToComp, title in zip(all_images_to_compare, titels):
        matchPercent= comare_2img(imgsrc, imageToComp, 8000, 8000,sift)

def compare(img1SRC,img2SRC):
    # img1SRC = r'C:\Users\user\PycharmProjects\compare_Images\views\img_2.jpg'
    # img2SRC = r'C:\Users\user\PycharmProjects\compare_Images\views\img_2.jpg'

    # sift = cv2.xfeatures2d.SIFT_create()
    # comare_many(imgsrc,sift)
    sift = cv2.xfeatures2d.SIFT_create()

    matchPercent= comare_2img(img1SRC, img2SRC, 6000, 6000,sift)

    print(matchPercent/100)
    return matchPercent/100



if __name__ == '__main__':
    import PIL.Image

    pistol_paths = []
    pistol_im = []
    for file in glob.iglob(r"C:/Users/user/data/pistol/*"):
        pistol_paths.append(file)
        pistol_im.append(PIL.Image.open(file))

    pistol_data = {
        0: ["pistol-> Android 9.0", pistol_paths[0]],
        1: ["pistol -> Android 8.0", pistol_paths[1]],
        2: ["pistol -> Android 7.0", pistol_paths[2]],
        3: ["pistol ->Android 4.4", pistol_paths[3]],
        # 4: ["pistol -> Android 4.3", pistol_paths[4]]

    }

    pistol_ssim = []
    for i in range(len(pistol_paths)):
        new = []
        for j in range(len(pistol_paths)):
            new.append(compare(pistol_paths[i], pistol_paths[j]))
            print(pistol_paths[i] + " beween " + pistol_paths[j])
        pistol_ssim.append(new)

    print(pistol_ssim)


    paths = []
    im = []
    for file in glob.iglob(r"C:/Users/user/data/emojis_ios/*"):
        paths.append(file)

    emojis_data = {
        0: ["Face with Tears of Joy-iOS 13.3", "😂-> iOS 13.3", paths[0]],
        1: ["Face with Tears of Joy-iOS 8.3", "😂-> iOS 8.3", paths[1]],
        2: ["Grinning Face-iOS 13.3", "😀-> iOS 13.3", paths[2]],
        3: ["Grinning Face-iOS 8.3", "😀-> iOS 8.3", paths[3]],
        4: ["Pistol-iOS 13.3", "🔫-> iOS 13.3", paths[4]],
        5: ["Pistol-iOS 8.3", "🔫-> iOS 8.3", paths[5]],
        6: ["Cookie-iOS 13.3", "🍪-> iOS 13.3", paths[6]],
        7: ["Cookie-iOS 8.3", "🍪-> iOS 8.3", paths[7]],
        8: ["Grape-iOS 13.3", "🍇-> iOS 13.3", paths[8]],
        9: ["Grape-iOS 8.3", "🍇-> iOS 8.3", paths[9]]

    }

    imgComp = []
    for i in range(len(paths)):
        new = []
        for j in range(len(paths)):
            new.append(compare(paths[i], paths[j]))
            print(paths[i])
            print(paths[j])
        imgComp.append(new)

print(imgComp)

