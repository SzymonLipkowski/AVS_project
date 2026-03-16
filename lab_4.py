import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

good_img=[]
bad_img=[]
mask_img=[]
good_img=[cv2.imread("toothbrush\\train\\good\\%03d.png" % (i)) for i in range(60)]
bad_img=[cv2.imread("toothbrush\\train\\defective\\%03d.png" % (i)) for i in range(30)]
# GT_bad_img=[cv2.imread("toothbrush\ground_truth\defective\\%03d.png" % (i)) for i in range(30)]
# GT_bad_img_test=GT_bad_img[20:]
IG_org_good,IG_org_bad=good_img[45:],bad_img[20:]
IG_good_img=[cv2.cvtColor(good_img[i],cv2.COLOR_BGR2GRAY) for i in range(60)]
IG_bad_img=[cv2.cvtColor(bad_img[i],cv2.COLOR_BGR2GRAY) for i in range(30)]

IG_good_img = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for img in IG_good_img]
IG_good_img = [((img > 40) * 255).astype('uint8') for img in IG_good_img]

IG_bad_img = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for img in IG_bad_img]
IG_bad_img = [(((img > 20)&(img<90)) * 255).astype('uint8') for img in IG_bad_img]

IG_good_img=[cv2.medianBlur(IG_good_img[i],5) for i in range(60)]
IG_bad_img=[cv2.medianBlur(IG_bad_img[i],5) for i in range(30)]

for i in range(20,30):
    cv2.imshow("Bad Image",IG_bad_img[i])
    cv2.waitKey(0)

# IG_good_img_train,IG_good_img_test=IG_good_img[:45],IG_good_img[45:]
# IG_bad_img_train,IG_bad_img_test=IG_bad_img[:20],IG_bad_img[20:]


# median_img_good=np.median(IG_good_img_train,axis=0)
# std_img_good=np.std(IG_good_img_train,axis=0)

# def detection(img):
#     diff=np.abs(img-median_img_good)
#     mask=diff>(2.5*std_img_good)
#     defect_pixels=np.sum(mask)
#     defect_ratio = defect_pixels / (img.shape[0]*img.shape[1])
#     if defect_ratio>0.08:
#         return 0, diff
#     else:
#         return 1, diff
    
# y_true=[]
# y_pred=[]
# for i in range(0,15):
#     p_g=detection(IG_good_img_test[i])
#     y_true.append(1)
#     y_pred.append(p_g)

# for i in range(0,10):
#     p_b,mask=detection(IG_bad_img_test[i])
#     y_true.append(0)
#     y_pred.append(p_b)
#     cv2.imshow("Mask",mask)
#     cv2.waitKey(0)

# print(confusion_matrix(y_true, y_pred))
# print(classification_report(y_true, y_pred))
