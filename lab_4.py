import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

good_img=[]
bad_img=[]

good_img=[cv2.imread("toothbrush\\train\\good\\%03d.png" % (i)) for i in range(60)]
bad_img=[cv2.imread("toothbrush\\train\\defective\\%03d.png" % (i)) for i in range(30)]

IG_good_img=[cv2.cvtColor(good_img[i],cv2.COLOR_BGR2GRAY) for i in range(60)]
IG_bad_img=[cv2.cvtColor(bad_img[i],cv2.COLOR_BGR2GRAY) for i in range(30)]

#Preprocesing img
IG_good_img=[cv2.medianBlur(IG_good_img[i],5) for i in range(60)]
IG_bad_img=[cv2.medianBlur(IG_bad_img[i],5) for i in range(30)]
IG_good_img_train,IG_good_img_test=IG_good_img[:45],IG_good_img[45:]
IG_bad_img_train,IG_bad_img_test=IG_bad_img[:20],IG_bad_img[20:]


median_img_good=np.median(IG_good_img_train,axis=0)
std_img_good=np.std(IG_good_img_train,axis=0)

def detection(img):
    diff=np.abs(img-median_img_good)
    mask=diff>(0.8*std_img_good)
    defect_pixels=np.sum(mask)
    defect_ratio = defect_pixels / (img.shape[0]*img.shape[1])
    if defect_ratio>0.0001:
        return 0
    else:
        return 1
    
y_true=[]
y_pred=[]
for i in range(0,15):
    p_g=detection(IG_good_img_test[i])
    y_true.append(1)
    y_pred.append(p_g)

for i in range(0,10):
    p_b=detection(IG_bad_img_test[i])
    y_true.append(0)
    y_pred.append(p_b)


print(confusion_matrix(y_true, y_pred))
