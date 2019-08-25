
import numpy
import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
img=cv2.imread('test3.png')
img=imutils.resize(img, width=500)
cv2.imshow("ACTUAL IMAGE",img)
cv2.waitKey(0)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("STEP 1",gray)
cv2.waitKey(0)

gray=cv2.bilateralFilter(gray,11,17,17)
cv2.imshow("STEP 2",gray)
cv2.waitKey(0)

edged=cv2.Canny(gray,170,200)
cv2.imshow("STEP 3",edged)
cv2.waitKey(0)

noofcounts, new=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

img1=img.copy()
cv2.drawContours(img1,noofcounts,-1,(0,255,0),3)
cv2.imshow("STEP 4",img1)
cv2.waitKey(0)

noofcounts=sorted(noofcounts, key=cv2.contourArea,reverse=True)[:30]
NumberPlateCnt=None

img2=img.copy()
cv2.drawContours(img2,noofcounts,-1,(0,255,0),3)
cv2.imshow("STEP 5",img2)
cv2.waitKey(0)

count=0
idx=7
for c in noofcounts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)

    if len(approx)==4:
        NumberPlateCnt=approx

        x,y,w,h=cv2.boundingRect(c)
        new_img=img[y:y+h,x:x+w]
        cv2.imwrite('Cropped imgs-Text/'+str(idx)+'.png',new_img)
        idx+=1

        break


cv2.drawContours(img,[NumberPlateCnt],-1,(0,255,0),3)
cv2.imshow("Final img With Number Plate Detected",img)
cv2.waitKey(0)

Cropped_img_loc='Cropped imgs-Text/7.png'
cv2.imshow("Cropped img",cv2.imread(Cropped_img_loc))
text=pytesseract.img_to_string(Cropped_img_loc,lang='eng')
print("Number is:",text)