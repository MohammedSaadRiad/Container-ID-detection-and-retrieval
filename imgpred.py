import os
import argparse
from ultralytics import YOLO
import cv2
import keyboard
# import pytesseract
import numpy as np
import easyocr
# import pyautogui

""" 
parser = argparse.ArgumentParser
parser.add_argument('--source')
arg = parser.parse_args()
 """

# conf= r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
reader = easyocr.Reader(['en'])

model_path = os.path.join('.', 'runs', 'detect', 'containernumregion','best.pt')
input_dir=os.path.join(".","images")
model = YOLO(model_path)

for img_name in os.listdir(input_dir):
   if keyboard.is_pressed('esc'):
    break
   img = os.path.join(input_dir, img_name)
   img=cv2.imread(img)
 

   height,width,_=img.shape


   threshold = 0.3



   class_name_dict = ["region"]


   results = model.predict(img)[0]
   textup=0
   for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

       
        if score > threshold:
          
       # Crop the detected object
         center_x = int(x1 + (x2-x1)/2)
         center_y = int(y1 + (y2-y1)/2) 
         w = int(x2 - x1)         
         h = int(y2 - y1)
         x = int(center_x - w / 2)
         y = int(center_y - h / 2)
         img_pred = img.copy()
         imgcrop = img_pred[y:y+h, x:x+w]
         
         
         imgcropresize = cv2.resize(imgcrop, (0,0), fx=6, fy=4 ,interpolation=cv2.INTER_CUBIC)

         # imgdenoised = cv2.fastNlMeansDenoisingColored(imgcropresize, None, 10, 10, 7, 21)

         

         kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
         imgsharpen = cv2.filter2D(imgcropresize, -1, kernel)
         imgenhance = cv2.detailEnhance(imgsharpen, sigma_s=10, sigma_r=0.15)

         
         """ clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

         imgcontrasted = clahe.apply(imggray) """

         imgdenoised2 = cv2.fastNlMeansDenoisingColored(imgenhance, None, 10, 10, 7, 21)


         imggray = cv2.cvtColor(imgdenoised2, cv2.COLOR_BGR2GRAY)

         
         # _,imgcropthresh = cv2.threshold(imggray,100, 255, cv2.THRESH_BINARY_INV)

         # imgprocessed = cv2.adaptiveThreshold(imgcropgray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
         #                                      cv2.THRESH_BINARY,101,20)
                  
      #    whatsfed = imgcropgray
         
         output = reader.readtext(imggray, paragraph=False, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
         containernumber=""
         letters=""
         numbers=""
         for out in output:
                  
                  text_bbox, text, text_score = out
                  # if text_score > 0.4:
                  """ if text.isalpha():
                   letters=text
                  elif text.isdigit():
                   numbers=text
                  else:
                     containernumber+=text
                  containernumber+=letters+number """
                  containernumber+=text
                  # print(text,text_score)
         # text=pytesseract.image_to_string((imggray), config=conf)
         # print(text)     
         if containernumber[6:].isalpha() and containernumber[:6].isdigit():
            containernumber = containernumber[6:] + containernumber[:6]
         
         print(containernumber)
                           
      #  containernumber = containernumber.upper()          
      #  print(containernumber)

         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), int(height*0.002))
               
         cv2.putText(img, class_name_dict[int(class_id)], (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, height*0.0013, (0, 255, 0), 3, cv2.LINE_AA)
         cv2.putText(img, containernumber , (int(width*0.2), int(height*0.9+textup)),cv2.FONT_HERSHEY_SIMPLEX, int(height*0.0013), (0, 0, 255), int(height*0.0013), cv2.LINE_AA) 


         textup+=height*0.07
   cv2.imshow("output",img)
   # cv2.imwrite("output.jpg", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
