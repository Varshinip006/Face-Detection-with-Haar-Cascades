# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  
## PROGRAM:
### Name : PRIYA AVRSHINI P
### Register Number : 212224240119

```
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

withglass = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)  
group = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)


if withglass is None:
    raise FileNotFoundError("image1.png not found or path incorrect")
if group is None:
    raise FileNotFoundError("image2.png not found or path incorrect")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

print("Face cascade loaded:", not face_cascade.empty())
print("Eye cascade loaded: ", not eye_cascade.empty())

def show(img, title=None):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    if title: plt.title(title)
    plt.axis('off')
    plt.show()

def detect_face_and_eyes(gray_img,
                         face_scale=1.1, face_neighbors=5, face_min_size=(30,30),
                         eye_scale=1.1, eye_neighbors=3, eye_min_size=(10,10)):
    img_copy = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR) 
    faces = face_cascade.detectMultiScale(gray_img,
                                          scaleFactor=face_scale,
                                          minNeighbors=face_neighbors,
                                          minSize=face_min_size)
    print(f"Detected {len(faces)} face(s).")
    for (x,y,w,h) in faces:
        
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0,255,0), 2)

        
        face_roi_gray  = gray_img[y:y+h, x:x+w]
        face_roi_color = img_copy[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_roi_gray,
                                            scaleFactor=eye_scale,
                                            minNeighbors=eye_neighbors,
                                            minSize=eye_min_size)
        print(f"  -> {len(eyes)} eye(s) in this face ROI.")
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)  # blue for eyes

   
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_rgb


res_withglass = detect_face_and_eyes(withglass,
                                     face_scale=1.1, face_neighbors=5, face_min_size=(40,40),
                                     eye_scale=1.15, eye_neighbors=3, eye_min_size=(15,15))
show(res_withglass, "withglass - detection")


res_group = detect_face_and_eyes(group,
                                 face_scale=1.2, face_neighbors=5, face_min_size=(60,60),
                                 eye_scale=1.1, eye_neighbors=3, eye_min_size=(12,12))
show(res_group, "group - detection")

```
## OUTPUT:
### INPUT IMAGES:

![image](https://github.com/user-attachments/assets/96ea5690-5a57-4863-8f08-44c9474af7c7)

![image](https://github.com/user-attachments/assets/78da36d8-10c0-4668-bf99-92bfb0f7e427)

### FACE DETECTION:
![image](https://github.com/user-attachments/assets/d84ff4f4-87fc-4f31-a989-2198358a640f)

![image](https://github.com/user-attachments/assets/b77389cc-f4f5-4f26-a151-6324551b184a)

### EYE DETECTION:

![image](https://github.com/user-attachments/assets/2cde8f7e-4de8-41c2-98e8-9f567f302352)


## RESULT:
Thus, to write a Python program using OpenCV to perform image manipulations for the given objectives is executed sucessfully.
