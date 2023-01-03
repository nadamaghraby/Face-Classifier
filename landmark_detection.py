import cv2
import dlib
import numpy as np
class Landmarks():
  """A class for facial landmarks detection
    Args: 
        model_path: expects a path to dlib facial landmarks pretrained predictor

    Attributes: 
        detector: initializes and stores the face detector model. This pretrained model is based on HOG and SVM.
        predictor: initializes and stores the landmarks detector model. This pretrained model is based on decision trees ensemble.

    Methods:
        detect_faces: detect faces in original_image
            Returns: 
                rectangles: coordinates of facial boundaries

        detect_landmarks: detect 68 facial landmark for each face
            Args:
                rectangles: coordinates of facial boundaries returned from detect_faces method
            Returns:
                rectangles_landmarks: coordinates of 68 facial landmark for each face
                                      indices are as follows: jaw: [0, 16], right eyebrow: [17, 21], left eyebrow: [22, 26], nose: [27, 35], right eye: [36, 41], left eye:[42, 47], mouth: [48, 67].

        apply_rectangles: draw green rectangles around each face
            Args: 
                input_image: the image to draw facial boundaries on
                rectangles: coordinates of facial boundaries returned from detect_faces method 
            Returns:
                detected_faces_image: modified image after drawing facial boundaries 
        
        apply_landmarks: draw blue dots on each facial landmark
            Args:
                input_image: the image to draw facial landmarks on
                rectangles_coordinates: coordinates of facial landmarks returned from detect_landmarks method
            Returns:
                detected_landmarks_image: modified image after drawing facial landmarks 
        image_mask_concat: creates mask and concatenate it with image
            Args:
                image: the image to concatenate facial landmarks mask with 
            Returns:
                4 channels array of rgb image concatenated with landmarks mask                  
                  """

  def __init__(self,model_path):
    self.detector = dlib.get_frontal_face_detector() #instance of face detection model
    self.predictor = dlib.shape_predictor(model_path)  #instance of landmark detection model, pretrained model download link: https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat
  

  #draws a rectangle without overwriting the original image
  def _draw_rectangle(self,image, coordinates1, coordinates2):
    new_image=image.copy()
    thickness=int(0.007*(image.shape[0]*image.shape[1])**0.5)
    new_image=cv2.rectangle(new_image, coordinates1, coordinates2, (0, 255, 0), thickness)   #last two arguments are color of rectangle and thickness
    return new_image

  #draws a circle without overwriting the original image
  def _draw_circle(self,image, x, y):
    new_image=image.copy()
    thickness=int(0.007*(image.shape[0]*image.shape[1])**0.5)
    new_image=cv2.circle(new_image, (x, y), thickness, (0, 0, 255),-1)   #last three arguments are radius,color,and thickness (-1 means filled circles)
    return new_image

  #detects faces and returns rectangles of faces (regions of interests)
  def detect_faces(self,image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #transform image into gray scale
    rectangles = self.detector(gray_image)   # detect the faces (rectangles)
    return rectangles #rectangles coordinates

  #detects landmarks and returns landmarks of faces
  def detect_landmarks(self,image,rectangles):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #transform image into gray scale
    rectangles_landmarks=[]  #list to append landmarks of each face
    for rectangle in rectangles: #iterate on faces
      landmarks = self.predictor(gray_image, rectangle)   # apply the shape predictor to the face ROI
      landmarks_coordinates=[] #list to append coordinates of each rectangle
      for n in range(landmarks.num_parts):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_coordinates.append((x,y))
      rectangles_landmarks.append(landmarks_coordinates)
    
    modified_landmarks=[[]]
    for idx,landmark in enumerate(rectangles_landmarks[0]):
      if ((idx<=16) or (idx>=27 and idx<=30)):
        modified_landmarks[0].append(landmark)
    return modified_landmarks #returns list of lists (each list contains landmarks coordinates of 1 face)

  #draw rectangles on input image
  def apply_rectangles(self,input_image,rectangles):
    detected_faces_image=input_image
    for rectangle in rectangles:
      # extract the coordinates of the bounding box
      x1 = rectangle.left()
      y1 = rectangle.top()
      x2 = rectangle.right()
      y2 = rectangle.bottom()
      detected_faces_image=self._draw_rectangle(detected_faces_image, (x1, y1), (x2, y2)) #last two arguments are color of rectangle and thickness
    return detected_faces_image
  
  #draw landmarks on input image
  def apply_landmarks(self,input_image,rectangles_coordinates):
    detected_landmarks_image=input_image
    for rectangle_coordinates in rectangles_coordinates:
      for x,y in rectangle_coordinates:
        detected_landmarks_image=self._draw_circle(detected_landmarks_image, x, y) #last three arguments are radius,color,and thickness (-1 means filled circles)
    return detected_landmarks_image #landmarks are now drawn on the image

  #Create a landmarks mask with equivalent image dimensions
  def _create_mask(self,image,thickness=1):
    landmarks=self.detect_landmarks(image,self.detect_faces(image))
    modified_landmarks=[[]]
    if len(landmarks) >0:
      for idx,landmark in enumerate(landmarks[0]):
        if ((idx<=16) or (idx>=27 and idx<=30)):
          modified_landmarks[0].append(landmark)
    modified_landmarks_mask=np.zeros((image.shape[0],image.shape[1]))
    for modified_landmark in modified_landmarks:
      for x,y in modified_landmark:
        modified_landmarks_mask=self._draw_circle(modified_landmarks_mask, x, y,thickness)
    return modified_landmarks_mask

  #concatenate landmarks with original image
  def image_mask_concat(self, image, thickness=1):
    mask=self._create_mask(image,thickness)
    return np.dstack((image,mask)) 



##########################################################
# uncomment this section to test the module individually #
##########################################################
#import matplotlib.pyplot as plt
#image=cv2.imread("dataset_example.jpg")
#model=Landmarks(image)
#rectangles=model.detect_faces()
#landmarks=model.detect_landmarks(rectangles)
#print(rectangles)
#print("landmarks")
#print(landmarks)
#image_with_rectangles=model.apply_rectangles(image,rectangles) 
#image_with_both= model.apply_landmarks(image_with_rectangles,landmarks) #draw landmarks on image with rectangles
#plt.imshow(cv2.cvtColor(image_with_both, cv2.COLOR_BGR2RGB));
#plt.show()