import cv2
import numpy as np
import tensorflow as tf
import os

#from keras.models import load_model 

# load model
model_name = 'model_srp.h5'
path = os.path.join(os.getcwd(),model_name)
model = tf.keras.models.load_model(path)
_, width, height, c=model.layers[0].input_shape

# Labeling
gesture_names={0:'paper',
               1:'scissor',
               2:'rock'}

def capture_display():
    # retrieve: True if read is successful
	# frame: image frame
    ret, frame = cap.read()
	
	# Resize
    frame = cv2.resize(frame,(224,224))
	
	# Copy original image to prevent moodify original image
    #frame_new = frame.copy()
    frame_new = frame
	
    ## flip image
    # 1: horizontal
    # 0: vertical
    # -1: horizontal & vertical

    flipCode = 1
    frame_new = cv2.flip(frame_new, flipCode)

	# Add rectangle
	# cv2.rectangle(image, top_left_corner, bottom_right_corner, color, thickness )    
    frame_new = cv2.rectangle(frame_new, (100, 100), (220, 220), (255, 0, 0), 2) 
		
	# Capture hand
    #img_hand = frame_new.copy()[100:220,100:220]
    img_hand = frame_new[100:220,100:220]
    # Convert to grayscale
    img_hand = cv2.cvtColor(img_hand, cv2.COLOR_BGR2GRAY)
    
    return ret, frame_new, img_hand

# Image Preprocessing and prediction
def predict_image(img_hand, height, width):
    img_scale=cv2.resize(img_hand,(height,width))
    img_arr = np.array(img_scale, dtype='float32')
    img_arr = img_arr.reshape((1,width,height, 1))   #Reshape array dimension to 4 for tensorflow model
    img_arr /= 255
    pred = model.predict(img_arr)
    result = gesture_names[np.argmax(pred)]
    score = float("%0.2f" % (max(pred[0]) * 100))
    return img_scale,result, score


# VideoCapture
# 0: default camera / USB webcam
# 1: USB webcam 2
# 2: USB webcam 3
# -1: New pluged USB webcam
cap = cv2.VideoCapture(0)


# Camera Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

# Create a window called "camera"
cv2.namedWindow('camera',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# Resize Window
cv2.resizeWindow('camera',500,300)

# image count
img_count = 0
save_img_count = 0
# Check if camera is opened
print("Is camera opened? {}".format(cap.isOpened()))

# Help information
helpInfo = '''
Press 'q'： Quit
Press 'c'： Capture Screen
Press 'm':  Multiple images
Press 'r':  Recognize images
'''
print(helpInfo)


while(True):
    # show image
    ret, frame_new, img_hand=capture_display()
    
    if not ret:
        print("Unable to load camera.")
        break    
		
    # Wait 1ms when event happened
    key = cv2.waitKey(1)
    
	# Show the result and accuracy of prediction.
    #_, result,score=predict_image(img_hand, height, width)
    #cv2.putText(frame_new, f"Prediction: {result} ({score}%)", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),1)
    cv2.imshow('camera',frame_new)
    
    # Prevent memory overflow
    #del result, score
    
    # Press 'q' to quit 
    if key == ord('q'):
        # Quit program
        print("...Bye")
        break
    
    # Press 'c' to save capture
    elif key == ord('c'):
        # Capture and save photo as .png
        cv2.imwrite("{}.png".format(save_img_count), frame_new)
        print("Save captured photo as  {}.png".format(save_img_count))
        save_img_count += 1
    
    # Press 'm' to save multiple images
    elif key == ord('m'):
        gesture='03_rock'
        cwd = os.getcwd()
        output_dir= os.path.join(cwd,'image',gesture)
        if not os.path.exists(os.path.join(cwd,'image')):
            os.mkdir(os.path.join(cwd,'image'))
        if not os.path.exists(output_dir):           
            os.mkdir(output_dir)
        else:
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    img_count+=1
        while cv2.waitKey(100) != ord('m'):
            ret, frame_new, img_hand=capture_display()
            cv2.imshow('camera',frame_new)
            filename = os.path.join(output_dir,gesture+'_'+str(img_count)+'.png')        
            cv2.imwrite(filename, img_hand)
            print("Save "+gesture+"_{}.png".format(img_count), '...Press "m" to stop')
            img_count += 1
        img_count = 0 
        print(helpInfo)    
        
    #Press 'r' to start recognition
    elif key == ord('r'):
        print('...Press "r" to stop')
        while cv2.waitKey(100) != ord('r'):            
            ret, frame_new, img_hand =capture_display()
            cv2.imshow('camera',frame_new)
            img, result,score=predict_image(img_hand, height, width)
            cv2.imshow('hand', img)
            print('result: ', result, ' score: ', score)
            # Delete object otherwise memory overflow
            del img, result, score, ret, frame_new, img_hand
        cv2.destroyWindow("hand")
        print(helpInfo)   

    
# Release VideoCapture
cap.release()
# Distroy all windows
cv2.destroyAllWindows()