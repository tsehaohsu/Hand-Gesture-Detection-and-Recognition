import cv2 

# VideoCapture
# 0: default camera / USB webcam
# 1: USB webcam 2
# 2: USB webcam 3
# -1: New pluged USB webcam
cap = cv2.VideoCapture(0)

# Camera Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Create a window called "camera"
cv2.namedWindow('camera',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

# Resize Window
cv2.resizeWindow('camera',500,300)

# image count
img_count = 1

# Check if camera is opened
print("Is camera opened? {}".format(cap.isOpened()))

# Help information
helpInfo = '''
Press Q： Quit
Press C： Capture
'''
print(helpInfo)
while(True):
	# retrieve: True if read is successful
	# frame: image frame
    ret, frame = cap.read()
    if not ret:
        print("Unable to load camera.")
        break

    # Convert color RGB to gray
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## flip image
    # 1: horizontal
    # 0: vertical
    # -1: horizontal & vertical

    # flipCode = 1
    # frame = cv2.flip(frame, flipCode)

    # show image
    cv2.imshow('camera',frame)

    # Wait 1ms when event happened
    key = cv2.waitKey(1)
    if key == ord('q'):
        # Quit program
        print("...Bye")
        break
    elif key == ord('c'):
        # Capture and save photo as .png
        cv2.imwrite("{}.png".format(img_count), frame)
        print("Save captured photo as  {}.png".format(img_count))
        img_count += 1

# Release VideoCapture
cap.release()
# Distroy all windows
cv2.destroyAllWindows()