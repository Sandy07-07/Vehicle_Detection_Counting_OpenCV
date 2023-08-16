import cv2
import numpy as np
import cvzone

# Fetch the video
cap = cv2.VideoCapture("vehicle_counter.mp4")

# Counting Line position
count_line_pos = 550

# Minimum width & height of rectangle over each vehicle
min_width = 80
min_height = 80

# Initialize the Background Subtractor Algorithm which subtracts the background from the object
algo = cv2.createBackgroundSubtractorKNN()


# To draw the circle passing which the vehicle would be counted
def centre_handle(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# The list to position of vehicles
detect = []

# Allowable error between pixel is offset
offset = 6

# Counter to store the number of vehicles
counter = 0

while True:
    # Read the frames of the video one by one
    ret, frame1 = cap.read()

    # Convert to Grayscale
    imgGray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Add Blur to the image
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 5)

    # Applying Algo on each frame
    img_sub = algo.apply(imgBlur)

    # Dilate the frame image
    imgDilate = cv2.dilate(img_sub, np.ones((3, 3)))

    # Get the structuring element for morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # To perform advanced morphological operations gives to them
    dilat = cv2.morphologyEx(imgDilate, cv2.MORPH_CLOSE, kernel)
    dilat = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

    # Contours the binary image
    contour_shape, h = cv2.findContours(dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing the counter line
    cv2.line(frame1, (25, count_line_pos), (1200, count_line_pos), (255, 127, 0), 5)

    # To automatically create rectangles over each vehicle
    for i, c in enumerate(contour_shape):
        # Draw a rectangle of exact shape in the Binary image
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width) and (h >= min_height)
        if not validate_counter:
            continue

        # Draw the rectangle over the vehicle
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Print the vehicle no. over the top of the rectangle
        cv2.putText(frame1, "Vehicle No." + str(counter), (x, y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

        # Circle to check if the car passes the line
        centre = centre_handle(x, y, w, h)
        detect.append(centre)
        cv2.circle(frame1, centre, 4, (0, 0, 255), -1)

        # Counting the vehicle
        for (x, y) in detect:
            if (count_line_pos + offset) > y > (count_line_pos - offset):
                counter += 1

            # Change the colour of line if vehicle pass
            cv2.line(frame1, (25, count_line_pos), (1200, count_line_pos), (71, 71, 209), 5)
            detect.remove((x, y))

    # Show the total no. of vehicles passed till now
    cvzone.putTextRect(frame1, "Vehicle Counter :- " + str(counter), (450, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, colorR=(0, 200, 0))

    # Show the video frame by frame
    cv2.imshow("Vehicle Original", frame1)

    if cv2.waitKey(10) == 13:
        break

# Close all the windows open
cv2.destroyAllWindows()

# Release all the video frames
cap.release()
