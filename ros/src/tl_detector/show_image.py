import cv2
#import matplotlib.pyplot as plt
#im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
#plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
#plt.show()

cv2.namedWindow("output", cv2.WINDOW_NORMAL)         # Create window with freedom of dimensions
while True:
    im = cv2.imread('images/image.jpg')                  # Read image
    imS = cv2.resize(im, (400, 300))                     # Resize image
    cv2.imshow("output", imS)                            # Show image
    cv2.waitKey(1)                                       # Display the image infinitely until any keypress