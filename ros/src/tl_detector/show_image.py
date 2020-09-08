import cv2
#import matplotlib.pyplot as plt
#im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
#plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
#plt.show()

# while True:
#     img = cv2.imread('images/image.jpg')
#     cv2.imshow('image', img)
#     cv2.waitKey(1)


cv2.namedWindow("output", cv2.WINDOW_NORMAL)         # Create window with freedom of dimensions
im = cv2.imread('images/image.jpg')                  # Read image
imS = cv2.resize(im, (400, 300))                     # Resize image
cv2.imshow("output", imS)                            # Show image
cv2.waitKey(0)                                       # Display the image infinitely until any keypress