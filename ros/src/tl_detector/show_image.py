import cv2
#import matplotlib.pyplot as plt
#im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
#plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
#plt.show()

while True:
    img = cv2.imread('images/image.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(1)