import sys, getopt, os
import cv2

def usage():
    print('Usage:\n   python test.py -m <mode> -d <target directory>')
    print('      --mode[m]: realtime or all_saved')

def show(directory, file):
    path = os.path.join(directory, file)
    im = cv2.imread(path)
    #imS = cv2.resize(im, (400, 300))
    cv2.imshow("output", im)
    cv2.waitKey(100)

def main(argv):
    runmode = ''
    directory = ''
    try:
        opts, args = getopt.getopt(argv,"hm:d:",["mode=","directory="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-m", "--mode"):
            if arg not in ("realtime", "all_saved"):
                usage()
                sys.exit(2)
            else:
                runmode = arg
        elif opt in ("-d", "--directory"):
            directory = arg
    print('Mode is "', runmode)
    print('Directory is "', directory)

    #cv2.namedWindow("output", cv2.WINDOW_NORMAL) # Create window with freedom of dimensions
    cv2.namedWindow("output")

    if runmode == 'realtime':
        file = 'image.jpg'
        while True:
            show(directory, file)
    elif runmode == 'all_saved':
        for file in sorted(os.listdir(directory)):
            if file.endswith(".jpg"):
                show(directory, file)

if __name__ == "__main__":
   main(sys.argv[1:])

#import matplotlib.pyplot as plt
#im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
#plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
#plt.show()

