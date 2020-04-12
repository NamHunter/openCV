import numpy as np
import cv2
import cv2 as cv
import video
import math

cascade = 0
counter = 0

class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src)
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')

        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False


    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    '''
    @param: img the image for the face detection
    @param: cascade the cascade of the ViolaJones face detection
    @return: rects, an array of the cornors of the detected face. [x1 y1 x2 y2]
    '''
    def detect(self,img, cascade):

        # Detect the faces
        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(150, 150), flags = cv.CV_HAAR_SCALE_IMAGE)

        # Check if any faces are detected
        if len(rects) == 0:

            # return empty array
            return []
        else:
            # Get the correct x and y values
            rects[:,2:] += rects[:,:2]

            # loop over the recs and shrink the width with 40%
            for rec in rects:
                rec[0] = rec[0] + int(math.floor(((rec[2] - rec[0])*0.4)/2))
                rec[2] = rec[2] - int(math.floor(((rec[2] - rec[0])*0.4)/2))

            return rects

    def draw_rects(self,img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


    def getFace(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rects = self.detect(gray, cascade)
        self.rects = rects
        img = self.draw_rects(img, rects, (0, 255, 0))

        if len(rects) != 0:
            self.selection = rects[0][1], rects[0][0], rects[0][3], rects[0][2]

        return rects


    def run(self):
        counter= 0
        rects = None
        while True:
            counter +=1;
            ret, self.frame = self.cam.read()
            vis = self.frame.copy()

            if counter % 150 == 0:
                rects = self.getFace(vis);

            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            if rects is not None:
                self.draw_rects(vis, rects, (0, 255, 0))

            if self.selection:

                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1-x0, y1-y0)
                hsv_roi = hsv[x0:x1,y0:y1]
                mask_roi = mask[x0:x1,y0:y1]
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                self.hist = hist.reshape(-1)
                self.show_hist()

                vis_roi = vis[x0:x1,y0:y1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0
                self.tracking_state = 1
                self.selection = None



            if self.tracking_state == 1:
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                try: cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                except: print (track_box)
            cv2.imshow('camshift', vis)


            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "haarcascade_frontalface_alt.xml")
    cascade = cv2.CascadeClassifier(cascade_fn)

    App(video_src).run()