from yoloOpencv import opencvYOLO
import cv2
import imutils
import time

yolo = opencvYOLO(modeltype="yolov3", \
    objnames="../models/cfg.road.yolo_tiny.all/obj.names", \
    weights="../models/cfg.road.yolo_tiny.all/weights/yolov3-tiny_500000.weights",\
    cfg="../models/cfg.road.yolo_tiny.all/yolov3-tiny.cfg")

inputType = "video"  # webcam, image, video
media = "../test_videos/Edge03_20190321095303.avi"
write_video = False
video_out = "/media/sf_VMshare/out_hand1_yolov3.avi"
output_rotate = False
rotate = 0

#fps count
start = time.time()
def fps_count(num_frames):
    end = time.time()
    seconds = end - start
    fps  = num_frames / seconds;
    #print("Estimated frames per second : {0}".format(fps))
    return fps

if __name__ == "__main__":

    if(inputType == "webcam"):
        INPUT = cv2.VideoCapture(0)
    elif(inputType == "image"):
        INPUT = cv2.imread(media)
    elif(inputType == "video"):
        INPUT = cv2.VideoCapture(media)

    if(inputType == "image"):
        yolo.getObject(INPUT, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
        format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
        cv2.imshow("Frame", INPUT)

        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()

    else:
        if(video_out!=""):
            width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

            if(write_video is True):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(video_out,fourcc, 30.0, (int(width),int(height)))

        frameID = 0

        while True:
            hasFrame, frame = INPUT.read()
            frame = cv2.resize(frame, (800,480))
            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                break

            if(output_rotate is True):
                frame = imutils.rotate(frame, rotate)

            yolo.getObject(frame, labelWant="", drawBox=True, bold=2, textsize=1.2, bcolor=(0,255,0), tcolor=(0,0,255))
            #print ("Object counts:", yolo.objCounts)
            #yolo.listLabels()
            if(yolo.objCounts>0):
                print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
                    format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )

            cv2.imshow("Frame", frame)
            frameID += 1
            fps_count(frameID)

            if(write_video is True):
                out.write(frame)

            k = cv2.waitKey(1)
            if k == 0xFF & ord("q"):
                if(write_video is True):
                    out.release()

                break
