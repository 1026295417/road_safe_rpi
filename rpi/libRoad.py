import time, datetime
import imutils
import cv2
import numpy as np
import serial
import socket  # Import socket module

class objDetect_ssd:
    def __init__(self, objnames, model_path, pbtxt_path, img_size=(300,300), score=0.5):
        self.objnames = objnames
        #detect
        self.score = score
        self.nms = 0.5
        #bounding box
        self.bbox_show = True  #display on image
        self.bbox_color = (0,255,0)
        self.bbox_bolder = 1
        #classes
        self.classes_show = True  #display on image
        self.classes_color = (255,0,0)
        with open(objnames, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.inpWidth = img_size[0]
        self.inpHeight = img_size[1]

        dnn = cv2.dnn.readNetFromTensorflow(model_path, pbtxt_path)
        dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        self.net = dnn

    def getObject(self, frame, labelWant):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, size=(self.inpWidth, self.inpHeight), swapRB=True)
        model = self.net
        model.setInput(blob)
        output = model.forward()
        output[0,0,:,:].shape is (100, 7)

        classIds = []
        labelName = []
        confidences = []
        boxes = []
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > self.score:
                print(detection[1])
                class_id = int(detection[1]-1)
                label = self.classes[class_id]
                image_height, image_width, _ = frame.shape
                box_x=detection[3] * image_width
                box_y=detection[4] * image_height
                box_width=detection[5] * image_width
                box_height=detection[6] * image_height

                classIds.append(class_id)
                confidences.append(float(confidence))
                boxes.append((box_x, box_y, box_width, box_height))
                labelName.append(label)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score, self.nms)
        self.indices = indices
        out_bbox, out_scores, out_labels = [], [], []
        for ii in indices:
            i = ii[0]
            out_bbox.append(boxes[i])
            out_scores.append(confidences[i])
            out_labels.append(labelName[i])
        
        return out_bbox, out_scores, out_labels
        #return boxes, confidences, labelName



class obDetect_yolo:
    def __init__(self, objnames, weights, cfg, img_size=(416,416)):
        self.objnames = objnames
        self.weights = weights
        self.cfg = cfg
        #detect
        self.score = 0.3
        self.nms = 0.6
        #bounding box
        self.bbox_show = True  #display on image
        self.bbox_color = (0,255,0)
        self.bbox_bolder = 1
        #classes
        self.classes_show = True  #display on image
        self.classes_color = (255,0,0)
        with open(objnames, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.inpWidth = img_size[0]
        self.inpHeight = img_size[1]

        dnn = cv2.dnn.readNetFromDarknet(cfg, weights)
        #dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        self.net = dnn

    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def getObject(self, frame, labelWant):
        #print("YOLO DETECT.")
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        labelName = []
        confidences = []
        boxes = []
        blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
        # Sets the input to the network
        net = self.net
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(self.getOutputsNames(net))

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                label = self.classes[classId]
                print(label, confidence, classId, scores)
                if( confidence > self.score) :
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append((left, top, width, height))
                    labelName.append(label)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score, self.nms)
        self.indices = indices
        out_bbox, out_scores, out_labels = [], [], []
        '''
        predicts = {}
        for id, ii in enumerate(indices):
            i = ii[0]
            bbox = boxes[i]
            class_name = labelName[i]
            score = confidences[i]

            predicts.update( {id: (class_name, score, bbox)}  )
        '''
        for ii in indices:
            i = ii[0]
            out_bbox.append(boxes[i])
            out_scores.append(confidences[i])
            out_labels.append(labelName[i])

        return out_bbox, out_scores, out_labels

class webCam:
    def __init__(self, id=0, videofile="", size=(1920, 1080)):
        self.camsize = size
        #for FPS count
        self.start_time = time.time()
        self.last_time = time.time()
        self.total_frames = 0
        self.last_frames = 0
        self.fps = 0
        self.device = True

        if(len(videofile)>3):
            self.cam = cv2.VideoCapture(videofile)
            self.playvideo = True
        else:
            try:
                self.cam = cv2.VideoCapture(0+id)
                #self.cam = cv2.VideoCapture(cv2.CAP_DSHOW+id)
                self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
                self.playvideo = False
            except:
                self.device = False
                pass

    def fps_count(self, seconds_fps=10):
        fps = self.fps

        timenow = time.time()
        if(timenow - self.last_time)>seconds_fps:
            fps  = (self.total_frames - self.last_frames) / (timenow - self.last_time)
            self.last_frames = self.total_frames
            self.last_time = timenow
            self.fps = fps

        return round(fps,2)

    def working(self):
        webCam = self.cam

        try:
            test_cam = webCam.isOpened()
        except:
            test_cam = False
            pass

        if(test_cam is True):
            self.device = True
            return True
        else:
            if(self.playvideo is True):
                self.device = True
                return True
            else:
                self.device = False
                return False

    def camRealSize(self):
        webcam = self.cam
        width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def getFrame(self, rotate=0, vflip=False, hflip=False, resize=None):
        webcam = self.cam
        try:
            hasFrame, frame = webcam.read()
        except:
            hasFrame, frame = False, None
        
        if(frame is not None):
            if(vflip==True):
                frame = cv2.flip(frame, 0)
            if(hflip==True):
                frame = cv2.flip(frame, 1)
    
            if(rotate>0):
                frame = imutils.rotate_bound(frame, rotate)
            if(resize is not None):
                frame_resized = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
            else:
                frame_resized = None

        else:
            frame = None
            hasFrame = False
            frame_resized = None

        self.total_frames += 1


        return hasFrame, frame_resized, frame

    def release(self):
        webcam = self.cam
        webcam.release()

class recordDefects:
    def __init__(self, logfile, countList):
        self.logfile = open(logfile, "a")
        self.id = 0
        self.list = countList

    def write_log(self, date_time, label_log, dataGPS, url_prvIMG, url_orgIMG):
        id = self.id
        file = self.logfile
        dicCount = self.list
        string = ''
        for i, labelData in enumerate(label_log):
            x = labelData[0][0]
            y = labelData[0][1]
            w = labelData[0][2]
            h = labelData[0][3]
            cclass = labelData[1]
            score = labelData[2]

            string = "{}|{}|{}|{}|{}|{}|{}|{}|{}\n".format(id, i, date_time, (x,y,w,h), cclass, score, dataGPS, url_prvIMG, url_orgIMG)

            if(cclass in dicCount):
                num = dicCount[cclass] + 1
                dicCount.update( {cclass:num} )

        if(len(string)>0):
            id += 1
            self.id = id
            self.list = dicCount
            file.write(string)
            return True
        else:
            return False

    def close(self):
        file = self.logfile
        file.close()


class GPS:
    def __init__(self, comport, portrate, test=False):
        self.comport = comport
        if(test is False):
            try:
                self.hardware = True
                self.ser = serial.Serial(comport, portrate, 8, 'N', 1, timeout=1)
            except:
                self.hardware = False
        else:
            self.ser = None
            self.hardware = True
            
        self.portrate = portrate
        self.ddmmyy = ''
        self.hhmmss = ''
        self.localtime = ''
        self.gpsStatus = False
        self.NS = ''
        self.latitude = 0.0  #ddmm.mmmm
        self.longitude = 0.0 #dddmm.mmmm
        self.EW = 0.0
        self.speed = 0.0
        self.direction = 0.0
        self.mDeclination = 0.0
        self.mDirection = ''
        self.gmLati = 0.0 
        self.gmLong = 0.0

    def getGPS_rawline(self, gps_type="$GPRMC"):
        ser = self.ser
        data = ser.readline()
        if(gps_type in data):
            return data

    def UTC2TW_time(self, objDatetime): #objDatetime=(ddmmyy, hhmmss)
        txtDate = objDatetime[0]
        txtTime = objDatetime[1]
        txtDatetime = '20'+txtDate[4:6]+'-'+txtDate[2:4]+'-'+txtDate[0:2]+' '+txtTime[0:2]+':'+txtTime[2:4]+':'+txtTime[4:6]
        dateObj = datetime.datetime.strptime(txtDatetime, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(hours=8)
        txtYear = str(dateObj.year)
        txtYear = txtYear[2:4]
        txtMonth = str(dateObj.month)
        txtMonth = txtMonth.zfill(2)
        txtDay = str(dateObj.day)
        txtDay = txtDay.zfill(2)
        txtHour = str(dateObj.hour)
        txtMinute = str(dateObj.minute)
        txtSecond = str(dateObj.second)
        txtHour = txtHour.zfill(2)
        txtMinute = txtMinute.zfill(2)
        txtSecond = txtSecond.zfill(2)
        #print("TEST:",txtDay+txtMonth+txtYear)
        #print("TEST:", txtHour+txtMinute+txtSecond)

        return ( txtDay+txtMonth+txtYear, txtHour+txtMinute+txtSecond, \
            "20{}/{}/{} {}:{}:{}".format(txtYear,txtMonth,txtDay,txtHour,txtMinute,txtSecond) )

    def updateGPS(self):
        
        data = '$GPRMC,133900.709,V,,,,,,,180319,,,N*49'
        
        try:
            ser = self.ser
            while(ser.inWaiting()):
                data = str(ser.readline().decode('utf-8'))
                
            self.hardware = True    
        except:
            data = '$GPRMC,133900.709,V,,,,,,,180319,,,N*49'
            
            self.hardware = False
            pass

        if("GPRMC" in data):
            gps_raw = data.split(',')

            if(gps_raw[2] != 'A'):
                self.gpsStatus = False

            else:
                #print(gps_raw)
                gpsTWtime = self.UTC2TW_time( (gps_raw[9], gps_raw[1]) )
                self.ddmmyy = gpsTWtime[0]
                self.hhmmss = gpsTWtime[1]
                self.localtime = gpsTWtime[2]
                self.gpsStatus = True
                self.NS = gps_raw[4]
                self.latitude = float(gps_raw[3])  #ddmm.mmmm
                self.longitude = float(gps_raw[5]) #dddmm.mmmm
                #self.EW = gps_raw[6)
                self.speed = float(gps_raw[7])
                self.direction = float(gps_raw[8])
                self.mDeclination = float(gps_raw[9])
                self.mDirection = gps_raw[10]
                Lati, Long = float(gps_raw[3]), float(gps_raw[5])
                #float(txtLong[:3]) + float(txtLati[3:])/60
                lat1 = int(Lati/100)
                lat2 = (Lati/100 - lat1)*100/60
                self.gmLati = round(lat1 + lat2, 4)
                long1 = int(Long/100)
                long2 = (Long/100 - long1)*100/60
                self.gmLong = round(long1 + long2,4)

    def getGMinfo(self):
        return (self.hardware, self.gmLati, self.gmLong, self.ddmmyy, self.hhmmss)
        
class SOCKETSEND:
    def __init__(self, host, port, recv_bit, interval_wait=1):
        self.host = host
        self.port = port
        self.recv_bit = recv_bit
        self.wait = interval_wait
        self.detected_img = None
        self.detected_time = None

    def connect(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)             # Create a socket object
            self.s.settimeout(10)
            self.s.connect((self.host, self.port))
            self.conn_status = True

        except socket.error as msg:
            print(msg)
            #sys.exit(1)
            self.conn_status = False
            pass

    def send_file(self, gpsData, org_img_path ):
        
        filename = str.encode(gpsData)
        self.send_status = 1  # 0:Failed  1:sending  2: success
        recv_bit = self.recv_bit
        try:
            print('Send filename.')
            self.s.send(filename)
            self.send_status = 1

        except:
            print("Send filename.... got error.")
            self.send_status = 0

            return False

        f = open(org_img_path,'rb')
        l = f.read(recv_bit)

        try:
            print('Send defect image.')
            while (l):

               self.s.send(l)
               #print('Sent ',repr(l))
               l = f.read(recv_bit)

            f.close()
            self.send_status = 2

            return True

        except:
            f.close()
            print("Send image.... got error.")
            self.send_status = 0

            return False
            
    def send_gps(self, gpsData ):
        
        gps_info_data = str.encode(gpsData)
        self.send_status = 1  # 0:Failed  1:sending  2: success
        recv_bit = self.recv_bit
        try:
            print('Send gps data.')
            self.s.send(gps_info_data)
            self.send_status = 1

        except:
            print("Send gps data.... got error.")
            self.send_status = 0

            return False


        self.send_status = 2
        return True

