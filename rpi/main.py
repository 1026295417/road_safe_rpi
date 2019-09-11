#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, sys, time
import signal
import random
import time
from datetime import datetime
import cv2
import numpy as np
import imutils
from libRoad import webCam
from libRoad import recordDefects
from libRoad import GPS
from libRoad import obDetect_yolo
from libRoad import objDetect_ssd
from PIL import ImageFont, ImageDraw, Image
import multiprocessing as mp
from easygui import ynbox
from subprocess import call
from configparser import ConfigParser
import multiprocessing as mp
from libRoad import SOCKETSEND
import glob

defect_list = { 'D00': '縱向裂縫輪痕', 'D01': '縱向裂縫施工', 'D10': '橫向裂縫間隔', 'D11': '橫向裂縫施工', \
    'D12': '縱橫裂縫', 'D20': '龜裂', 'D21': '人孔破損', 'D30': '車轍', 'D31': '路面隆起', \
    'D40': '坑洞', 'D41': '人孔高差', 'D42': '薄層剝離', 'D50': '人孔缺失', 'D51': '正常人孔' }
defect_count = { 'D00': 0, 'D01': 0, 'D10': 0, 'D11': 0, 'D12': 0, 'D20': 0, 'D21': 0, 'D30': 0, 'D31': 0, \
    'D40': 0, 'D41': 0, 'D42': 0, 'D50': 0, 'D51': 0 }

cfg = ConfigParser()
cfg.read("road.ini",encoding="utf-8")

#--Configurations for users ------------------------------------
#global
car_id = cfg.get("global", "car_id")
flash_disk_folder = cfg.get("client_server", "flash_disk_folder")

# web camera
cam_id = cfg.getint("camera", "cam_id")
webcam_size = (cfg.getint("camera", "webcam_size_w"), cfg.getint("camera", "webcam_size_h"))  #(1920, 1080)
screen_size = (cfg.getint("screen", "screen_size_w"), cfg.getint("screen", "screen_size_h")) #size for video frame
lcd_size = (cfg.getint("screen", "lcd_size_w"), cfg.getint("screen", "lcd_size_h"))
cam_rotate = cfg.getint("camera", "cam_rotate")
flip_vertical = cfg.getboolean("camera", "flip_vertical")
flip_horizontal = cfg.getboolean("camera", "flip_horizontal")

#file logging
#flash_disk_name = cfg.get("log", "flash_disk_name")
log_folder = os.path.join(flash_disk_folder, cfg.get("log", "log_folder"))
interval_seconds_logging = cfg.getint("log", "interval_seconds_logging")

# GPS
#PC的TTL2USB port
comPort = cfg.get("gps", "comPort")
baudRate = cfg.getint("gps", "baudRate")
interval_gps_upload = cfg.getint("gps", "interval_gps_upload")

#web
web_path = "FLASH\\web"
defect_info_write = os.path.join(web_path,"defects.log")

#detect
interval_detect = cfg.getint("detect", "interval_detect")  #frames
full_screen = cfg.getboolean("screen", "full_screen")
same_gps_no_uload = cfg.getboolean("detect", "same_gps_no_uload")

#upload
#flash_disk_folder = cfg.get("client_server", "flash_disk_folder")
upload_host = cfg.get("client_server", "host")
upload_port = cfg.getint("client_server", "port")
upload_interval = cfg.getint("client_server", "interval_upload")
recv_bit = cfg.getint("client_server", "recv_bit")
img_waiting_path = os.path.join(flash_disk_folder, cfg.get("client_server", "img_waiting_up"))
img_possible_defetct = os.path.join(flash_disk_folder, cfg.get("client_server", "img_possible_defetct"))
img_uploaded_path = os.path.join(flash_disk_folder, cfg.get("client_server", "img_uploaded"))

#--Configurations for developers -------------------------------
#for Demo ir debug, use video to replace camera
simulate = cfg.get("simulate", "simulate")
#simulate = ""
appStatus = cfg.getboolean("simulate", "appStatus")   # the status for this program, if False then exit
win_name = cfg.get("developer", "win_name")  #cv2 window name
reference_size = (cfg.getint("developer", "reference_size_w"), cfg.getint("developer", "reference_size_h"))  # frame will resized for inference


dmodel = objDetect_ssd(objnames="../models/cfg.road.yolo_tiny.all/obj.names", \
    model_path="../models/ssd_mobilenet.all/frozen_inference_graph.pb", \
    pbtxt_path="../models/ssd_mobilenet.all/frozen_all_road.pbtxt", \
    img_size=(300,300))

'''
dmodel = obDetect_yolo(objnames="../models/cfg.road.yolo_tiny.all/obj.names", \
    weights="../models/cfg.road.yolo_tiny.all/yolov3-tiny_500000.weights", \
    cfg="../models/cfg.road.yolo_tiny.all/yolov3-tiny.cfg", \
    img_size=reference_size )
'''


#--Configurations for auto-caculated ---------------------------
reference_size_ratio = (reference_size[0]/webcam_size[0],reference_size[1]/webcam_size[1] )

def setEnv():
    if len(simulate)>0 and (not os.path.exists(simulate)):
        print("No such video file:", simulate)
        sys.exit()

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if(os.path.isfile("count_upload.txt")):
        os.remove("count_upload.txt")

    if not os.path.exists(img_waiting_path):
        os.makedirs(img_waiting_path)

    if not os.path.exists(img_possible_defetct):
        os.makedirs(img_possible_defetct)

    if not os.path.exists(img_uploaded_path):
        os.makedirs(img_uploaded_path)

    if(full_screen is True):
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)        # Create a named window
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

def check_env(conn_data):
    (host, port, recv_bit, interval) = conn_data
    imgUP = SOCKETSEND(host, port, recv_bit, interval)
    imgUP.connect()
    send_status = imgUP.send_file("test_upload.jpg", "test_upload.jpg" )
    if(send_status is True):
        print("[OK] Image upload to taifo.bim-group.com")
    else:
        print("[FAILED] Image upload to taifo.bim-group.com")
        
    if(gpsDevice.gpsStatus is True):
        print("[OK] GPS device status.")
    else:
        print("[FAILED] GPS device status.")
        
    if(CAMERA.working() is True):
        print("[OK] Web camera device.")
    else:
        print("[FAILED] Web camera device.")
        
    return (send_status, gpsDevice.gpsStatus, CAMERA.working())

def mouseClick(event,x,y,flags,param):
    if event == 4:
        if(y<410+btn_poweroff.shape[0] and y>410) and (x<700+btn_poweroff.shape[1] and x>700):
            if ynbox("您要結束程式並關機嗎？", "結束並關機"):     # show a Continue/Cancel dialog
                exit_app(poweroff=True)
            else:  # user chose Cancel
                pass


def printText(txt, bg, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        ## Use cv2.FONT_HERSHEY_XXX to write English.
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "wt009.ttf"
        #print("TEST", txt)
        font = ImageFont.truetype(fontpath, int(size*20))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)

    return bg

def new_log_filename():
    #now = datetime.now()
    #return now.strftime("%Y-%m-%d_%H%M%S")+".log"
    now = gpsDevice.localtime
    if(len(now)>0):
        now=now.replace(' ','_')
        nowtxt = now+".log"
    else:
        now = datetime.now()
        nowtxt = now.strftime("%Y-%m-%d_%H%M%S")+".log"

    return nowtxt

def desktop_bg():
    img_desktop = np.zeros((lcd_size[1], lcd_size[0], 3), np.uint8)
    img_desktop[0:logo.shape[0], 0:logo.shape[1]] = logo
    img_desktop[410:410+btn_poweroff.shape[0], 700:700+btn_poweroff.shape[1]] = btn_poweroff
    locX, locY = lcd_size[0]-155, 120
    height_y = 25
    for class_name in defect_count:
        if(defect_count[class_name]>0):
            txtNum = str(defect_count[class_name]).rjust(5, ' ')
            img_desktop = printText(txtNum, img_desktop, color=(0,255,255,0), size=0.75, pos=(locX, locY), type="Chinese")
            img_desktop = printText(defect_list[class_name], img_desktop, color=(255,255,255,0), size=0.75, pos=(locX+55, locY), type="Chinese")
            locY += height_y

    #gps_status, gps_lati, gps_long, gps_dmy, gps_hms
    locX, locY = 10, lcd_size[1]-30
    btm_line = "[GPS] 緯度: {} 經度: {} ".format( gps_lati, gps_long)
    img_desktop = printText(btm_line, img_desktop, color=(255,255,0,0), size=0.70, pos=(locX, locY), type="Chinese")
    img_desktop = printText(gpsDevice.localtime, img_desktop, color=(0,255,255,0), size=1.05, pos=(380,35), type="Chinese")

    return img_desktop
    
def upload_img(interval_gps, waiting_path, uploaded_path, conn_data):

    gps_upload_time = time.time()
    (host, port, recv_bit, interval) = conn_data
    imgUP = SOCKETSEND(host, port, recv_bit, interval)
    print("Starting upload process.....", waiting_path, conn_data)
    id = 0
    waiting = 0
    
    while True:
    
        if(time.time() - gps_upload_time > interval_gps ):
            gps_upload_time = time.time()
            if(os.path.isfile("gps_tracking.txt")):
                fgps = open("gps_tracking.txt", "r")
                gps_line = fgps.readline()
                fgps.close()
                
                imgUP.connect()
                gps_upload_time = time.time()
                send_status = imgUP.send_gps(gps_line )
            
        else:
            list_of_files = glob.glob(os.path.join(waiting_path,'*.jpg'))
            waiting = len(list_of_files)
            
            if(len(list_of_files)>0):
                print("File counts:", len(list_of_files))
                file = max(list_of_files, key=os.path.getctime)
                img_filename = os.path.basename(file)
                filename, file_extension = os.path.splitext(img_filename)
                file_extension = file_extension.lower()
                
                if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                    imgUP.connect()

                    send_status = imgUP.send_file(img_filename, os.path.join(waiting_path,img_filename) )
                    
                    if(send_status is True):
                        id += 1
                        waiting -= 1
                        try:
                            os.rename(os.path.join(waiting_path,img_filename), os.path.join(uploaded_path,img_filename))
                            print("uploaded, remove it from detected folder", file)
                            
                        except:
                            print("uploaded, but move file to uploaded folder failed, delete it.")
                            os.remove(os.path.join(waiting_path,img_filename))
                            pass
                            
                        f = open("count_upload.txt","w")
                        f.write("{},{}".format(id, waiting))
                        f.close()
        
def save_img_waiting(uploaded_path, img_org_data, img_prv_data):

    rtn = False
    path_org = img_org_data[0]
    img_org = img_org_data[1]
    path_prv = img_prv_data[0]
    img_prv = img_prv_data[1]

    try:
        cv2.imwrite(path_org, img_org)
        #os.rename(path_org, uploaded_path)
        cv2.imwrite(path_prv, img_prv)
        rtn = True

    except:
        print("write image error.")
        pass
    '''
    if(writed is True):    
        filename = os.path.basename(path_org)
        imgUP.connect()
        send_status = imgUP.send_file(filename, path_org )
        
        if(send_status is True):
            try:
                print("uploaded, remove it from detected folder", file)
                os.rename(path_org, uploaded_path)
            except:
                print("uploaded, but move file to uploaded folder failed.")
                pass
        
        rtn = True
    '''


    return rtn

def process_local_web(img_org_data, img_big_data, img_prv_data):
    rtn = False
    path_org = img_org_data[0]
    img_org = img_org_data[1]
    path_prv = img_prv_data[0]
    img_prv = img_prv_data[1]
    path_large = img_big_data[0]
    img_large = img_big_data[1]

    try:
        cv2.imwrite(path_org, img_org) #暫時不寫原尺寸檔案（太大）
        cv2.imwrite(path_prv, img_prv)
        cv2.imwrite(path_large, img_large)
        rtn = True

    except:
        pass

    return rtn

def new_imgname(gps_status):
    if(gps_status is True):
        filename = "{}_{}_{}_{}.jpg".format(gps_lati,gps_long,gps_dmy,gps_hms)
    else:
        filename = "{}_{}_{}_{}.jpg".format('no','gps','device',time.time())

    return filename

def exit_app(poweroff=False):
    pool_saveimg.close()
    pool_saveimg.terminate()
    pool_saveimg.join()
    pool_uploadimg.close()
    pool_uploadimg.terminate()
    pool_uploadimg.join()

    logDefects.close()
    #f.close()
    print("End.....")
    #sys.exit(0)
    os.system("taskkill /f /im Python.exe")
    os.system("taskkill /f /im ROAD-safe.exe")
    
    if(poweroff is False):
        sys.exit(0)
    else:
        call("sudo poweroff", shell=True)

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    exit_app()

if __name__ == '__main__':
    setEnv()
    signal.signal(signal.SIGINT, signal_handler)
    cv2.setMouseCallback(win_name , mouseClick)
    last_logging_time = 0
    frameID = 0
    nBoxes = []
    #desktop_bg
    logo = cv2.imread("img/logo.png")
    btn_poweroff = cv2.imread("img/poweroff.png")

    #multi-process
    pool_result = []
    pool_saveimg = mp.Pool(processes = 2)
    pool_uploadimg = mp.Pool(processes = 1)
    proc_upload = pool_uploadimg.apply_async(upload_img, (interval_gps_upload, img_waiting_path, img_uploaded_path,(upload_host, upload_port, recv_bit, upload_interval), ))
    #upload_img(interval_gps_upload, img_waiting_path, img_uploaded_path,(upload_host, upload_port, recv_bit, upload_interval))

    print("Upload img status:", proc_upload)
    gpsDevice = GPS(comport=comPort, portrate=baudRate, test=True)
    last_gps_logging = time.time()

    CAMERA = webCam(id=cam_id, videofile=simulate, size=webcam_size)
    if(CAMERA.working() is False):
        print("webcam cannot work.")
        appStatus = False
        sys.exit()

    defect_file = new_log_filename()
    logDefects = recordDefects(os.path.join(log_folder, defect_file), defect_count)
    #f = open(defect_info_write, "a")  # for web

    last_long, last_lati = 00, 0.0
    count_waiting, count_upload = 0, 0
    (s_upload, s_gps, s_cam) = check_env((upload_host, upload_port, recv_bit, upload_interval))
    
    while appStatus:
        gpsDevice.updateGPS()
        (gps_status, gps_lati, gps_long, gps_dmy, gps_hms) = gpsDevice.getGMinfo()
            
        if(time.time() - last_gps_logging > interval_gps_upload):
            f = open("gps_tracking.txt","w")
            f.write("{},{},{},{},{},{}".format("GPS_TRACKS", car_id, gps_lati, gps_long, gps_dmy, gps_hms))
            f.close()
            last_gps_logging = time.time()

        hasFrame, frame_inference, frame_org = CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=None)
        #frame_inference[0:int(frame_inference.shape[0]/2), 0:frame_inference.shape[1]] = (0,0,0)
        frameID += 1
        #cv2.imshow("TEST", frame_inference)
        if(hasFrame is False):
            exit_app()

        frame_screen = cv2.resize(frame_org, screen_size, interpolation=cv2.INTER_CUBIC)
        frame_inference = frame_screen.copy()
        frame_inference[0:int(frame_screen.shape[0]/4), 0: frame_screen.shape[1]] = (255,255,255)
        #cv2.imshow("TEST", frame_inference)
        web_defect_txt = ""
        
        if((frameID % interval_detect==0) and (last_long!=gps_long and last_lati!=gps_lati)):
        #if((frameID % interval_detect==0)):
            last_long, last_lati = gps_long, gps_lati
            
            #check upload counts
            if(os.path.isfile("count_upload.txt")):
                try:
                    fcount = open("count_upload.txt", "r")
                    [count_upload, count_waiting] = fcount.readline().split(",")
                    fcount.close()
                except:
                    print("Read count_upload.txt error.")
            
            #-put labels to frame------------------
            nBoxes = []
            boxes, scores, class_name = dmodel.getObject(frame_inference, labelWant="")
            for i, (nx,ny,nw,nh) in enumerate(boxes):
                #nx = int( (x/reference_size[0]) * screen_size[0] )
                #ny = int( (y/reference_size[1]) * screen_size[1] )
                #nw = int( (w/reference_size[0]) * screen_size[0] )
                #nh = int( (h/reference_size[1]) * screen_size[1] )
                nx, ny, nw, nh = int(nx), int(ny), int(nw), int(nh)
                nBoxes.append( (nx, ny, nw, nh) )
                print(nx, ny, nw, nh)
                cv2.rectangle(frame_screen, (nx, ny), (nx+nw, ny+nh), (0,255,0), 1)
                frame_screen = printText("{}({}%)".format(defect_list[class_name[i]], int(scores[i]*100)), frame_screen, color=(255,255,0,0), size=0.95, pos=(nx, ny-30), type="Chinese")
                web_defect_txt += "{}:{}({}%)  ".format( class_name[i], defect_list[class_name[i]] ,int(scores[i]*100))

                defect_count[class_name[i]] += 1

            if(len(nBoxes)>0):
                img_file_name = new_imgname(gps_status)
                waiting_path = os.path.join(img_waiting_path, img_file_name)
                possible_defetct = os.path.join(img_possible_defetct, img_file_name)
                uploaded_path = os.path.join(img_uploaded_path, img_file_name)

                r = pool_saveimg.apply_async(save_img_waiting, (uploaded_path, (waiting_path,frame_org), (possible_defetct, imutils.resize(frame_screen,width=300)), )) #向進程池中添加事件
                now = gpsDevice.localtime
                #f.write("{}|{},{}|{}|{}\n".format(gpsDevice.localtime , gps_lati, gps_long, web_defect_txt, img_file_name))

                last_logging_time = time.time()
                

        #-generate defect statistics------------
        # Create a blank 300x300 black image
        desktop = desktop_bg()
        nx, ny = 10, 75  #nx, ny is the left point of the video frame
        #print(frame_screen.shape, desktop.shape)
        desktop[ny:ny+frame_screen.shape[0], nx:nx+frame_screen.shape[1]] = frame_screen
 
        desktop = printText("waiting:{}".format(count_waiting), desktop, color=(0,255,255,0), size=0.85, pos=(680,30), type="Chinese")
        desktop = printText("uploaded:{}".format(count_upload), desktop, color=(0,255,0,0), size=0.85, pos=(680,50), type="Chinese")
        desktop = printText("UP:{} GPS:{} CAM:{}".format(s_upload, gps_status, s_cam), desktop, color=(0,255,255,0), size=0.7, pos=(400,lcd_size[1]-30), type="Chinese")
        
        
        
        cv2.imshow(win_name, desktop)

        key = cv2.waitKey(1)
        if(key==113):
            exit_app()

        #print("FPS:", CAMERA.fps_count(10), "GPS:", gps_status, gps_lati, gps_long, gps_dmy, gps_hms)
