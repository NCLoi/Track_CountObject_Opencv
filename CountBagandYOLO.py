from tkinter import NORMAL, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter.ttk import Label, Button, Entry
import cv2
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from ultralytics import YOLO
import math
import datetime

# Tracker
class Tracker:
    def __init__(self):
        # Luu center point obj
        self.center_points = {}
        # tang id khi co obj moi
        self.id_count = 1

    def update(self, objects_rect):
        # boxes and ids
        objects_bbs_ids = []

        # Lay diem trung tam new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # check doi tuong da được phát hiện
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                # khoảng cách giữa tâm của bounding box hiện tại và tâm của đối tượng đã được theo dõi trước đó
                if dist < 150:
                    self.center_points[id] = (cx, cy)
                    # cập nhật đối tượng id trước đó với center point của bb hiện tại 
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # gán ID cho đối tượng mới 
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids


# Tkiner
class Window(Tk):
    def __init__(self):
        super().__init__()
        self.wm_title("Đếm sản phẩm trên băng chuyền")
        self.geometry("750x570")
        self.initUI()

        self.path_model = StringVar()
        show_path_model = Entry(self, textvariable=self.path_model, width=20)
        show_path_model.place(x=200, y=380)
        btn_slm = Button(self, text="Model", command=self.select_model)
        btn_slm.place(x=320, y=377, width=50, height=25)

    def initUI(self):
        self.paused = False
        pause_button = Button(self, text="Pause", command=self.pause_video)
        pause_button.place(x=300, y=300, width=50, height=35)
        play_button = Button(self, text="Play", command=self.play_video)
        play_button.place(x=350, y=300, width=50, height=35)

        btn_run = Button(self, text="OpenCV", command=self.show_frames)
        btn_run.place(x=70, y=440, width=100, height=50)
        
        btn_run_yl = Button(self, text="YOLOv8", command=self.show_frames_yolo)
        btn_run_yl.place(x=70, y=500, width=100, height=50)

        btn_slvid = Button(self, text="Select Video", command=self.select_file)
        btn_slvid.place(x=200, y=440, width=100, height=50)

        btn_cmr = Button(self, text="Camera", command=self.camera)
        btn_cmr.place(x=200, y=500, width=100, height=50)
        
        reset_btn = Button(self, text="Reset", command=self.reset)
        reset_btn.place(x=330, y=440, width=100, height=50)

        img = cv2.imread('no_input.png')
        img = cv2.resize(img, (480, 270))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.img = ImageTk.PhotoImage(img)
        self.label = Label(self, image=self.img)
        self.label.place(x=140, y=20)

        label_if_Ymax = Label(self, text="Y min = 0",font=("Arial", 10))
        label_if_Ymax.place(x=70, y=20)
        label_if_Ymin = Label(self, text="Y max = 270",font=("Arial", 10))
        label_if_Ymin.place(x=55, y=270)

        label_if_Xmax = Label(self, text="X max = 480",font=("Arial", 10))
        label_if_Xmax.place(x=570, y=300)
        label_if_Xmin = Label(self, text="X min = 0",font=("Arial", 10))
        label_if_Xmin.place(x=140, y=300)

        label_fp = Label(self, text="File path:",font=("Arial", 11))
        label_fp.place(x=160, y=410)
        self.file_path_var = StringVar()
        show_fp = Entry(self,textvariable=self.file_path_var, width=30)
        show_fp.place(x=220, y=410)

        #
        label_fp = Label(self, text="Total:",font=("Arial", 11))
        label_fp.place(x=420, y=410)

        self.cnt_obj = IntVar()
        show_num = Entry(self,textvariable=self.cnt_obj, width=5)
        show_num.place(x=460, y=410)
        btn_txt = Button(self, text="Xuất file",command=self.txt_file)
        btn_txt.place(x=500, y=410)

        self.obj_detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=30)
        self.cntframe = 0
        self.tracker = Tracker()
        self.arr = set()

        # get area 
        self.xmin = 10
        self.xmax, self.ymin, self.ymax = 470, 5, 260
        self.area_Cnt = [(self.xmin, self.ymin), (self.xmin, self.ymax), (self.xmax, self.ymax), (self.xmax, self.ymin)]
        self.area_Cnt = np.array(self.area_Cnt, np.int32)
        
        label_area = Label(self, text="Nhập giá trị vùng đếm:",font=("Arial", 11))
        label_area.place(x=470, y=435)
        self.xmin_entry = Entry(self,textvariable=self.xmin,width=4)
        self.xmin_entry.place(x=520, y=460)
        label_xmin = Label(self, text="X min:",font=("Arial", 11))
        label_xmin.place(x=470, y=460)

        self.xmax_entry = Entry(self, textvariable=self.xmax, width=4)
        self.xmax_entry.place(x=620, y=460)
        label_xmax = Label(self, text="X max:",font=("Arial", 11))
        label_xmax.place(x=570, y=460)
        
        self.ymin_entry = Entry(self,textvariable=self.ymin, width=4)
        self.ymin_entry.place(x=520, y=490)
        label_ymin = Label( text="Y min:",font=("Arial", 11))
        label_ymin.place(x=470, y=490)

        self.ymax_entry = Entry(self, textvariable=self.ymax, width=4)
        self.ymax_entry.place(x=620, y=490)
        label_ymax = Label(self, text="Y max:",font=("Arial", 11))
        label_ymax.place(x=570, y=490)

        get_area_button = Button(self, text="Lưu giá trị", command=self.save_area)
        get_area_button.place(x=650, y=470)

    def txt_file(self):
        # mở file để ghi
        with open('Total.txt', 'w', encoding='utf-8') as f:
            if self.file_path_var == 0:
                f.write('Camera')
            else:
                f.write(self.file_path_var.get())
            # ghi biến
            f.write('\nSố lượng đã đếm được: ' + str(self.cnt_obj.get()))
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write('\nThời gian ghi: ' + str(current_time))

    def save_area(self):
        self.xmin = int(self.xmin_entry.get())
        self.xmax = int(self.xmax_entry.get())
        self.ymin = int(self.ymin_entry.get())
        self.ymax = int(self.ymax_entry.get())

        self.area_Cnt[0] = (self.xmin, self.ymin)
        self.area_Cnt[1] = (self.xmin, self.ymax)
            
        self.area_Cnt[2] = (self.xmax, self.ymax)
        self.area_Cnt[3] = (self.xmax, self.ymin)

        self.area_Cnt[0] = (self.xmin, self.ymin)
        self.area_Cnt[3] = (self.xmax, self.ymin)

        self.area_Cnt[1] = (self.xmin, self.ymax)
        self.area_Cnt[2] = (self.xmax, self.ymax)
        
        self.update_area_cnt()
    
    def update_area_cnt(self):
        self.area_Cnt = np.array([(self.xmin, self.ymin), (self.xmin, self.ymax), (self.xmax, self.ymax), (self.xmax, self.ymin)], np.int32)

    def reset(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.initUI()

    def select_model(self):
        file_model = filedialog.askopenfile()
        if file_model:
            file_path = file_model.name
            self.path_model.set(file_path)
            self.model = YOLO(file_path)

    def select_file(self):
        self.initUI()
        file_path = filedialog.askopenfilename()
        self.cap = cv2.VideoCapture(file_path)
        self.file_path_var.set(file_path)    

    def camera(self):
        self.initUI()
        file_path = 0
        self.cap = cv2.VideoCapture(file_path)

    def show_frames(self):
        if self.paused == False:
            ret, frame = self.cap.read()
            if self.cap is None:
                messagebox.showwarning("Warning", "Please select a video file first.")
                return
            self.cntframe += 1
            if self.cntframe % 2 == 0:
                frame = cv2.resize(frame,(480, 270))
                mask = self.obj_detector.apply(frame)
                
                mask  = cv2.GaussianBlur(mask , (3,3), 0)
                kernel1 = np.ones((5,5), np.uint8)
                kernel2 = np.ones((15,15), np.uint8)
                kernel3 = np.ones((9,9), np.uint8)
                mask = cv2.erode(mask, kernel1, iterations=1)
                mask = cv2.dilate(mask, kernel2, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3)
                _, mask = cv2.threshold(mask, 130, 255, cv2.THRESH_BINARY)
                # Lấy vẽ đường viền
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                list = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 4000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        list.append([x, y, w, h])
                # tracking
                bbox_idx = self.tracker.update(list)
                for bbox in bbox_idx:
                    x, y, w, h, id = bbox
                    center_points = (x + w/2, y + h/2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    results = cv2.pointPolygonTest(self.area_Cnt,center_points,False)
                    if results>=0:
                        self.arr.add(id)
                    self.cnt_obj.set(len(self.arr))

                cv2.polylines(frame,[np.array(self.area_Cnt,np.int32)],True,(0,0,255),2)
                mask = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mask = Image.fromarray(mask)
                mask = ImageTk.PhotoImage(image=mask)
                self.label.configure(image=mask)
                self.label.image = mask
        self.label.after(3, self.show_frames)
    
    def show_frames_yolo(self):
        if self.paused == False:
            ret, frame1 = self.cap.read()
            if self.cap is None:
                messagebox.showwarning("Warning", "Please select a video file first.")
                return
            self.cntframe += 1
            if self.cntframe % 5 ==0:
                frame1 = cv2.resize(frame1,(480, 270))
                results=self.model.predict(frame1)
                a=results[0].boxes.boxes
                px=pd.DataFrame(a).astype("float")
                list=[]
                for index,row in px.iterrows():
                    x1=int(row[0])
                    y1=int(row[1])
                    x2=int(row[2])
                    y2=int(row[3])
                    d=int(row[5])
                    list.append([x1,y1,x2,y2])
                bbox_idx = self.tracker.update(list)
                for bbox in bbox_idx:
                    x3,y3,x4,y4,id=bbox
                    cx=int(x3+x4)//2
                    cy=int(y3+y4)//2
                    #cv2.circle(frame1,(cx,cy),5,(255,0,255),-1)
                    cv2.rectangle(frame1,(x3,y3),(x4,y4),(0,255,0),2)
                    #cv2.putText(frame1,str(int(id)),(cx,cy),cv2.FONT_HERSHEY_TRIPLEX,1.5,(255, 0, 0),1)
                    results = cv2.pointPolygonTest(self.area_Cnt,((cx,cy)),False)
                    if results>=0:
                        self.arr.add(id)
                    self.cnt_obj.set(len(self.arr))

                cv2.polylines(frame1,[np.array(self.area_Cnt,np.int32)],True,(0,0,255),2)
                mask1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                mask1 = Image.fromarray(mask1)
                mask1 = ImageTk.PhotoImage(image=mask1)
                self.label.configure(image=mask1)
                self.label.image = mask1
        self.label.after(1, self.show_frames_yolo)

    def pause_video(self):
        self.paused = True
        
    def play_video(self):
        self.paused = False

if __name__ == "__main__":
    app = Window()
    app.mainloop()
