import cv2
import time
import numpy as np
import datetime



webcam = cv2.VideoCapture(0)

Frame1 = None
first_frame_face_check = None
Face_ROI = None
face_pic_check = None
Skipped = False

face_checker = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cy = 0
cw = 0
ch = 0
count = 0
diff = 0


Detected = False
wait_time = 3
timer_started = False
started = None

Recording_Count = 1

Frame_size = (int(webcam.get(3)), int(webcam.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
Blur_count = None


while True:
    cam, frame = webcam.read()
    #Motion detector
    #Making frame grey
    Greyed_ver = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if Frame1 is None:
        Frame1 = Greyed_ver
        continue
    #Applying the blur
    Gblur = cv2.GaussianBlur(Greyed_ver, (21, 21), 0)
    #Getting absolute difference
    builtdiff = cv2.absdiff(Frame1, Gblur)
    Threshold_value = cv2.threshold(builtdiff, 40, 255, cv2.THRESH_BINARY)[1]
    #changed from 2 to 3 iterations
    Threshold_Frame = cv2.dilate(Threshold_value, None, iterations = 3)
    #Finding countours
    (countourFind,_) = cv2.findContours(Threshold_Frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for countour in countourFind:
        #if countour less than amount restart
        if cv2.contourArea(countour) < 6000:
            continue
        #make the rectangle and the text
        (x, y, width, height) = cv2.boundingRect(countour)
        rect = cv2.rectangle(frame, (x,y), (x+width, y+height), (255, 0, 0), 3)
        cv2.putText(rect, "Motion", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (366, 255, 12), 2)





    #Face detector
    faces_found = face_checker.detectMultiScale(Greyed_ver, scaleFactor = 1.3, minNeighbors = 5)
    for (x, y, width, height) in faces_found:
        if count == 0 or count % 2 == 0:
            cx = x
            cy = y
            cw = width
            ch = height
            count = count + 1
        diff = cx - x + cy - y + cw - width + ch - height

        if first_frame_face_check == None or face_pic_check == None:
            Face_ROI = frame[y:y+height, x:x+width]
            Blur_count = cv2.Laplacian(Face_ROI, cv2.CV_64F).var()
            if Blur_count < 110:
                print("Skipped")
                print(Blur_count)
                face_pic_check = None
                Skipped = True
            else:
                cv2.imwrite(f"Face in encounter {Recording_Count}.jpg", Face_ROI)
                print("usuable")
                first_frame_face_check = True
                face_pic_check = True
                Skipped = False
        if diff > -50 and diff < 50:
            Face_rect = cv2.rectangle(frame, (x,y),(x+width, y+height),(0,0,255),3)
            cv2.putText(Face_rect, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            Face_rect_change = cv2.rectangle(frame, (x,y),(x+width, y+height),(0,0,255),3)
            cv2.putText(Face_rect_change, "Face Motion Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    if len(faces_found) == 0 and Skipped == True:
        cv2.imwrite(f"(May be blurry) Face in encounter {Recording_Count}.jpg", Face_ROI)
        print("unusable")



#look at notebook for recording logic
    if (len(countourFind) > 0 and len(faces_found) > 0) or len(faces_found) > 0:
        if Detected == True:
            timer_started = False
        else:
            Detected = True
            current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            output = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, Frame_size)
            print("Started Recording")
            first_frame_face_check = True
    elif Detected == True:
        if timer_started == True:
            if time.time() - started >= wait_time:
                Detected = False
                timer_started = False
                output.release()
                print("Stopped Recording")
                first_frame_face_check = None
                Recording_Count = Recording_Count + 1
        else:
            timer_started = True
            started = time.time()

    if Detected:
        output.write(frame)


    cv2.imshow("Capture", Threshold_Frame)
    cv2.imshow("motion detector", frame)
    space_key = cv2.waitKey(1)
    if space_key == ord("q"):
        break

    if space_key == ord("r"):
        print("restarting")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        Frame1 = None

output.release()
webcam.release()
cv2.destroyAllWindows()
