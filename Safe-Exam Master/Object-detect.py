import  cv2
import mediapipe as mp

#  Face Detection Resource
from simple_facerec import SimpleFacerec






sfr = SimpleFacerec()
sfr.load_encoding_images("images/")




# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255,)


# Load class lists
classes = []


with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# print("Objects list")
# print(classes)



# initialize Camera
cam = cv2.VideoCapture(0);
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)



while True:
    _, frame = cam.read()
    # Mirror Screen

    frame = cv2.flip(frame, 1)

    # RGB Color
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecting Eyes
    output = face_mesh.process(frame)
    (class_ids, score, bboxes) = model.detect(frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # cv2.circle(frame, (x, y), 3, (0, 255, 0))
            # print(x,y)

    (class_ids, scores, bboxes) = model.detect(frame,)
    for class_id,score,bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h) = bbox;

        class_name = classes[class_id]


        if class_name == "person":
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                if name != "Unknown":
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)



                else:
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)





        else:
            cv2.putText(frame,class_name,(x,y-5),cv2.FONT_HERSHEY_PLAIN, 2,(0,255,100),2 )
            cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,0),3)
        # print(x,y,w,h)

    # print("class ids",class_ids)
    # print("Scors",score)
    # print("Bbox",bboxes)




    cv2.imshow("Object Detaction", frame);
    key = cv2.waitKey(1);

    if key == ord('s'):
        break;

