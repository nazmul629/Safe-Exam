import  cv2
import mediapipe as mp


#  Face Detection Resource
from simple_facerec import SimpleFacerec
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")





#  Eyes Tracking
import math
import  numpy as np
import  mediapipe as mp




LEFT_EYES = [362, 382, 381, 380, 374 ,373,390,249,263,466,388,387,386,385,384, 398 ]
RIGHT_EYES = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

RIGHT_IRIS= [474,475,476,477]
LEFT_IRIS = [469,470,471,472]

L_H_LEFT =  [33];
L_H_RIGHT = [133];
L_TOP =[159]
L_DOWN =[145]


R_H_LEFT = [362]
R_H_RIGHT = [263]
def euclidean_distance (point1,point2):
    x1,y1 = point1.ravel()
    x2,y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def iris_position(iris_center,right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center,right_point)
    total_distance = euclidean_distance(right_point,left_point)
    ratio = center_to_right_dist/total_distance

    iris_possition = ""

    if ratio<= 0.42:
        iris_possition = "Left"
    # elif ratio> 0.42 and ratio <= 0.57:
    elif ratio> 0.42 and ratio <= 0.60:
        iris_possition = "Center"
    else:
        iris_possition = "Right "
    return  iris_possition, ratio




def iris_up_down_position(iris_center,right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center,right_point)
    total_distance = euclidean_distance(right_point,left_point)
    ratio = center_to_right_dist/total_distance

    iris_possition = ""

    if ratio>= 55:
        iris_possition = "Top"
    # elif ratio> 0.42 and ratio <= 0.57:
    elif ratio> 0.42 and ratio <= 0.60:
        iris_possition = "Center"
    else:
        iris_possition = "Right "
    return  iris_possition, ratio


mp_face_mesh = mp.solutions.face_mesh;


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

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=.05
) as face_mesh:


    while True:
        _, frame = cam.read()
        # Mirror Screen

        frame = cv2.flip(frame, 1)





        # for Eyes Tracking
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(frame);


        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            if name != "Unknown":
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                # For Eyes Tracking

                if results.multi_face_landmarks:
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
                                                        results.multi_face_landmarks[0].landmark])
                    # print(mesh_points)
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)

                    cv2.circle(frame, center_left, 2, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, center_right, 2, (255, 255, 255), -1, cv2.LINE_AA)

                    cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

                    # Showing Right eyes  4 Point
                    cv2.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)

                    cv2.circle(frame, mesh_points[L_TOP][0], 2, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[L_DOWN][0], 2, (0, 255, 255), -1, cv2.LINE_AA)

                    iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT],mesh_points[R_H_LEFT][0])
                    iris_up_down_pos, u_d_ratio = iris_up_down_position(center_left, mesh_points[L_TOP],
                                                                                    mesh_points[L_DOWN][0])

                    print(u_d_ratio)
                    if iris_pos == "Center":
                        cv2.putText(frame, f"Iris Positon: {iris_pos} {ratio:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1, cv2.LINE_AA)
                    else:
                         cv2.putText(frame, f"Iris Positon: {iris_pos} {ratio:.2f}", (30, 30),
                         cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)



            else:
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)




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

            #
            # if class_name == "person":
            #     face_locations, face_names = sfr.detect_known_faces(frame)
            #     for face_loc, name in zip(face_locations, face_names):
            #         y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            #         if name != "Unknown":
            #             cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            #
            #             # For Eyes Tracking
            #
            #             if results.multi_face_landmarks:
            #                 mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in
            #                                         results.multi_face_landmarks[0].landmark])
            #                 # print(mesh_points)
            #                 (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            #                 (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            #
            #                 center_left = np.array([l_cx, l_cy], dtype=np.int32)
            #                 center_right = np.array([r_cx, r_cy], dtype=np.int32)
            #
            #                 cv2.circle(frame, center_left, 2, (255, 255, 255), -1, cv2.LINE_AA)
            #                 cv2.circle(frame, center_right, 2, (255, 255, 255), -1, cv2.LINE_AA)
            #
            #                 cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            #                 cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)
            #
            #                 # Showing Right eyes  4 Point
            #                 cv2.circle(frame, mesh_points[R_H_RIGHT][0], 3, (255, 255, 255), -1, cv2.LINE_AA)
            #                 cv2.circle(frame, mesh_points[R_H_LEFT][0], 3, (0, 255, 255), -1, cv2.LINE_AA)
            #
            #                 cv2.circle(frame, mesh_points[L_TOP][0], 2, (255, 255, 255), -1, cv2.LINE_AA)
            #                 cv2.circle(frame, mesh_points[L_DOWN][0], 2, (0, 255, 255), -1, cv2.LINE_AA)
            #
            #                 iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT],
            #                                                 mesh_points[R_H_LEFT][0])
            #                 iris_up_down_pos, u_d_ratio = iris_up_down_position(center_left, mesh_points[L_TOP],
            #                                                                     mesh_points[L_DOWN][0])
            #
            #                 print(u_d_ratio)
            #                 if iris_pos == "Center":
            #                     cv2.putText(frame, f"Iris Positon: {iris_pos} {ratio:.2f}", (30, 30),
            #                                 cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 0), 1, cv2.LINE_AA)
            #                 else:
            #                     cv2.putText(frame, f"Iris Positon: {iris_pos} {ratio:.2f}", (30, 30),
            #                                 cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)
            #
            #
            #
            #
            #
            #         else:
            #             cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            #
            #

            # else:
            if class_name != "person":
                cv2.putText(frame,class_name,(x,y-5),cv2.FONT_HERSHEY_PLAIN, 2,(0,255,255),2 )
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)


        cv2.imshow("Object Detaction", frame);
        key = cv2.waitKey(1);

        if key == ord('s'):
            break;

