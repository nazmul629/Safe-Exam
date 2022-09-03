import math
import cv2
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

    if ratio<= 0.50:
        iris_possition = "Left  "
    # elif ratio> 0.42 and ratio <= 0.57:
    elif ratio> 0.42 and ratio <= 0.60:
        iris_possition = "Center"
    else:
        iris_possition = "Right "
    return  iris_possition, ratio




mp_face_mesh = mp.solutions.face_mesh;

cam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=.05
) as face_mesh:
    while True:
        ret, frame = cam.read()
        if not ret:
            break;

        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2RGB);
        img_h, img_w = frame.shape[:2]

        # print("image_h ", image_h, "\nimage_w", image_w)

        results = face_mesh.process(rgb_frame);

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y],  [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points)

            #     Print the Whole Eyes
            # cv2.polylines(frame,[mesh_points[LEFT_EYES]], True, (255,255,0),1, cv2.LINE_AA)
            # cv2.polylines(frame,[mesh_points[RIGHT_EYES]], True, (255,255,0),1, cv2.LINE_AA)

            (l_cx,l_cy),l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx,r_cy),r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            center_left = np.array([l_cx,l_cy], dtype=np.int32)
            center_right = np.array([r_cx,r_cy], dtype=np.int32)

            cv2.circle(frame,center_left,2,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(frame,center_right,2,(255,255,255),-1,cv2.LINE_AA)


            cv2.circle(frame,center_left,int(l_radius),(255,0,255),1,cv2.LINE_AA)
            cv2.circle(frame,center_right,int(r_radius),(255,0,255),1,cv2.LINE_AA)

            # Showing Right eyes  4 Point
            cv2.circle(frame,mesh_points[R_H_RIGHT][0],3,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(frame,mesh_points[R_H_LEFT][0],3,(0,255,255),-1,cv2.LINE_AA)

            # cv2.circle(frame,mesh_points[R_TOP][0],2,(255,255,255),-1,cv2.LINE_AA)
            # cv2.circle(frame,mesh_points[R_DOWN][0],2,(0,255,255),-1,cv2.LINE_AA)



            iris_pos , ratio = iris_position(center_right,mesh_points[R_H_RIGHT] ,mesh_points[R_H_LEFT][0])

            # print(iris_pos)
            if  iris_pos== "Center":
                cv2.putText(frame, f"Iris Positon: {iris_pos}",(30,30),cv2.FONT_HERSHEY_PLAIN,1.2,(0,255,0),1,cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Iris Positon: {iris_pos}",(30,30),cv2.FONT_HERSHEY_PLAIN,1.2,(0,0,255),1,cv2.LINE_AA)


            # print("Left",l_cx,l_cy, l_radius)
            # print("Right",r_cx,r_cy, r_radius)



        cv2.imshow("img",frame)


        key = cv2.waitKey(1)
        if key ==  ord('s'):
            break;

cam.release()
cv2.destroyAllWindows()