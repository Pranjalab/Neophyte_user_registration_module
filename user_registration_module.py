# USAGE
# python user_registration_module.py --user_name Abhinav

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import shutil


## defined functions
#########################################################################
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


#########################################################################
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
embedder = cv2.dnn.readNetFromTorch("embedding_gen_model/openface.nn4.small2.v1.t7")


conf_thr = 0.85
no_samples_per_user = 40
#########################################################################

ap = argparse.ArgumentParser()
ap.add_argument("-user", "--user_name", required=True,
	help="provide the name of the user to be registered")
args = vars(ap.parse_args())

## Paths for saving the unique images of the user
user_folder_blobs = os.path.sep.join(["DATA", args["user_name"], "blobs"])
user_folder_frames = os.path.sep.join(["DATA", args["user_name"], "frames"])


## cleaning up the folders before the registration
if os.path.exists(user_folder_blobs):
    for f in os.listdir(user_folder_blobs):
        os.remove(os.path.join(user_folder_blobs, f))

if os.path.exists(user_folder_frames):
    for f in os.listdir(user_folder_frames):
        os.remove(os.path.join(user_folder_frames, f))

#########################################################################

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

count = 0
FEmbd = []
EMBEDDINGS = []
# loop over frames from the video file stream
while True:
    # count = count + 1
    rval, frame = vs.read()
    orig_frame = frame
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # applying face detection model (opencv) : TODO: Change to better detection model, e.g. mtcnn, or any else
    detector.setInput(imageBlob)
    detections = detector.forward()
    

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > conf_thr:
            
            ## creating the directories for registering the samples of the user
            if not os.path.exists(user_folder_blobs):
                os.makedirs(user_folder_blobs)

            if not os.path.exists(user_folder_frames):
                os.makedirs(user_folder_frames)
            
            

            # compute the (x, y) coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            orig_face = face.copy()
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()


            # draw the bounding box of the face
            text = "Your face got detected"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    EMBEDDINGS.append(vec)

    ## registering the first sample of the user
    if len([f for f in os.listdir(user_folder_blobs) if not f.startswith('.')]) == 0:
        faceBlob_name = user_folder_blobs + "/" +  str(count) + ".jpg"
        faceFrame_name = user_folder_frames + "/" +  str(count) + ".jpg"
        # writing cropped faces
        cv2.imwrite(faceBlob_name, orig_face)
        # writing full frame
        cv2.imwrite(faceFrame_name, orig_frame )
        FEmbd.append(vec)
        print("****** first sample is registered ********")
        

    no_of_samples = len([f for f in os.listdir(user_folder_blobs) if not f.startswith('.')])
    if len([f for f in os.listdir(user_folder_blobs) if not f.startswith('.')]) > no_of_samples:
        count = count + 1


    ## defining the stopping criteria
    if [f for f in os.listdir(user_folder_blobs) if not f.startswith('.')] != 0:
        if len([f for f in os.listdir(user_folder_blobs) if not f.startswith('.')]) > no_samples_per_user:
            print("USER REGISTRATION COMPLETED: SAMPLES COLLEDTEC = ", no_of_samples)
            break

        
        if len(EMBEDDINGS)> 2:
            vec1 = EMBEDDINGS[-2]
        else:
            vec1 = EMBEDDINGS[-1]

        vec0 = FEmbd[-1]

        ## computing distance between the first sample and the immediate previous sample
        embedding_distance = findEuclideanDistance(vec1, vec)
        fisrt_embedding_distance = findEuclideanDistance(vec0, vec)

        embedding_distance = np.float64(embedding_distance)
        fisrt_embedding_distance = np.float64(fisrt_embedding_distance)

        ## setting up the thresholds for selecting unique faces
        if embedding_distance > 0.7:
            if fisrt_embedding_distance > 0.7:
                faceBlob_name = user_folder_blobs + "/" +  str(count) + ".jpg"
                faceFrame_name = user_folder_frames + "/" +  str(count) + ".jpg"
                cv2.imwrite(faceBlob_name, orig_face)
                cv2.imwrite(faceFrame_name, orig_frame)
                print("Registered sample no: ", no_of_samples)
                count = count+1


    fps.update()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
