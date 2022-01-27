Steps to run the code:

1. Create a conda environment by running the following command:
    conda env create -f face_register.yml

2. Once the enviroenment is created, activate the env by running the following command
    conda activate face_register

3. Once the env is activated, finally run the face registration code using the following command:
    user_registration_module.py --user_name name

4. The code will display webcam UI with detected face in live stream. You need to do the following activities:
    a. keep the face frontal in the begining
    b. rotate your face slowly toward RIGHT (at different angles)
    c. rotate your face slowly toward LEFT (at different angles)
    d. rotate your face slowly toward UP (at different angles)
    e. rotate your face slowly toward DOWN (at different angles)
    f. try different poses, expressions, occlusions, etc for better face samples


5. The face samples registered can be found at  
    DATA/name/blobs >> stores the detected and cropped faces
    DATA/name/frames >> stores original frames for better quality (only after detection is successfull)