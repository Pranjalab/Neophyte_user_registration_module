Steps to run the code:

1. Create a environment by running the following command:
    pip install -r requirements.txt

2. Once the environment is created, activate the env by running the following command
    conda activate face_register

3. Once the env is activated, finally run the face registration code using the following command:
    user_registration_module.py --user_name name

4. The code will display webcam UI with detected faces in the live stream. You need to do the following activities:
    a. keep the face frontal in the beginning
    b. rotate your face slowly toward RIGHT (at different angles)
    c. rotate your face slowly toward LEFT (at different angles)
    d. rotate your face slowly toward UP (at different angles)
    e. rotate your face slowly toward DOWN (at different angles)
    f. try different poses, expressions, occlusions, etc for better face samples


5. The face samples registered can be found at  
    DATA/name/blobs >> stores the detected and cropped faces
    DATA/name/frames >> stores original frames for better quality (only after detection is successful)

6. Once the data is stored in the DATA location, please upload it on the given drive location: [drive directory](https://drive.google.com/drive/folders/1CoppjFQT6d1Lhvnu1XkKsmjy6HOz38Ab?usp=sharing)