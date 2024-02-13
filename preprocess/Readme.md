# Preprocess Script

## 1.Data Preprocess

### Voice
- Extract .mp4 to wav: `preprocess/preprocess/1_mp4_extract_wav.py`
- Perform Voice Activity Detection: `preprocess/preprocess/2_wav_vad.py`

### Face
- Extract mp4 to frames: `preprocess/preprocess/3_mp4_extract_frames.py`
- MTCNN: `preprocess/preprocess/4_face_crop_mtcnn.py`
- Pose Estimation: `preprocess/preprocess/5_pose_estimation.py`

## 2.Feature Extract

### Voice
- ECAPA-TDNN: `preprocess/voice_extractor/1_ecapa_tdnn.py`
- Resemblyzer: `preprocess/voice_extractor/2_resemblizer.py`

### Face
- Inception-v1: `preprocess/face_extractor/1_inception_v1.py`

- Deepface: `preprocess/face_extractor/2_deepface.py`
  - Support: "VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace"
  
- Face-X-Zoo: `preprocess/face_extractor/3_facexzoo.py`

  Example command for running feature extraction:

  - MobileFaceNet: `python script_name.py --backbone_type MobileFaceNet --model_pkl 1_MobileFaceNet --batch_size 2048 --size 112 --save_name MobileFaceNet.pkl`
  - ResNet: `python script_name.py --backbone_type ResNet --model_pkl 5_Resnet152-irse --batch_size 512 --size 112 --save_name ResNet.pkl`
  - AttentionNet: `python script_name.py --backbone_type AttentionNet --model_pkl 2_Attention56 --batch_size 512 --size 112 --save_name AttentionNet.pkl`

## 
