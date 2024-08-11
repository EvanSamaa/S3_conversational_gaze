# S3_conversational_gaze
This repo is for the paper [S3: Speech, Script and Scene driven Head and Eye Animation](https://dl.acm.org/doi/10.1145/3658172) presented at Siggraph 2024. 

The code will be released soon (by Oct 2024). 

This Repo will contain:
1. data processing pipeline for generating training data
2. link to dataset
3. training code for gaze aversion/no-aversion classification model
4. code for generating head rotation angles (yaw, pitch, roll) and gaze points (x,y,z positions in local space) from speech audio, scene and script

This Repo will not contain:
1. the rig used for generating the supplementary videos
2. model for generating lip-sync animation and paralingual (we used the automatic solution from [JALI Research](https://jaliresearch.com/))
