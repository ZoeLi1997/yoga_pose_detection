#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import os
from random import randint
from os.path import isfile, join
from keras.models import load_model
import A2_Functions as a2


# In[13]:


model = load_model("yoga_pals_model.h5")
save_file = "output.jpg"
read_file = "fromclient.jpg"
pose_file = "predicted_pose.txt"


# In[14]:


detected_keypoints, frameClone, image = a2.perform_keypoint_analysis(read_file)
missing = []
found = []
i = 0
for el in detected_keypoints:
    if el == []:
        missing.append(i)
    else:
        found.append(i)
    i += 1
if a2.apply_filter(missing) is True:
    results = a2.scale_keypoints(read_file)
    predict = model.predict(np.array([results]))
    predicted_pose, confidence = a2.predict_pose(predict)
    with open(pose_file, "w") as fp:
        fp.write("Our model predicted that your pose is: " + predicted_pose + (" with %.2f" % (confidence * 100)) + "% confidence.")
#    print("Our model predicted that your pose is: " + predicted_pose + (" with %.2f" % (confidence * 100)) + "% confidence.")
else:
    with open(pose_file, "w") as fp:
        fp.write("Error: Photo did not pass filter.")
    print("Error: Photo did not pass filter.")


# In[ ]:





# In[ ]:




