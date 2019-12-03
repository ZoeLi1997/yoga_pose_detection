#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import os
from random import randint
from os.path import isfile, join
from keras.models import load_model
import A2_Functions as a2


# In[5]:


model = load_model("../yoga_pals_model.h5")
save_file = "output.png"
read_file = "fromclient.jpg"
pose_file = "predicted_pose.txt"


# In[7]:


a2.perform_keypoint_analysis(read_file)
results = a2.scale_keypoints(read_file)
predicted_pose = a2.predict_pose(model.predict(np.array([results])))
with open(pose_file, "w") as fp:
    fp.write(predicted_pose)
#print("Our model predicted that your pose is: " + predicted_pose)


# In[ ]:




