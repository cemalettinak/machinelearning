
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:07:47 2023

@author: cemalettin
"""


# Categorical conversion for XGBoost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['AdmissionType'] = le.fit_transform(data['AdmissionType'])
