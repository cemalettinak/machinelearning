# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:27:30 2023

@author: cemalettin
"""




# Categorical conversion for XGBoost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['AdmissionType'] = le.fit_transform(data['AdmissionType'])
