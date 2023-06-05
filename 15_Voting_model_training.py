# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:27:30 2023

@author: cemalettin
"""


# Combining multiple models with voting classifier
voting_model = VotingClassifier(estimators=[('Random Forest', rf_model),('Decision Tree', dt_model), ('Gradient Boosting', gb_model)], voting='soft')
voting_model.fit(X_train, y_train)

