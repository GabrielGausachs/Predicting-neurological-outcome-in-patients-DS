# Predicting-neurological-outcome-in-patients-DS
Predicting neurological outcome in patients with a severe postanoxic encephalopathy [EEG] - Final Project for the Data Science course

## Introduction
Each year, about 7000 patients with a postanoxic coma after a cardiac arrest are admitted to the Intensive Care Unit. Early prediction of neurological outcome is highly relevant, not only for the treating physicians, but also for family members. This can prevent futile treatment, but will also assist in providing care for those with a high probability of good recovery.

Early recording of the electroencephalogram (EEG) allows reliable prediction of both poor and good outcome in a significant percentage of patients (about 50-60%). While these recordings are typically assessed by visual analysis, machine learning may assist or even outperform human
visual assessment.

### Goals
Our goal is to build a machine learning model to accurately predict whether a patient with postanoxic encephalopathy will have a "good" (CPC 1-2) or "poor" (CPC 3-5) neurological outcome based on EEG features.
In addition, for predicting "poor" outcome, the model must have 100% specificity (no false positives). This is crucial to avoid falsely labeling a patient as having a poor outcome. On the other hand, for predicting "good" outcome, the model must have at least 95% specificity.
We also evaluate model performance using ROC curves and provide confidence intervals to assess the reliability of the results.

### Dataset 
We use a dataset with various quantitative EEG features and the neurological outcome, the Cerebral Performance Category Score (CPC). The data is contained in two excel files, featuresNEW 12hrs.xls (corresponding to 12 hours after CA) and
featuresNEa 24hrs.xls (corresponding to 24 hours after CA) from several patients. Some patients can have both 12hrs and 24hrs EEG.
