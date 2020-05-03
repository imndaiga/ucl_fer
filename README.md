The entire project report can be read [here](https://github.com/wndaiga/ucl_fer/blob/master/report.pdf).

## Background
With  the  rapid  growth  of  media  and  entertainment consumption in recent years, online streaming services and content makers are now seeking novel methods toindex and categorise their multimedia content to im-prove  user  engagement.
Because  the  affective  states of  a  user  can  influence  greatly  their  choices  of  mul-timedia  content  and  reflect  their  preferences  for  thecontent (Soleymani et al., 2008), users’ emotional re-actions could be used to label multimedia content and help  create  customised  content  recommendation  sys-tems through a better understanding of users’ prefer-ences.
Emotions  can  be  categorised  using  a  number  of  different methods.  Both discrete categorisation, the six basic emotions by (P. Ekman & Ricci-Bitti, 1987), and continuous categorisation, the valence-arousal scale by (Russell, 1980), are used in this project.
Results from both categorisations could convey useful information, as  multimedia  commonly  elicits  one  of  the  six  emo-tions, and valence-arousal levels indicate value and importance (Clore & E Palmer, 2009).

## Purpose of Project
The purpose of this project is to create an automatic emotion  recognition  system  capable  of  detecting  different  affective  states  of  users  while  they  are  watching movies or TV shows.
This particular system implemented in the project aims to detect seven different emotions (Sad, Disgust, Anger, Neutral, Surprise, Fear and Happy) from facial expressions, and classify high/low valence and arousal levels from physiological signs, using machine learning models.
To achieve this, we carried out experiments where participants were shown a series of videos chosen to elicit specific emotions, and recorded their facial expressions and physiological signs.  The collected data, combined with two larger public data sets, are then used to train a deep learning facial emotion recognition model and two binary physiological data classifiers for high/low valence and arousal.

## Conclusion
In  this  work,  we  used  video  stimuli  to  elicit  certainemotions in four participants.
Each subject watched and rated their affective response to seven videos, with both  discrete  and  continuous  labelling  adopted.
Facial expressions were captured using frontal face video, while  an  Empatica  bracelet  measured  electrodermal and heart activity.
We went on to train a deep neural network using facial images and discrete labels (happy, sad,  fear,  disgust,  surprise,  anger,  neutral),  and  also presented classifiers which linked physiological signals to ratings of valence and arousal.
In both cases, results were shown to be significantly better than random classification. The final output of this project was a deep neural network trained to recognise emotions in consumers of online media using webcams and physiological sensors.
