# Emotion Classification 

##  Personal Objectives
I used this project to learn the basics of converting a Keras model trained in Python to a JavaScript model

## Project Goal
Implement an emotion-classification model directly in browser to detect a person's current emotion via the webcam, and display the detected emotion to the user

##  Technologies Used
- [TensorFlowJS](https://github.com/tensorflow/tfjs)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [Bulma CSS](https://bulma.io/)
## Description
This project is implemented purely in JavaScript using TensorFlowJS. The facial-recognition model is ported from [blazeface](https://github.com/tensorflow/tfjs-models/tree/master/blazeface) and the emotion-detection model was trained in Python on the [fer-2013](https://ieeexplore.ieee.org/abstract/document/9288560) dataset


## Sample Output

Angry             |  Surprise
:-------------------------:|:-------------------------:
<img src="samples/angry.png" alt="drawing" width="400"/>  |  <img src="samples/surprise.png" alt="drawing" width="400"/>

## TODO
- [x] Extract bounding box for face from output
- [x] Port classification code to looping function
- [x] Center webcam stream
- [x] Show percentage emotion using output tensor
- [x] Incorporate Bulma
- [x] Transition between colour changes 
- [ ] Push to heroku