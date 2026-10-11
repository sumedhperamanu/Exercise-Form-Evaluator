# Machine Learning Exercise Form Evaluator

Program designed to track exercise movement, rep counts, and quality of form. Utilizes MediaPipe and OpenCV for video tracking.

Tracks user movement according to MediaPipe joint landmarks, and finds whether the angles between the joints of the user are aligned such that the exercise was completed with optimal form.

Tracks user state, particularly in squats (descending, ascending, standing) to circumvent camera/user noise issues as either moves around.

Landmark coordinates are streamed into a csv data file for every rep performed, which allows the neural network to train itself on readily accessible data.

Two collection data Python pipelines were made so that the developer does not have to sift through each squat to separate good from bad. The good ones will automatically be filtered in collect_good_data.py (bad squats performed in collect_good_data.py will not even be recorded as squat attempts) and the bad form ones will be filtered automatically into collect_shallow_data.py.

The data is then passed into a FNN (feedforward neural network)/MLP (multilayer perceptron) to train the model on good and bad form squats. A prediction model will then be created such that it can identify the problem with the user's squats depending on the form it most closely aligns with (good, shallow, or knee_valgus).

