# Machine Learning Exercise Form Evaluator

Program designed to track exercise movement, rep counts, and quality of form. Utilizes MediaPipe and OpenCV for video tracking.

Tracks user movement according to MediaPipe joint landmarks, and finds whether the angles between the joints of the user are aligned such that the exercise was completed with optimal form.

Tracks user state, particularly in squats (descending, ascending, standing) to circumvent camera/user noise issues as either moves around.

Landmark coordinates are streamed into a csv data file for every rep performed, which allows the neural network to train itself on readily accessible data.

Two collection data Python pipelines were made so that the developer does not have to sift through each squat to separate good from bad. The good ones will automatically be filtered in collect_good_data.py (bad squats performed in collect_good_data.py will not even be recorded as squat attempts) and the bad form ones will be filtered automatically into collect_shallow_data.py.

The data is then passed into a FNN (feedforward neural network)/MLP (multilayer perceptron) to train the model on good and bad form squats. A prediction model will then be created such that it can identify the problem with the user's squats depending on the form it most closely aligns with (good, shallow, or knee_valgus).

## Setup
- Install the Python packages the live squat coach imports:
  ```
  pip install mediapipe opencv-python requests python-dotenv
  ```
  plus whatever `predict_model.py` needs to load the trained model.
- The live program imports two local modules: `camera.py` (`frameCapture()` supplies the camera frames) and `predict_model.py` (`predict_rep()` classifies a finished rep).
- **Optional: ThingSpeak uploads.** Create a `.env` file next to the program containing your channel's Write API Key (no spaces or quotes around the `=`):
  ```
  THINGSPEAK_API_KEY=your_write_key
  ```
  Don't commit `.env` (add it to `.gitignore`). Without the key, the program still runs, prints a warning, and uploads are disabled.

## Running the live squat coach
1. Start the program and stand in view of the camera. Keep your whole body in frame, since the full set of landmarks is passed to the model.
2. **Calibration:** the program waits 5 seconds, then collects 30 frames in which your shoulders and hips are clearly visible. "Stand still..." is shown until it finishes. This records your standing hip height and torso length.
3. Perform squats. After each rep, the screen shows the rep count and feedback, and the result is printed to the terminal.
4. Controls:
   - `q` quits.
   - `r` recalibrates and **resets all rep counts**.

## How reps are detected
Rep detection uses the height of the hips (the average of the left and right hip landmarks) compared with the standing height from calibration. The thresholds are fractions of the image height:

| State change | Condition | Setting |
|---|---|---|
| standing → descending | hips drop more than 0.02 below standing height | `start_thresh` |
| descending → ascending | hips rise 0.01 above their lowest point | `reversal` |
| ascending → standing | hips return to within 0.012 of standing height | `end_thresh` |

While descending and ascending, every frame's landmarks (x, y, z, and visibility for all 33 landmarks) are recorded. When the rep ends, the recorded frames are passed to `predict_rep()`. Reps with fewer than 5 frames are rejected as too short and are not counted.

## Feedback
| Model label | Message shown |
|---|---|
| `good` | Good rep |
| `knee_valgus` | Don't let your knees cave |
| `shallow` | Go deeper |
| `shallow + knee_valgus` | Go deeper and don't let your knees cave |

Good reps are shown in green and the others in red. A rep that is too short shows "Rep too short; extend your reps" in orange.

## Depth measurement
For each counted rep, the program calculates **relative depth**: the lowest hip drop of the rep divided by torso length (shoulder-to-hip distance, measured during calibration). Dividing by torso length makes the value less dependent on the camera distance, but it is still best compared only within the same camera setup. The program tracks the average over all reps and the most recent value.

## ThingSpeak data upload
When uploads are enabled, the program sends a snapshot of the session to ThingSpeak every **20 seconds**, starting once at least one rep has been counted. The upload runs in a background thread, so the video does not freeze. The free ThingSpeak tier accepts at most one update every 15 seconds.

| Field | Contents |
|---|---|
| 1 | Total reps counted (good or bad) |
| 2 | Good reps |
| 3 | Shallow reps (label `shallow` only) |
| 4 | Knee valgus reps (label `knee_valgus` only) |
| 5 | Shallow + knee valgus reps |
| 6 | Fraction of good reps (good ÷ total, 0 to 1) |
| 7 | Average relative depth |
| 8 | Relative depth of the last rep |

- The combined label is counted only in field 5, so total shallow reps = field 3 + field 5, and total knee valgus reps = field 4 + field 5.
- The values are running totals for the session, and they reset to zero when you press `r`.
- Enable all 8 fields in the channel's settings and name them, or the data won't appear.
- Use a **separate channel** for this program. ThingSpeak's rate limit applies to the whole channel, so another device writing to the same one can cause rejected updates.
- Reps completed in the last 20 seconds before you quit may not be uploaded.

## Notes and limitations
- Positions are measured as fractions of the image height, so keep the camera in a fixed position, and recalibrate with `r` if you move or the camera moves.
- A rep only ends when the hips return to within 0.012 of the calibrated standing height. If your standing posture changes after calibration, a rep may not finish, and recalibrating fixes it.
- Feedback comes after each rep is completed, not during it.
- The quality of the feedback depends on the data the model was trained on.