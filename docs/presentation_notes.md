# Viva Presentation Notes

## One-minute introduction

This project is a real-time American Sign Language fingerspelling system. It captures a hand from a webcam, detects the hand and its 21 landmarks using CVZone and MediaPipe, draws a normalized skeleton on a white 400 by 400 canvas, and sends that image to a trained Keras CNN. The model first predicts one of eight visually similar groups. Landmark geometry then separates letters inside the group. The recognized letters are accumulated into a sentence and can be spoken using pyttsx3.

## Presentation order

1. Problem: communication is difficult when the other person does not understand sign language.
2. Objective: convert static alphabet gestures into text and speech using an ordinary webcam.
3. Input: live webcam frames; no special gloves or sensors.
4. Preprocessing: mirror frame, detect one hand, crop the bounding box, extract 21 landmarks, draw lines and points on a white canvas.
5. Classification: CNN predicts eight groups; geometric rules use landmark coordinates and distances to distinguish similar letters.
6. Application: Tkinter shows the camera, skeleton, current character, sentence, suggestions, Speak, and Clear controls.
7. Output: text sentence and optional speech.
8. Evidence: bundled model, 26 alphabet folders, 4,681 dataset images, and the supplied report/poster/presentation.
9. Limitations and future work.

## Demo script

1. Activate the virtual environment and run `python src\\sign_language_gui.py`.
2. Show the live camera and place one hand inside the frame.
3. Hold a clear static letter until the current-character display updates.
4. Demonstrate several letters and the sentence field.
5. Demonstrate a suggestion button, Clear, and Speak.
6. Press `Esc` to close.

Keep the demo short and use good lighting, one hand, a plain background, and a stable camera. Have screenshots or a recorded demo ready because webcam access and speech backends can fail during a review.

## Technical explanation

The model does not directly recognize all 26 letters as independent CNN classes. The CNN output is an eight-class coarse prediction. The program examines the top predictions and then applies rules based on landmark x/y coordinates and Euclidean distances. Examples include comparing fingertip and finger-joint positions, checking whether fingers are extended, and measuring distances between landmarks. This reduces confusion between visually similar letters such as C/O, G/H, Y/J, and P/Q/Z.

The application keeps recent predictions to stabilize sentence construction. It also uses pyenchant for word suggestions and pyttsx3 for offline text-to-speech. The dataset collector creates skeleton images for each alphabet folder; it is separate from the inference application.

## Likely viva questions

**Why use landmarks instead of the raw camera image?**
Landmarks remove much of the background, skin-color, and lighting variation. The classifier receives a consistent skeleton representation instead of the full scene.

**Why use eight groups?**
Some letters have similar visual structure. Coarse grouping reduces the CNN classification problem, while landmark geometry resolves the final letter.

**Why CNN?**
A CNN learns spatial patterns in the normalized skeleton image through convolution and pooling, then maps those features to class scores.

**What happens when no hand is detected?**
The frame continues to display, but prediction is skipped because there are no landmarks to classify.

**What are the limitations?**
This is static fingerspelling, not continuous dynamic sign-language translation. It expects one visible hand and can be affected by occlusion, camera quality, pose, lighting, and background. The dataset is not perfectly balanced: class C has extra images and class Q is missing two numbered images. The reported accuracy should therefore be presented as the project's historical result, not as a freshly reproduced benchmark.

**How would you improve it?**
Balance and split the data by signer, add confidence thresholds and temporal smoothing, clamp bounding boxes, handle camera/model errors explicitly, replace hard-coded heuristic rules with a validated classifier, and add automated evaluation on a held-out test set.

## Important distinction

The repository contains a trained model and inference/data-collection programs. It does not contain a complete model-training script or a reproducible accuracy-evaluation script. Do not claim that the current run retrains the CNN.
