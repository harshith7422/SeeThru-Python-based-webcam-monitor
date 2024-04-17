<h2> SeeThru: Python-based-webcam-monitor-that-detects-users-and-objects-keeping-you-in-the-loop. </h2>
Install all the libraries given in the requirements.txt file

```bash
pip install -r requirements. txt
```
If there is any problem in installing the libraries through the requirements.txt file use the following commands:

```bash
pip install cv2 
pip install time
pip install pygame
pip install subprocess
pip install numpy 
pip install os
pip install gtts
pip install pydub
pip install reportlab
```
[face_detection.py](face_detection.py) file uses cascade filters to detect the face and there is a threshold value of 5sec set in the function such that, if the user is not detected for more than 5sec it shoots an alert with buzzer sound.

[object_detection.py](object_detection.py) file uses YOLO libraries to detect the objects that are present in the captured area and prints the objects with their respective positions in the terminal.

We integrate the features of both [face_detection.py](face_detection.py) and [object_detection.py](object_detection.py) in [main.py](main.py).

Note:

[1] Replace the path of haarcascade_frontalface_default.xml and buzzer_sound.wav in the code.

[2] Use "q" to quit capturing the video.
