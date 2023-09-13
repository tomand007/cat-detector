import numpy as np
import tensorflow as tf
import cv2

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_names = [line.rstrip('\n') for line in open("labelmap.txt")]

cap = cv2.VideoCapture('rtsp://tomand007:zxcq096zb@192.168.68.117:554/stream1')
if cap.isOpened() == False:
    print('file not found or problem with opening')

living_objects = ['person', 'cat', 'dog', 'bird', 'mouse']

while cap.isOpened():
    ret, frame = cap.read()

    if ret:

        original_height, original_width = frame.shape[:2]
        
        resized_frame = cv2.resize(frame, (300, 300))

        
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_frame, 0).astype('uint8')

        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        
        outputLocations = interpreter.get_tensor(output_details[0]['index'])
        outputClasses = interpreter.get_tensor(output_details[1]['index'])
        outputScores = interpreter.get_tensor(output_details[2]['index'])
        numDetections = interpreter.get_tensor(output_details[3]['index'])

        numDetectionsOutput = int(np.minimum(numDetections[0], 10))

        for i in range(numDetectionsOutput):
            class_name = label_names[int(outputClasses[0][i])]
            confidence = outputScores[0][i]

            if class_name in living_objects and confidence > 0.5:
                inputSize = 300
                left = int(outputLocations[0][i][1] * original_width)
                top = int(outputLocations[0][i][0] * original_height)
                right = int(outputLocations[0][i][3] * original_width)
                bottom = int(outputLocations[0][i][2] * original_height)
                
                
                
                
                color = (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                label = f"{class_name}: {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(frame, (left, top - int(1.5*h)), (left + w, top), color, -1)
                cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
