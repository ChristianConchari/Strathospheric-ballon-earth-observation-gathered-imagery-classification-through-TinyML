from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import time

# Cargar el modelo TFLite y ver algunos detalles sobre la entrada/salida
TFLITE_MODEL = "/home/pi/deepspace/Strathospheric-ballon-earth-observation-gathered-imagery-classification-through-TinyML/model_cnn_quant.tflite"
tflite_interpreter = Interpreter(model_path=TFLITE_MODEL)
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

i=0

# Keep looping
while True:
    # Get the current time, increase delta and update the previous variable
    delta += time() - previous
    previous = time()

    # Check if 60 (or some other value) seconds passed
    
    if delta > 60:
        i+=1
        # Operations on image
        # Create a new VideoCapture object
        cap = cv2.VideoCapture(0)
        _, img = cap.read()
        cv2.imwrite(f"image{i}.jpg", img)
        # Reset the time counter
        delta = 0
        cap.release()

        orig_h, orig_w, _ = image.shape
        cp_image = np.copy(image)
        image = cv2.resize(image, (100,100)).astype(np.float32)
        image = 1/255.0
        image = np.expand_dims(image, axis=0)
        tflite_interpreter.set_tensor(input_details[0]['index'], image)

        tflite_interpreter.invoke()
        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])

        if tflite_model_predictions > 0.5:
            pred = 'good'
            cv2.rectangle(cp_image, (0,0), (orig_w, orig_h), (0,255,0), 3)
        else:
            pred = 'bad'
            cv2.rectangle(cp_image, (0,0), (orig_w, orig_h), (0,0,255), 3)

        cv2.imwrite(f'prediction_{pred}.jpg', cp_image)
