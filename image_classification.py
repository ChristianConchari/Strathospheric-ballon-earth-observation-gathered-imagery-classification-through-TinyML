from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import time

# Cargar el modelo TFLite y ver algunos detalles sobre la entrada/salida
TFLITE_MODEL = "/home/deepspace/Strathospheric-ballon-earth-observation-gathered-imagery-classification-through-TinyML/model_cnn_quant.tflite"
tflite_interpreter = Interpreter(model_path=TFLITE_MODEL)
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

image = cv2.imread("/home/deepspace/Buena_238.jpg")
orig_h, orig_w, _ = image.shape
cp_image = np.copy(image)
image = cv2.resize(image, (100,100)).astype(np.float32)
image = np.expand_dims(image, axis=0)
tflite_interpreter.set_tensor(input_details[0]['index'], image)

tflite_interpreter.invoke()

start_time = time.time()
tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()

print(tflite_model_predictions)

print(f'Image processed in {round((end_time-start_time)*1000,2)} ms')

if tflite_model_predictions > 0:
    pred = 'good'
    cv2.rectangle(cp_image, (0,0), (orig_w, orig_h), (0,255,0), 3)
else:
    pred = 'bad'
    cv2.rectangle(cp_image, (0,0), (orig_w, orig_h), (0,0,255), 3)

cv2.imwrite(f'prediction_{pred}.jpg',cp_image)
