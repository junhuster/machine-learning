import tensorflow as tf
import onnxruntime as ort
import numpy as np

def load_saved_model(model_path):
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    return infer

def load_saved_model_and_io(model_path, batch_size=-1):
    infer = load_saved_model(model_path)
    input = {}
    output = {}
    for key, value in infer.structured_input_signature[1].items():
        new_shape = (batch_size,) + value.shape[1:]
        input[key] = new_shape
    for key, value in infer.structured_outputs.items():
        new_shape = (batch_size,) + value.shape[1:]
        output[key] = new_shape
    return infer,input,output
    
def create_input_data(inputs):
    data = {}
    for key, ishape in inputs.items():
        #tf.random.normal生成的数据无法请求onnxruntime, 因此改为np.random.normal
        #data[key] = tf.random.normal(shape=ishape, mean=0.0, stddev=0.01)
        data[key] = np.random.normal(loc=0.0, scale=0.01, size=ishape).astype(np.float32)
    return data


def load_onnx_model(model_path):
    inferSession = ort.InferenceSession(model_path)
    return inferSession

def load_onnx_model_io(model_path, batch_size=-1):
    inferSession = load_onnx_model(model_path)
    inputs = {}
    outputs = {}
    for ele in inferSession.get_inputs():
        print(ele.shape[1:])
        newShape = [batch_size] + ele.shape[1:]
        inputs[ele.name] = newShape
    for ele in inferSession.get_outputs():
        newShape = [batch_size] + ele.shape[1:]
        outputs[ele.name] = newShape
    return inferSession,inputs,outputs

def convert_tf_inputs_to_onnx_inputs(tf_req, ort_inputs):
    ort_req = {}
    for keyo,valueo in ort_inputs.items():
        for keyt,valuef in tf_req.items():
            if keyo.startswith(keyt):
                ort_req[keyo] = tf_req[keyt]
    return ort_req