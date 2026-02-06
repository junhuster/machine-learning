import model_util as mu
import tensorflow as tf
import numpy as np
import onnxruntime as ort

smpath = "/home/ubuntu/work/data/gnet/saved_model/"

infer,input,output = mu.load_saved_model_and_io(smpath, 8)
req = mu.create_input_data(input)
res = infer(**req)
print(res)

onnx_path = "/home/ubuntu/work/data/gnet/model.onnx"
inferSession,inputs,outputs = mu.load_onnx_model_io(onnx_path)
print(f'onnx inputs:{inputs}\nouputs:{outputs}')
ort_output = []
for key,value in outputs.items():
    ort_output.append(key)

ort_req = mu.convert_tf_inputs_to_onnx_inputs(req, inputs)
ort_resp = inferSession.run(ort_output, ort_req)
print(ort_resp)