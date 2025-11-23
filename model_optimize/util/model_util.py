import tensorflow as tf

def load_saved_model(model_path):
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]
    return infer

def load_saved_model_and_io(model_path, batch_size=128):
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
    
def create_tf_input_data(inputs):
    data = {}
    for key, ishape in inputs.items():
        data[key] = tf.random.normal(shape=ishape, mean=0.0, stddev=0.01)
    return data

smpath = "/home/ubuntu/work/data/gnet/saved_model/"

infer,input,output = load_saved_model_and_io(smpath, 8)
req = create_tf_input_data(input)
res = infer(**req)
print(res)