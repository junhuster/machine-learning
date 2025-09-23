import tensorflow as tf

def convert_saved_model_to_pbtxt(saved_model_dir, pbtxt_file_path):
    # 加载 SavedModel
    loaded_model = tf.saved_model.load(saved_model_dir)

    # 从加载的模型获取计算图
    concrete_fun = loaded_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # 获取计算图的 GraphDef
    graph_def = concrete_fun.graph.as_graph_def()

    # 将 GraphDef 写入 JSON 文件
    tf.io.write_graph(graph_def, "./", pbtxt_file_path, as_text=True)
    print(f"模型已成功转换为 {pbtxt_file_path}")

# 使用示例
convert_saved_model_to_pbtxt('../saved_model', 'model.pbtxt')
