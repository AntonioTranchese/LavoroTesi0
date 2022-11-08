import onnx_graphsurgeon as gs
import numpy as np
import onnx



model = onnx.load("yolov3-10.onnx")
graph = gs.import_onnx(model)


tensors = graph.tensors()


graph.inputs = [tensors["image_shape"].to_variable(dtype=np.float32, shape=(1,2)),tensors["convolution_output"].to_variable(dtype=np.float32, shape=(1,255,76,76)),tensors["convolution_output1"].to_variable(dtype=np.float32, shape=(1,255,38,38)),tensors["convolution_output2"].to_variable(dtype=np.float32, shape=(1,255,19,19))]
graph.outputs = [tensors["yolonms_layer_1/ExpandDims_1:0"].to_variable(dtype=np.float32, shape=(1,-1,4)),tensors["yolonms_layer_1/ExpandDims_3:0"].to_variable(dtype=np.float32, shape=(1,80,-1)),tensors["yolonms_layer_1/concat_2:0"].to_variable(dtype=np.int32, shape=(-1,3))]


graph.cleanup()


onnx.save(gs.export_onnx(graph), "subgraph0.onnx")

