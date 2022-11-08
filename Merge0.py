import sclblonnx as so


g1 = so.graph_from_file("yolov3(darknet53-coco)_new.onnx")
g2 = so.graph_from_file("subgraph0.onnx")

g_merge = so.merge(sg1=g1, sg2=g2, io_match=[("conv2d_59", "convolution_output2"),("conv2d_67","convolution_output1"),("conv2d_75","convolution_output")],_sclbl_check=False)

so.graph_to_file(g_merge, "merged_model_new.onnx")