input = load("yolov3(darknet53-coco).mat");
net = input.detector.Network;

inputImageSize = net.Layers(1,1).InputSize;

anchorBoxes = input.detector.AnchorBoxes;

%analyzeNetwork(net);



filename = 'yolov3(darknet53-coco)_new.onnx';
exportONNXNetwork(net,filename,"OpsetVersion",12);%Salvato con opset=12
