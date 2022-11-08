name = 'darknet53-coco';

detector = yolov3ObjectDetector(name);

disp(detector)

analyzeNetwork(detector.Network)

img = imread('zidane.jpg');
img = preprocess(detector,img);
img = im2single(img);
[bboxes,scores,labels] = detect(detector,img,'DetectionPreprocessing','none');

detectedImg = insertObjectAnnotation(img,'Rectangle',bboxes,labels);
figure
imshow(detectedImg)