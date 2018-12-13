Why object detection instead of image classification?
===
![object detection](https://cdn-images-1.medium.com/max/800/1*Hz6t-tokG1niaUfmcysusw.jpeg)

Performance Metric
===
## Intersection over Union (IoU)
![IOU](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png)
![Poor IOU](https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_examples.png)

## Mean Average Precision (mAP)

**Precision** measures how accurate is your predictions. i.e. the percentage of your positive predictions are correct.

**Recall** measures how good you find all the positives. For example, we can find 80% of the possible positive cases in our top K predictions.

### AP
![AP](https://cdn-images-1.medium.com/max/800/1*9ordwhXD68cKCGzuJaH2Rg.png)
![Precision vs Recall]

![TP](https://cdn-images-1.medium.com/max/800/1*KDBbl6fT1pBLjUzlKCCBQA.png)
![Ex](https://cdn-images-1.medium.com/max/800/1*0-EiedG6QZ07ysMPpLmHtQ.png)

Let’s compute the precision and recall value for the row with rank #3.

Precision is the proportion of TP = 2/3 = 0.67.

Recall is the proportion of TP out of the possible positives = 2/5 = 0.4.

![AP](https://cdn-images-1.medium.com/max/800/1*5C4GaqxfPrq-9lFINMix8Q.png)


Detection Alogorithms
===
## R-CNN
- Selective Search
- Region proposals
- an SVM to classify the region and 
- a linear regressor to tighten the bounding box of the object, if such an object exists
![R-CNN](https://cdn-images-1.medium.com/max/800/1*D2sFqL329qKKx4Tvl31IhQ.png)

## Fast R-CNN
- Performing feature extraction over the image before proposing regions, thus only running one CNN over the entire image instead of 2000 CNN’s over 2000 overlapping regions
- Replacing the SVM with a softmax layer, thus extending the neural network for predictions instead of creating a new model

![Fast R-CNN](https://cdn-images-1.medium.com/max/800/1*iWyUwIPO-5kA2ECAfaaPSg.png)

# Faster R-CNN
- replace the slow selective search algorithm with a fast neural net
- region proposal network (RPN)
- a 3x3 sliding window moves across the feature map and maps it to a lower dimension (e.g. 256-d)
- For each sliding-window location, it generates multiple possible regions based on k fixed-ratio anchor boxes
- Each region proposal consists of a) an “objectness” score for that region and b) 4 coordinates representing the bounding box of the region
- Faster R-CNN = RPN + Fast R-CNN.
![Faster-RCNN](https://cdn-images-1.medium.com/max/800/1*7heX-no7cdqllky-GwGBfQ.png)


![Faster R-CNN2](https://cdn-images-1.medium.com/max/800/1*LHk_CCzzfP9mzw280kG70w.png)


## R-FCN
- increase speed by maximizing shared computation.
- Region-based Fully Convolutional Net, shares 100% of the computations across every single output
- when performing classification of an object, we want to learn location invariance in a model
- when performing detection of the object, we want to learn location variance
- we’re trying to share convolutional computations across 100% of the net, how do we compromise between location invariance and location variance?
- position-sensitive score maps.

Steps:
-Run a CNN 
- generate a score bank of the aforementioned “position-sensitive score maps.”
- There should be k²(C+1) score maps, with k² representing the number of relative positions to divide an object 
- RPN: generate regions of interest (RoI’s)
- For each RoI, divide it into the same k² “bins” or subregions as the score maps
- For each bin, check the score bank to see if that bin matches the corresponding position of some object. 
-  k² bins has an “object match” value for each class, average the bins to get a single score per class.
- Classify the RoI with a softmax over the remaining C+1 dimensional vector
![R-FCN](https://cdn-images-1.medium.com/max/800/1*cHEvY3E2HW65AF-mPeMwOg.png)

![R-FCN](https://cdn-images-1.medium.com/max/800/1*Q20DdanzQbvBjg4DLvJkGg.png)

## You Only Look Once (YOLO)
- The final layer outputs a S*S*(C+B*5)
-  C is the number of estimated probabilities for each class. B is the fixed number of anchor boxes per cell, each of these boxes being related to 4 coordinates (coordinates of the center of the box, width and height) and a confidence value.
![YOLO](https://cdn-images-1.medium.com/max/800/1*n09xW-miKM_0M62a8VsVjw.png)

## Single-Shot Detector (SSD)

![SSD](https://cdn-images-1.medium.com/max/800/1*9juuB8HOBnoNqvEruiCT2A.png)
- Pass the image through a series of convolutional layers
- For each location in each of these feature maps, use a 3x3 convolutional filter to evaluate a small set of default bounding boxes
- For each box, simultaneously predict a) the bounding box offset and b) the class probabilities
- During training, match the ground truth box with these predicted boxes based on IoU. The best predicted box will be labeled a “positive,” along with all other boxes that have an IoU with the truth >0.5.
## Mask Region-based Convolutional Network (Mask R-CNN)
- The initial RoIPool layer used in the Faster R-CNN is replaced by a RoIAlign layer.
- It removes the quantization of the coordinates of the original RoI and computes the exact values of the locations. The RoIAlign layer provides scale-equivariance and translation-equivariance with the region proposals.
![Mask](https://cdn-images-1.medium.com/max/800/1*IX55uRz8s-E79AvNhpsBmg.png)

![Mask](https://cdn-images-1.medium.com/max/800/1*cje-Cm-RO_1hCe1YFxpd1Q.png)
REference List
====
1. [https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)

2. [https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)



