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

![R-CNN](https://cdn-images-1.medium.com/max/800/1*D2sFqL329qKKx4Tvl31IhQ.png)

## Fast R-CNN
![Fast R-CNN](https://cdn-images-1.medium.com/max/800/1*iWyUwIPO-5kA2ECAfaaPSg.png)

# Faster R-CNN

![Faster-RCNN](https://cdn-images-1.medium.com/max/800/1*7heX-no7cdqllky-GwGBfQ.png)


![Faster R-CNN2](https://cdn-images-1.medium.com/max/800/1*LHk_CCzzfP9mzw280kG70w.png)


## R-FCN

![R-FCN](https://cdn-images-1.medium.com/max/800/1*cHEvY3E2HW65AF-mPeMwOg.png)

![R-FCN](https://cdn-images-1.medium.com/max/800/1*Q20DdanzQbvBjg4DLvJkGg.png)

## You Only Look Once (YOLO)

![YOLO](https://cdn-images-1.medium.com/max/800/1*n09xW-miKM_0M62a8VsVjw.png)

## Single-Shot Detector (SSD)

![SSD](https://cdn-images-1.medium.com/max/800/1*9juuB8HOBnoNqvEruiCT2A.png)

## Mask Region-based Convolutional Network (Mask R-CNN)

![Mask](https://cdn-images-1.medium.com/max/800/1*IX55uRz8s-E79AvNhpsBmg.png)

![Mask](https://cdn-images-1.medium.com/max/800/1*cje-Cm-RO_1hCe1YFxpd1Q.png)
REference List
====
1. [https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)

2. [https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)



