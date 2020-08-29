### personTrack


#### NMS  TorchVision.ops.nms  
yolov4 nms 检测框比较少 所以速度尚可  
mtcnn nms 检测框

#### 迁移学习
yolov4 在coco上检测了80类,倘若只检测行人可以大幅度压缩模型  
目前可以在nms之前先将行人拿出来,减少压力

#### 速度  
+ yolov4 50ms  
+ reid 15ms  
+ mtcnn 20ms  
+ mobilefacenet 5ms

+ batch inference  
  同时刻t时间内 所有的检测框一同送入识别模块提取特征

+ ext 编译一下ext下的文件 python setup.py install

# 统一使用root路径执行文件

https://github.com/ultralytics/yolov5/issues/22