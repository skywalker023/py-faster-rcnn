# Korean Food Image Detection

* This is the server code running Faster-RCNN.
* We built a Korean food image dataset with 41 classes(40 + background).

Installation instructions will be on Girshick's original code.
I added some post-processing code and visualizing tools for custom inputs.

In order to use this code, you must have your dataset as the same format with Pascal VOC.

Below is part of our dataset images.

![dataset](https://github.com/skywalker023/py-faster-rcnn/blob/master/data/dataset.png?raw=true)
 
For the server, you need to install [Tornado](https://pypi.python.org/pypi/tornado).