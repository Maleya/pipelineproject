# pipelinetracker

The goal of the pipelinetracker was to develop a system that tracks a pipeline autonomously based on Computer Vision and can be integrated in a larger system such as autonomous underwater vehicles (AUV’s). The overarching objective is to enable an AUV to find, follow and inspect pipelines, whereof pipelinetracker represents the first phase. 

Developed in Python, the present code analyses data from a Multibeam Echosounder, that scans the seafloor while moving forward. With usage of the OpenCV library 2 different methods for line detection were developed and can be selected, hough transform (line_detect1) or PCA (line_detect2). A buffer is stabilizing the line detection and helps avoiding rapid changes of line rotations. This is according to the real-world application to pipelines, that are in most cases rectilinear and without sharp corners. The code is implemented in ROS and publishes the results of the line detection as nodes and as lines in the videos. Detailed steps can be followed directly in the code, where comments help for better understanding (see /src). The final presentation summarizes the pipelinetracker project, its methods and includes documenting photos of data as well as the resulting videos with detected lines. 

The final presentation can be found here:
https://docs.google.com/presentation/d/1tERIsS82jEA7BZE8sIjmZ8d3fV63z2_54cgfv__a2Oo/edit?usp=sharing

## Installation and usage

### Installation

The pipetracker package can installed directly from git. Clone it into your desired package folder, which should be where you keep the ROS packages. 
For scripts with ROS, do `chmod +x script.py` . 


in workspace dir
```sh
workspace_make
```

```sh
cd workspace/devel
chmod +x setup.bash
```

```sh
source ~/workspace/devel/setup.bash
```

```sh
cd ~/pipelineproject/src
chmod +x main.py
```




### Usage

From workspace in terminal 1

```sh
roscore
```

From workspace in terminal 2 to visualize line detection
```sh
rviz
```
From ~/pipelineproject/bags in Terminal 3
```sh
rosbag play bag-1.bag
rosbag play bag-2.bag
rosbag play bag-3.bag
```

From ~/pipelineproject/src Terminal 4

```sh
rosrun pipelineproject main.py --method line_detect1 
rosrun pipelineproject main.py --method line_detect2 
```

In rviz -> Add -> By Topic -> /line_image/Image
