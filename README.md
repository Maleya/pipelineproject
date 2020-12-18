# pipelinetracker

Here we need to describe and document our code a bit, as well as a manual and walk through how to run bags manually (rosbag play etc)


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

To get started right away with sensible defaults:

```sh
black {source_file_or_directory}
```

You can run _Black_ as a package if running it as a script doesn't work:

```sh
python -m black {source_file_or_directory}
```


From workspace in terminal 1
```sh
roscore
```

From workspace in terminal 2 to visualize line detection
```sh
rqt
```
From ~/pipelineproject/src/bags in Terminal 3
```sh
rosbag play bagName.bag
```

From ~/pipelineproject/src Terminal 4
```sh
rosrun pipelineproject main.py
```

In rviz -> Add -> By Topic -> /line_image/Image