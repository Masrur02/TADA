



#!/bin/bash

# MESH_DIR is the only argument passed
MESH_DIR=$1  # e.g., "mesh" or any folder name

# Fixed part of the path
BASE_PATH="/home/vail/TADA_WS/src/TADA/work_dirs/local-basic"

# Construct the full paths dynamically
CONFIG_FILE="${BASE_PATH}/${MESH_DIR}/a.py"
CHECKPOINT_FILE="${BASE_PATH}/${MESH_DIR}/iter_40000.pth"
SHOW_DIR="${BASE_PATH}/${MESH_DIR}/preds"

echo "----------------------------------------------------"
echo "Running IDANAV Inference + Traversability Projection"
echo "----------------------------------------------------"
echo "Config File:       $CONFIG_FILE"
echo "Checkpoint File:   $CHECKPOINT_FILE"
echo "Output Directory:  $SHOW_DIR"
echo "----------------------------------------------------"

# Start ROS core (if not already running)
if ! pgrep -x "roscore" > /dev/null; then
  echo "Starting roscore..."
  roscore &
  sleep 3
fi

# Launch IDANAV (segmentation + traversability)
echo "Starting IDANAV node..."
python3 -m tools.in_ros "$CONFIG_FILE" "$CHECKPOINT_FILE" --opacity 0.5 --dataset "rugd" &
IN_ROS_PID=$!

# Wait 8 seconds to ensure in_ros publishes /trav_map and /points
sleep 8

# Launch the traversability projector (x,y,z,trav)
echo "Starting projection node..."
python3 /home/vail/TADA_WS/src/TADA/tools/projection.py &
PROJECTION_PID=$!



# Wait a few seconds to ensure costmap starts publishing /semantic_costmap
sleep 5

# Launch the RViz goal relay (bridge /move_base_simple/goal → /goal)
echo "Starting RViz goal relay..."
python3 /home/vail/TADA_WS/src/TADA/tools/goal.py &
GOAL_PID=$!



# Wait for user to terminate
echo "Press [CTRL+C] to stop all nodes."
wait

# On termination, clean up all background processes
trap "kill $IN_ROS_PID $PROJECTION_PID; exit" INT

