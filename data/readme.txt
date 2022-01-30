this file is used for show how the file managed

the folder "gear" collect 2 kinds of error simulation
"gear_1.stl" is simulate a condition that the gear we need to assemble is inclined, rotate 0.1 rad around y
"gear_2.stl" is simulate a condition that a small foreign body near the assembly position,the foreign body make the process unstable
"gear_3.stl" is simulate a condition that there is no the base to be assembled,there is something wrong
"gear_4.stl" is simulate a condition that there is a workpiece all ready in the assembly position,there is something wrong

the folder "insert" collect 4 kinds of error simulation
"insert_1.stl" is simulate a condition that the workpiece we need to insert is inclined, rotate 0.1 rad around y
"insert_2.stl" is simulate a condition that a small foreign body in the assembly position
"insert_3.stl" is simulate a condition that there is no the base to be assembled
"insert_4.stl" is simulate a condition that there is a workpiece all ready in the assembly position

the folder "L-shaped workpiece" collect 3 kinds of error simulation
"L_1.stl" is simulate a condition that the workpiece we need to insert is inclined, rotate 0.3 rad around x
"L_2.stl" is simulate a condition that a small foreign body in the assembly position, maybe the foreign body is too small so that successful
"L_3.stl" is simulate a condition that there is no the base to be assembled
"L_4.stl" is simulate a condition that there is a workpiece all ready in the assembly position


learn and control framework of robotic assembly
DDPG+impedance


data collect
reconfiguration of tasks: change the grasped task for different task, change taskboard for different fault
run simulation of robotic assembly: roslaunch the robot
run assembly policy: choose and rosrun the policy according to task
run monitor: modify the filename and roslaunch monitor

