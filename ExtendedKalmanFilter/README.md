# Extended Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

In this project is to utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. 

# Prerequisites
1. [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) Git repository includes two files that can be used to set up and install for either Linux or Mac systems.
2. cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
3. make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
4. gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

Once the install for Prerequisites are complete, the main program can be built and run by doing the following from the project top directory.
#### Basic Build Instructions
1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make` 
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF 

# Input
Values provided by the simulator to the c++ program 
["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


# Output
values provided by the c++ program to the simulator
["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---

## Code Style
This project sticks to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Generating Additional Data
This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.
