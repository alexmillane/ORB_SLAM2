
#ifndef DENSEMAPPINGINTERFACE
#define DENSEMAPPINGINTERFACE

#include <mutex>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "orb_slam_2/Map.h"

namespace ORB_SLAM2 {

// A type of a stamped pose
typedef std::pair<cv::Mat, double> PoseStamped;

class Map;

class DenseMappingInterface {
 public:
  DenseMappingInterface(Map* pMap);

  // Tells the interface that a global bundle adjustment has occurred.
  void notifyFinishedGBA();

  // Functions for getting loop closed trajectories.
  bool isUpdatedTrajectoryAvailable();
  std::vector<PoseStamped> getUpdatedTrajectory();

 protected:

  // Compares two keyframes, returning true if KF1 ID is > KF2 ID. For sorting.
  static bool compareKeyframes(KeyFrame* keyframe1, KeyFrame* keyframe2);

  // Map
  Map* mpMap;

  // Flag indicating if a new (unfetched) trajectory is available
  bool mbUpdatedTrajectoryAvailable;

  // The latest loop closed trajectory stored here
  std::vector<PoseStamped> mvPoseTrajectory;

  // A mutex which locks the trajectory for read/write
  std::mutex mMutexTrajectory;
};

}  // namespace ORB_SLAM

#endif  // DENSEMAPPINGINTERFACE
