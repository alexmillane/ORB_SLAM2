
#ifndef DENSEMAPPINGINTERFACE
#define DENSEMAPPINGINTERFACE

#include <mutex>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "orb_slam_2/Map.h"

namespace ORB_SLAM2 {

// A type of a stamped pose
typedef std::pair<cv::Mat, double> PoseStamped;

// A type containing a keyframe pose, a timestampe and keyframe ID
struct PoseWithID {
  cv::Mat pose;
  double timestamp;
  long unsigned int id;
};

class Map;

class DenseMappingInterface {
 public:
  DenseMappingInterface(Map* pMap);

  // Tells the interface that a global bundle adjustment has occurred.
  void notifyFinishedGBA();
  // Functions for getting loop closed trajectories.
  bool isUpdatedTrajectoryAvailable();
  //std::vector<PoseStamped> getUpdatedTrajectory();
  std::vector<PoseWithID> getUpdatedTrajectory();

  // Tells interface if the last frame was a keyframe
  void notifyKeyFrameStatusAvailable(bool keyframe_flag);
  // Functions for getting last keyframe status
  bool isKeyFrameStatusAvailable();
  bool getKeyFrameStatus();

 protected:

  // TODO(alexmillane): Replace this with a lambda in the trajectory function
  // Compares two keyframes, returning true if KF1 ID is > KF2 ID. For sorting.
  static bool compareKeyframes(KeyFrame* keyframe1, KeyFrame* keyframe2);

  // Map
  Map* mpMap;

  // Members to do with bundle adjusted trajectory
  bool mbUpdatedTrajectoryAvailable;
  std::mutex mMutexTrajectory;
  //std::vector<PoseStamped> mvPoseTrajectory;
  std::vector<PoseWithID> mvPoseTrajectory;


  // Members to do with keyframe status
  bool mbKeyFrameStatusAvailable;
  std::mutex mMutexKeyFrameStatus;
  bool mbKeyFrameStatus;

};

}  // namespace ORB_SLAM

#endif  // DENSEMAPPINGINTERFACE
