
#ifndef DENSEMAPPINGINTERFACE
#define DENSEMAPPINGINTERFACE

#include <mutex>
// Using c++11 in my stuff.
#include <memory>

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
  bool covarianceValid;
  Eigen::Matrix<double, 6, 6> covariance;
};

class Map;

class DenseMappingInterface {
 public:
  DenseMappingInterface(Map* pMap);

  // Tells the interface that a global bundle adjustment has occurred.
  void notifyFinishedGBA();
  // Functions for getting loop closed trajectories.
  bool isUpdatedTrajectoryAvailable();
  std::vector<PoseWithID> getUpdatedTrajectory();

  // Tells interface if the last frame was a keyframe
  void notifyKeyFrameStatusAvailable(bool keyframe_flag);
  // Functions for getting last keyframe status
  bool isKeyFrameStatusAvailable();
  bool getKeyFrameStatus();

  // Stores the covariance matrix in the interface for external reading
  void storePoseCovarianceMatrix(const Eigen::MatrixXd& poseCovariance, const std::map<unsigned long, int>& KFidToHessianCol);

 protected:

  // TODO(alexmillane): Replace this with a lambda in the trajectory function
  // Compares two keyframes, returning true if KF1 ID is > KF2 ID. For sorting.
  static bool compareKeyframes(KeyFrame* keyframe1, KeyFrame* keyframe2);

  // Map
  Map* mpMap;

  // Members to do with bundle adjusted trajectory
  bool mbUpdatedTrajectoryAvailable;
  std::mutex mMutexTrajectory;
  std::vector<PoseWithID> mvPoseTrajectory;

  // Members to do with the covariance from the trajectory
  bool mbUpdatedTrajectoryCovarianceAvailable;
  std::mutex mMutexTrajectoryCovariance;
  std::unique_ptr<Eigen::MatrixXd> mpPoseCovariance;
  std::map<unsigned long, int> mKFidToHessianCol;

  // Members to do with keyframe status
  bool mbKeyFrameStatusAvailable;
  std::mutex mMutexKeyFrameStatus;
  bool mbKeyFrameStatus;

};

}  // namespace ORB_SLAM

#endif  // DENSEMAPPINGINTERFACE
