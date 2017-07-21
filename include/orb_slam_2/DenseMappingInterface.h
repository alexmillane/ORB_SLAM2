
#ifndef DENSEMAPPINGINTERFACE
#define DENSEMAPPINGINTERFACE

#include <mutex>
// Using c++11 in my stuff.
#include <memory>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "orb_slam_2/Map.h"

namespace ORB_SLAM2 {

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
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DenseMappingInterface(Map* pMap);

  // Tells the interface that a global bundle adjustment has occurred.
  void notifyFinishedGBA();
  // Functions for getting loop closed trajectories.
  bool isUpdatedTrajectoryAvailable();
  std::vector<PoseWithID> getUpdatedTrajectory();

/*  // Tells interface if the last frame was a keyframe
  void notifyKeyFrameStatusAvailable(bool keyframe_flag);
  // Functions for getting last keyframe status
  bool isKeyFrameStatusAvailable();
  bool getKeyFrameStatus();*/

  // Stores the covariance matrix in the interface for external reading
  void storePoseCovarianceMatrix(
      const Eigen::MatrixXd& poseCovariance,
      const std::map<unsigned long, int>& KFidToHessianCol);

  // Marks a keyframe as a patch baseframe
  bool addKeyframeAsPatchBaseframe(unsigned long KFid);
  void removeAllKeyframesAsPatchBaseFrames();

  // Gets the patches baseframe poses and covariances
  bool getPatchBaseFramePosesAndCovariances(
      std::vector<cv::Mat>& patchPoses,
      std::vector<Eigen::Matrix<double, 6, 6>,
                  Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>>&
          patchConditionalCovariances);
  bool extractPatchBaseFramePoses(std::vector<cv::Mat>& patchPoses);
  bool extractPatchBaseFrameConditionalCovariances(
      std::vector<Eigen::Matrix<double, 6, 6>,
                  Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>>&
          patchConditionalCovariances);

 protected:
  // Extracts the conditional covariance of a pair of keyframes
  bool extractCoditionalFromMarginalCovarianceOfPair(
      const Eigen::Matrix<double, 12, 12>& marginalCovariance,
      Eigen::Matrix<double, 6, 6>& conditionalCovariance);
  // Extracts the joint marginal covariance of a pair of keyframes
  bool extractJointMarginalCovarianceOfPair(
      const std::array<unsigned long, 2>& KFids,
      Eigen::Matrix<double, 12, 12>& marginalCovariance);

  // Map
  Map* mpMap;

  // Members to do with keyframe status
  bool mbKeyFrameStatusAvailable;
  std::mutex mMutexKeyFrameStatus;
  bool mbKeyFrameStatus;

  // Members to do with bundle adjusted trajectory
  bool mbUpdatedTrajectoryAvailable;
  std::mutex mMutexTrajectory;
  std::vector<PoseWithID> mvPoseTrajectory;

  // Members to do with the covariance from the trajectory
  bool mbTrajectoryCovarianceAvailable;
  std::mutex mMutexTrajectoryCovariance;
  std::unique_ptr<Eigen::MatrixXd> mpPoseCovariance;
  std::map<unsigned long, int> mKFidToHessianCol;

  // Members to do with patches
  std::vector<unsigned long> mvPatchKFids;
};

}  // namespace ORB_SLAM

#endif  // DENSEMAPPINGINTERFACE
