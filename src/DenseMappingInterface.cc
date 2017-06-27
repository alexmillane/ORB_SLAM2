#include <iostream>
#include <algorithm>

#include "orb_slam_2/Converter.h"
#include "orb_slam_2/DenseMappingInterface.h"

namespace ORB_SLAM2 {

DenseMappingInterface::DenseMappingInterface(Map* pMap)
    : mpMap(pMap), mbUpdatedTrajectoryAvailable(false) {
  //
}

void DenseMappingInterface::notifyFinishedGBA() {
  // DEBUG
  std::cout << "Bundle Adjustment finsihed. Storing pose trajectory."
            << std::endl;
  // Locking the stored trajectory to overwrite
  unique_lock<mutex> lock_traj(mMutexTrajectory);
  unique_lock<mutex> lock_cov(mMutexTrajectoryCovariance);
  // Getting the keyframes in the map
  vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  // Sort the keyframes according to their ID
  // TODO(alex.millane:) Do this with a lambda.
  std::sort(vpKFs.begin(), vpKFs.end(), compareKeyframes);
  // Storing the keyframe poses and timestamps
  mvPoseTrajectory.clear();
  mvPoseTrajectory.reserve(vpKFs.size());
  for (size_t i = 0; i < vpKFs.size(); i++) {
    // Extracting keyframe
    KeyFrame* pKF = vpKFs[i];
    // Extracting and storing pose with ID
    PoseWithID pose_with_id;
    pose_with_id.pose = pKF->GetPose();
    pose_with_id.timestamp = pKF->mTimeStamp;
    pose_with_id.id = pKF->mnId;
    // Extracting and storing the covariance if its available
    if (mbUpdatedTrajectoryCovarianceAvailable) {
      auto itKFidToHessianCol = mKFidToHessianCol.find(pKF->mnId);
      if (itKFidToHessianCol != mKFidToHessianCol.end()) {
        pose_with_id.covarianceValid = true;
        int hessianCol = itKFidToHessianCol->second;
        pose_with_id.covariance =
            mpPoseCovariance->block<6, 6>(hessianCol, hessianCol);

        // DEBUG
        std::cout << "covariance found for keyframe ID: " << pKF->mnId << std::endl;
        //std::cout << "covariance: " << pose_with_id.covariance << std::endl;
      } else {
        // DEBUG
        std::cout << "covariance not found for keyframe ID: " << pKF->mnId << std::endl;
      }
    }
    // Storing this pose
    mvPoseTrajectory.push_back(pose_with_id);
  }
  // Indicating the trajectory is ready for retreival
  mbUpdatedTrajectoryAvailable = true;
  // Indicating that the covariances have been used and freeing the memory
  mbUpdatedTrajectoryCovarianceAvailable = false;
}

bool DenseMappingInterface::isUpdatedTrajectoryAvailable() {
  return mbUpdatedTrajectoryAvailable;
}

std::vector<PoseWithID> DenseMappingInterface::getUpdatedTrajectory() {
  mbUpdatedTrajectoryAvailable = false;
  unique_lock<mutex> lock(mMutexTrajectory);
  return mvPoseTrajectory;
}

void DenseMappingInterface::notifyKeyFrameStatusAvailable(bool keyframe_flag) {
  // Saving the keyframe status
  mbKeyFrameStatus = keyframe_flag;
  // Indicating a new status is available
  mbKeyFrameStatusAvailable = true;
}

bool DenseMappingInterface::isKeyFrameStatusAvailable() {
  return mbKeyFrameStatusAvailable;
}
bool DenseMappingInterface::getKeyFrameStatus() {
  unique_lock<mutex> lock(mMutexKeyFrameStatus);
  mbKeyFrameStatusAvailable = false;
  return mbKeyFrameStatus;
}

void DenseMappingInterface::storePoseCovarianceMatrix(
    const Eigen::MatrixXd& poseCovariance, const std::map<unsigned long, int>& KFidToHessianCol) {
  // Copy constructor
  // NOTE(alexmillane): Could possibly replace this with a move sometime.
  unique_lock<mutex> lock(mMutexTrajectoryCovariance);
  mpPoseCovariance.reset(new Eigen::MatrixXd(poseCovariance));
  // Replacing the KF to Hessian collumn map 
  mKFidToHessianCol = KFidToHessianCol;
  // Indicating that the covariance is ready
  mbUpdatedTrajectoryCovarianceAvailable = true;
}

bool DenseMappingInterface::compareKeyframes(KeyFrame* keyframe_1, KeyFrame* keyframe_2) {
    return (keyframe_1->mnId) < (keyframe_2->mnId);
}

}  // namespace ORB_SLAM