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
  // Locking the trajectory to overwrite
  unique_lock<mutex> lock(mMutexTrajectory);
  // Looping over the keyframes and storing poses
  vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

  // Sort the keyframes according to their ID
  std::sort(vpKFs.begin(), vpKFs.end(), compareKeyframes);

  mvPoseTrajectory.clear();
  mvPoseTrajectory.reserve(vpKFs.size());
  for (size_t i = 0; i < vpKFs.size(); i++) {
    // Extracting keyframe
    KeyFrame* pKF = vpKFs[i];
    // Extracting and storing pose
    cv::Mat Twc = pKF->GetPose();
    mvPoseTrajectory.push_back(Converter::toAffine3d(Twc));
  }

  // Indicating the trajectory is ready for retreival
  mbUpdatedTrajectoryAvailable = true;
}

bool DenseMappingInterface::isUpdatedTrajectoryAvailable() {
  return mbUpdatedTrajectoryAvailable;
}

std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d> >
DenseMappingInterface::getUpdatedTrajectory() {
  // Locking the trajectory to prevent modification on copy
  mbUpdatedTrajectoryAvailable = false;
  unique_lock<mutex> lock(mMutexTrajectory);
  return mvPoseTrajectory;
}

bool DenseMappingInterface::compareKeyframes(KeyFrame* keyframe_1, KeyFrame* keyframe_2) {
    return (keyframe_1->mnId) < (keyframe_2->mnId);
}


}  // namespace ORB_SLAM