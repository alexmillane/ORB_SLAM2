#include <algorithm>
#include <iostream>

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
  std::sort(vpKFs.begin(), vpKFs.end(),
            [](KeyFrame const* kf1, KeyFrame const* kf2) {
              return (kf1->mnId) < (kf2->mnId);
            });
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
    // Extracting and storing the marginal covariance if its available
    if (mbTrajectoryCovarianceAvailable) {
      auto itKFidToHessianCol = mKFidToHessianCol.find(pKF->mnId);
      if ((itKFidToHessianCol != mKFidToHessianCol.end()) && (i != 0)) {
        pose_with_id.covarianceValid = true;
        int hessianCol = itKFidToHessianCol->second;
        pose_with_id.covariance =
            mpPoseCovariance->block<6, 6>(hessianCol, hessianCol);
      }
    }
    // Storing this pose
    mvPoseTrajectory.push_back(pose_with_id);
  }
  // Indicating the trajectory is ready for retreival
  mbUpdatedTrajectoryAvailable = true;
}

bool DenseMappingInterface::isUpdatedTrajectoryAvailable() {
  return mbUpdatedTrajectoryAvailable;
}

std::vector<PoseWithID> DenseMappingInterface::getUpdatedTrajectory() {
  mbUpdatedTrajectoryAvailable = false;
  unique_lock<mutex> lock(mMutexTrajectory);
  return mvPoseTrajectory;
}

/*void DenseMappingInterface::notifyKeyFrameStatusAvailable(bool keyframe_flag) {
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
}*/

void DenseMappingInterface::storePoseCovarianceMatrix(
    const Eigen::MatrixXd& poseCovariance,
    const std::map<unsigned long, int>& KFidToHessianCol) {
  // DEBUG
  std::cout << "Storing the pose covariance." << std::endl;

  // Copy constructor
  // NOTE(alexmillane): Could possibly replace this with a move sometime.
  unique_lock<mutex> lock(mMutexTrajectoryCovariance);
  mpPoseCovariance.reset(new Eigen::MatrixXd(poseCovariance));
  // Replacing the KF to Hessian collumn map
  mKFidToHessianCol = KFidToHessianCol;
  // Indicating that the covariance is ready
  mbTrajectoryCovarianceAvailable = true;

  // DEBUG
  std::cout << "Pose covariance stored." << std::endl;
}

bool DenseMappingInterface::addKeyframeAsPatchBaseframe(unsigned long KFid) {
  // Checking if the keyframe exists
  // NOTE(alexmillane): This is to check that the keyframe still exists as it
  // might have been removed

  std::cout << "Trying to add KFid: " << KFid << " as patch base frame." << std::endl;

  std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
  auto it =
      std::find_if(vpKFs.begin(), vpKFs.end(),
                   [KFid](KeyFrame const* kf) { return (kf->mnId) == KFid; });
  // If found push this onto the list
  // NOTE(alexmillane): We also mark the keyframe associated with the baseframe as undeletable 
  // TODO(alexmillane): If the keyframe stops being a baseframe we should mark it as deletable
  //                    again.
  if (it != vpKFs.end()) {
    (*it)->SetNotErase();
    mvPatchKFids.push_back(KFid);
    return true;
  } else {
    // NOTE(alexmillane): This will cause serious issues. We should try to mark
    //                    baseframes as non-deletable to ensure it doesn't.
    std::cout
        << "Requested patch base frame no longer exists in the keyframe list..."
        << std::endl;
    return false;
  }
}

void DenseMappingInterface::removeAllKeyframesAsPatchBaseFrames() {
  mvPatchKFids.clear();
}

bool DenseMappingInterface::getPatchBaseFramePosesAndCovariances(
    std::vector<cv::Mat>& patchPoses,
    std::vector<Eigen::Matrix<double, 6, 6>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>>&
        patchConditionalCovariances) {
  bool poses_found = extractPatchBaseFramePoses(patchPoses);
  bool covariances_found =
      extractPatchBaseFrameConditionalCovariances(patchConditionalCovariances);
  return poses_found && covariances_found;
}

bool DenseMappingInterface::extractPatchBaseFramePoses(
    std::vector<cv::Mat>& patchPoses) {
  // Iterating over the patch base frames
  size_t patchNumber = 0;
  auto itPoseTrajectory = mvPoseTrajectory.begin();
  for (const unsigned long patchKFid : mvPatchKFids) {

    // DEBUG
    std::cout << "Searching for pose of patch: " << patchNumber
              << ", with associated KF: " << patchKFid << std::endl;

    // Incrementing over the bundle adjusted trajectory until we find the pose
    bool KFFound = false;
    while (!KFFound) {

      // DEBUG
      std::cout << "Index of the keyframe checked: "
                << (itPoseTrajectory - mvPoseTrajectory.begin()) << std::endl;

      // Checking were not at the end of available bundle adjusted poses
      if (itPoseTrajectory == mvPoseTrajectory.end()) {
        // NOTE(alexmillane): This is probably the wrong move here.
        // This condition will almost certainly be struck.
        std::cout << "Could not find the patch base frame in the bundle adjusted poses" << std::endl;
        return false;
      }
      // Checking if we have found the keyframe associated with the patch
      if (itPoseTrajectory->id == patchKFid) {
        KFFound = true;
        patchPoses.push_back(itPoseTrajectory->pose);
      }
      // Incrementing and checking the next pose in the trajectory for a match
      itPoseTrajectory++;
    }
    // DEBUG(alexmillane): Counter for display only.
    patchNumber++;
  }
}

// Extracts the conditional covariance of the keyframes on one another
bool DenseMappingInterface::extractPatchBaseFrameConditionalCovariances(
    std::vector<Eigen::Matrix<double, 6, 6>,
                Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>>&
        patchConditionalCovariances) {
  // Clearing the vector and reserving space
  patchConditionalCovariances.clear();
  patchConditionalCovariances.reserve(mvPatchKFids.size() - 1);

  // TODO(alexmillane): Change this flag to indicate simply if a covariance
  // matrix is available
  // Cant do anything if there is no covariance available
  if (!mbTrajectoryCovarianceAvailable) {
    return false;
  }

  // Looping over the registered patch base frame ids
  // NOTE(alexmillane): Skipping the first patch because it has nothing to be relative to it.
  for (size_t patchIdx = 1; patchIdx < mvPatchKFids.size(); patchIdx++) {
    // Extracting the KFids of the patch base frames
    std::array<unsigned long, 2> KFids;
    KFids[0] = mvPatchKFids[patchIdx-1];
    KFids[1] = mvPatchKFids[patchIdx];
    // Getting the joint marginal
    Eigen::Matrix<double, 12, 12> marginalCovariance;
    if (!extractJointMarginalCovarianceOfPair(KFids, marginalCovariance)) {
      std::cout << "Skipping conditional covariance for patch: " << patchIdx
                << std::endl;
      continue;
    }
    // Getting the conditional covariance
    Eigen::Matrix<double, 6, 6> conditionalCovariance;
    extractCoditionalFromMarginalCovarianceOfPair(marginalCovariance,
                                                  conditionalCovariance);
    // Saving into the result vector
    patchConditionalCovariances.push_back(conditionalCovariance);
  }
}

bool DenseMappingInterface::extractJointMarginalCovarianceOfPair(
    const std::array<unsigned long, 2>& KFids,
    Eigen::Matrix<double, 12, 12>& marginalCovariance) {

  // Cant do anything if there is no covariance available
  if (!mbTrajectoryCovarianceAvailable) {
    return false;
  }

  // Getting the KF collumns in the hessian
  std::array<int, 2> hessianIdxs;
  for (size_t KFidx = 0; KFidx < 2; KFidx++) {
    // Getting the KF id
    auto KFid = KFids[KFidx];
    // Finding the patch keyframe in the hessian map
    auto itKFidToHessianCol = mKFidToHessianCol.find(KFid);
    if (itKFidToHessianCol != mKFidToHessianCol.end()) {
      hessianIdxs[KFidx] = itKFidToHessianCol->second;
    } else {
      std::cout
          << "Patch keyframe ID: " << KFid
          << " is not available in the currently available covariance matrix."
          << std::endl;
      return false;
    }
  }

  // Constructing the joint marginal.
  for (size_t row_idx = 0; row_idx < 2; row_idx++) {
    for (size_t col_idx = 0; col_idx < 2; col_idx++) {
      // Getting the hessian indexes
      int hessianRow = hessianIdxs[row_idx];
      int hessianCol = hessianIdxs[col_idx];
      // Filling ouot the marginal covariance
      marginalCovariance.block<6, 6>(6 * row_idx, 6 * col_idx) =
          mpPoseCovariance->block<6, 6>(hessianRow, hessianCol);
    }
  }

  // Success!
  return true;
}

bool DenseMappingInterface::extractCoditionalFromMarginalCovarianceOfPair(
    const Eigen::Matrix<double, 12, 12>& marginalCovariance,
    Eigen::Matrix<double, 6, 6>& conditionalCovariance) {
  // Extracting the submatrices
  // NOTE(alexmillane): Could probably skip some unnessisary copies here by mapping.
  Eigen::Matrix<double, 6, 6> A = marginalCovariance.block<6,6>(0,0);
  Eigen::Matrix<double, 6, 6> B = marginalCovariance.block<6,6>(0,6);
  Eigen::Matrix<double, 6, 6> C = marginalCovariance.block<6,6>(6,6);

  // Calculating the schur's compliment
  // NOTE(alexmillane): There is a more numerically stable way to calculate this
  //                    which avoids the direct inversion of C.
  conditionalCovariance = A - B * C.inverse() * B.transpose();

/*  // DEBUG
  std::cout << "A: " << std::endl << A << std::endl;
  std::cout << "B: " << std::endl << A << std::endl;
  std::cout << "C: " << std::endl << A << std::endl;
  std::cout << "conditionalCovariance: " << std::endl
            << conditionalCovariance << std::endl;
*/
}

}  // namespace ORB_SLAM