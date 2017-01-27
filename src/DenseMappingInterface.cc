#include <iostream>

#include <geometry_msgs/Pose.h>
#include <eigen_conversions/eigen_msg.h>

#include "orb_slam_2/DenseMappingInterface.h"
#include "orb_slam_2/Converter.h"

namespace ORB_SLAM2
{

DenseMappingInterface::DenseMappingInterface(Map *pMap):
    mpMap(pMap), mbUpdatedTrajectoryAvailable(false)
{
    //
}

void DenseMappingInterface::notifyFinishedGBA() {
    // DEBUG
    std::cout << "Bundle Adjustment finsihed. Storing pose trajectory." << std::endl;

    // Locking the trajectory to overwrite
    unique_lock<mutex> lock(mMutexTrajectory);

    // Looping over the keyframes and storing poses
    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    mvPoseTrajectory.reserve(vpKFs.size());
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];
        cv::Mat Twc = pKF->GetPose();
        mvPoseTrajectory.push_back(Converter::toAffine3d(Twc));
    }

    // Publishing
    //geometry_msgs::Pose msg;
    //tf::poseEigenToMsg (Converter::toAffine3d(Twc), msg);

/*    std::cout << "Twc_cv Open CV: " << Twc_cv << std::endl;
    std::cout << "Twc_cv Eigen: " << Twc_eig.matrix() << std::endl;
*/
}

bool DenseMappingInterface::isUpdatedTrajectoryAvailable() {
    return mbUpdatedTrajectoryAvailable;
}

std::vector<Eigen::Affine3d> DenseMappingInterface::getUpdatedTrajectory() {
    // Locking the trajectory to overwrite
    unique_lock<mutex> lock(mMutexTrajectory);
    return mvPoseTrajectory;
}

} //namespace ORB_SLAM