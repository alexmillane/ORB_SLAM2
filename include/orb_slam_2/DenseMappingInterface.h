
#ifndef DENSEMAPPINGINTERFACE
#define DENSEMAPPINGINTERFACE

#include <mutex>

#include <Eigen/Dense>

#include "orb_slam_2/Map.h"

/*#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "LoopClosing.h"
#include "Tracking.h"

#include <mutex>*/

namespace ORB_SLAM2 {

// class Tracking;
// class LoopClosing;
class Map;

class DenseMappingInterface {
 public:
  DenseMappingInterface(Map* pMap);

  // Tells the interface that a global bundle adjustment has occurred.
  void notifyFinishedGBA();

  bool isUpdatedTrajectoryAvailable();

  std::vector<Eigen::Affine3d> getUpdatedTrajectory();

 protected:
  // Map
  Map* mpMap;

  // Flag indicating if a new (unfetched) trajectory is available
  bool mbUpdatedTrajectoryAvailable;

  // The trajectory stored by the interface
  std::vector<Eigen::Affine3d> mvPoseTrajectory; 

  // A mutex which locks the trajectory for read/write
  std::mutex mMutexTrajectory;

};

}  // namespace ORB_SLAM

#endif  // DENSEMAPPINGINTERFACE
