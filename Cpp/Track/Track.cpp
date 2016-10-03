#include "Track.hpp"
#include "utils/utils.hpp"

Track::Track(Cost cost){
    rows=cost.rows;
    cols=cost.cols;
    baseImage=lastFrame=thisFrame=cost.baseImage;
    cameraMatrix=cv::Mat(cost.cameraMatrix);
    depth=cost.depthMap();
    PToLie(cv::Mat(cost.pose),basePose);
    pose=basePose.clone();

}
Track::Track(CostVolume cost){
    rows=cost.rows;
    cols=cost.cols;
    cost.baseImage.download(thisFrame);
    baseImage=lastFrame=makeGray(thisFrame);
    cameraMatrix=cv::Mat(cost.cameraMatrix);
    RTToLie(cost.R,cost.T,basePose);
    pose=basePose.clone();
    verbose=0;
}
void Track::addFrame(cv::Mat frame){
    lastFrame=thisFrame;
    thisFrame=makeGray(frame);
}
