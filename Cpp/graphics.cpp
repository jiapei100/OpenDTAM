#include <opencv2/highgui/highgui.hpp>
#include <boost/thread/mutex.hpp>
#include <queue>
#include <string>
#include "set_affinity.h"
#include "utils/ImplThreadLaunch.hpp"
#include "graphics.hpp"


static std::queue<cv::Mat> toShow;
static std::queue<std::string> nameShow;
static std::queue<cv::Vec2d> autoScale;

static std::queue<int> props;
static std::queue<std::string> nameWin;
static boost::mutex Gmux; 
static volatile int ready=0;
static volatile int pausing=0;
int allDie=0;
void gpause(){
    CV_XADD(&pausing,1);
    gcheck();
}
void gcheck(){
    while(ready||CV_XADD(&pausing,0)){
        usleep(100);
        if(allDie)
                    return;
    }
}

void pfShow(const std::string name,const cv::Mat& _mat,int defaultscale, cv::Vec2d autoscale){
    assert(_mat.rows>0 && _mat.cols>0);

    if (defaultscale==1){
        autoscale=cv::Vec2d(-1,-1);
    }
    //cull frames
    Gmux.lock();
    nameShow.push(name);
    toShow.push(_mat.clone());
    autoScale.push(autoscale);
    ready++;
    assert(nameShow.size()==ready);
    
    Gmux.unlock();
    while(nameShow.size()>5||pausing){
        usleep(100);
        if(allDie)
                    return;
    }

}
void pfWindow(const std::string name,int prop){
    Gmux.lock();
    nameWin.push(name);
    props.push(prop);

    Gmux.unlock();
    while(nameWin.size()>5||pausing){
        usleep(100);
        if(allDie)
                    return;
    }
}
template <class T>
static inline T take(std::queue<T>& q){
    T ref=q.front();
    q.pop();
    return ref;
}




void guiLoop(int* die){
    cv::Mat mat;
    while(!*die){
        if (props.size()>0){//deal with new windows
            Gmux.lock();
            std::string name=take(nameWin);
            int prop=take(props);
            
            Gmux.unlock();
            cv::namedWindow(name,prop);
        }
        if (ready){//deal with imshows
            Gmux.lock();
            assert(nameShow.size()>0);
            mat=take(toShow);
            std::string name=take(nameShow);
            cv::Vec2d autoscale=take(autoScale);
            ready--;
            Gmux.unlock();
            if ((autoscale[0]==autoscale[1] && autoscale[0]==0)){
                double min; 
                double max;
                cv::minMaxIdx(mat, &min, &max);
                float scale = 1.0/ (max-min);
                mat.convertTo(mat,CV_MAKETYPE(CV_32F,mat.channels()), scale, -min*scale);
//                 std::cout<<name<<": view scale: "<<max-min<<std::endl;
//                 std::cout<<name<<": min: "<<min<<"  max: "<< max<<std::endl;
            }else if (autoscale[0]!=autoscale[1]){
                double scale= 1.0/(autoscale[1]-autoscale[0]);
                mat.convertTo(mat,CV_MAKETYPE(mat.type(),mat.channels()),scale,-autoscale[0]*scale);
            }
            mat.convertTo(mat,CV_MAKETYPE(CV_8U,mat.channels()), 255.0);//use 8 bit so we can have the nice mouse over
            if(mat.rows<250){
                name+=":small";
                cv::namedWindow(name, CV_WINDOW_KEEPRATIO | CV_GUI_NORMAL);
            }
            imshow( name, mat);
            cv::waitKey(1);//waitkey must occur here so matrix doesn't fall out of scope because imshow is dumb that way :(
//            std::cout<<name<<" std::queue:"<<ready<<std::endl;
        }else if(pausing){
            cv::namedWindow("control",CV_WINDOW_KEEPRATIO);
            std::cout<<"Paused: Space (in GUI window) to continue"<<std::endl;
            while(cv::waitKey()!=' ');
            
            CV_XADD(&pausing,-1);
        }else{
            cv::waitKey(1);
        }
        if(pausing<0){
            pausing=0;
        }
//         cv::waitKey(1);
//         usleep(100);
    }
    allDie=1;
    std::cout<<"Gui Shutting down"<<std::endl;
    cv::waitKey(1);
}
void initGui(){
    ImplThread::startThread(guiLoop,"Graphics"); 
    
}

