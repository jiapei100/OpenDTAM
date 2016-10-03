#include <opencv2/core/core.hpp>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <ctime>
#include <fstream>



//Mine
#include "fileLoader.hpp"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
// #include "OpenDTAM.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"

#include "utils/utils.hpp"


//debug
#include "tictoc.h"





const static bool valgrind=0;

//A test program to make the mapper run
using namespace cv::cuda;

int App_main( int argc, char** argv );

void myExit(){
    ImplThread::stopAllThreads();
}
int main( int argc, char** argv ){

    initGui();

    int ret=App_main(argc, argv);
    myExit();
    return ret;
}


int App_main( int argc, char** argv )
{
    volatile int debug=0; 
    srand(314159);
    rand();
    rand();
    cv::theRNG().state = rand();
    int numImg=50;
    int numFiles=515;

#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(),"App_main");
#endif

    char filename[500];
    cv::Mat image, cameraMatrix, R, T;
    std::vector<cv::Mat> images,Rs,ds,Ts,Rs0,Ts0,D0;
    std::vector<float> key,qual,vis,occ;

    cv::Mat ret;//a place to return downloaded images to
    
    std::ofstream file("outscale.csv");
    
    double reconstructionScale=5/5.;
    int inc=1;
    for(int i=0;i>0||inc>0;i+=inc){
        cv::Mat tmp,d,image;
        int offset=0;
        if(inc>0){
            
        loadAhanda("/home/paulf/Downloads/60fps_images_archieve/",
//         loadAhanda("/home/paulf/Downloads/traj_over_table/",
                   65535,
                   i+offset,
                   image,
                   d,
                   cameraMatrix,
                   R,
                   T);
        tmp=cv::Mat::zeros(image.rows,image.cols,CV_32FC3);
        randu(tmp,0,1);
        cv::cuda::resize(image+tmp/255,image,cv::Size(),reconstructionScale,reconstructionScale);
        images.push_back(image.clone());
        Rs.push_back(R.clone());
        Ts.push_back(T.clone());
        ds.push_back(d.clone());
        Rs0.push_back(R.clone());
        Ts0.push_back(T.clone());
        D0.push_back(1/d);
//         pfShow("load",image,0,Vec2d(0,1));
//          pfShow("depth",1/d*50,0,Vec2d(0,1));
        }
        else
        {
            images.push_back(images[i]);
            Rs.push_back(Rs[i]);
            Ts.push_back(Ts[i]);
            ds.push_back(ds[i]);
            Rs0.push_back(Rs0[i]);
            Ts0.push_back(Ts0[i]);
            D0.push_back(D0[i]);
        }
        
        key.push_back(0);
        qual.push_back(0);
        vis.push_back(0);
        occ.push_back(0);
//         Rs.push_back(cv::Mat());
//         Ts.push_back(cv::Mat());
        
        if(i==numImg-1)
            inc=-1;
    }
    numImg=numImg*2-2;
//     {//random first image
//         std::cout<<LieSub(RTToLie(Rs0[0],Ts0[0]),RTToLie(Rs0[1],Ts0[1]))<<std::endl;
//         Ts[0]=Ts0[1].clone();
//         Ts[1]=Ts0[1].clone();
//         randu(Ts[1] ,Scalar(-1),Scalar(1));
//         Ts[1]=Ts0[0]+Ts[1];
//         std::cout<<Ts[1]-Ts0[0]<<std::endl;
//         Rs[0]=Rs0[0].clone();
//         Rs[1]=Rs0[0].clone();
//     }
//    CudaMem cret(images[0].rows,images[0].cols,CV_32FC1);
//    ret=cret.createMatHeader();
    cv::Mat cret(images[0].rows,images[0].cols,CV_32FC1);
    //Setup camera matrix
    double sx=reconstructionScale;
    double sy=reconstructionScale;
    cameraMatrix+=(cv::Mat)(cv::Mat_<double>(3,3) <<    0.0,0.0,0.5,
                                                0.0,0.0,0.5,
                                                0.0,0.0,0.0);
    cameraMatrix=cameraMatrix.mul((cv::Mat)(cv::Mat_<double>(3,3) <<    sx,0.0,sx,
                                                                0.0,sy ,sy,
                                                                0.0,0.0,1.0));
    cameraMatrix-=(cv::Mat)(cv::Mat_<double>(3,3) <<    0.0,0.0,0.5,
                                                0.0,0.0,0.5,
                                                0.0,0.0,0);
    int layers=128;
    int desiredImagesPerCV=50;
    int imagesPerCV=50;
    int startAt=0;
    float occlusionThreshold=.05;
    Norm norm=L1T;
//     {//offset init
//         Rs[startAt]=Rs[0].clone();
//         Rs[startAt+1]=Rs[1].clone();
//         Ts[startAt]=Ts[0].clone();
//         Ts[startAt+1]=Ts[1].clone();
//     }
    
    CostVolume cv(images[startAt],(FrameID)startAt,layers,0.015,0.0,Rs[startAt],Ts[startAt],cameraMatrix,occlusionThreshold,norm);
    
 
    
    cv::cuda::Stream s;
    double totalscale=1.0;
    int tcount=0;
    int sincefail=0;
    for (int imageNum=(startAt+1)%numImg;imageNum<numImg;imageNum=(imageNum+1)%numImg){
 //       std::cout<<dec; // deleted by Pei JIA
        T=Ts[imageNum].clone();
        R=Rs[imageNum].clone();
        image=images[imageNum];

        if(cv.count<imagesPerCV){
            std::cout<<"using: "<< imageNum<<std::endl;
            cv.updateCost(image, R, T);
            cudaDeviceSynchronize();
//             gpause();
//             for( int i=0;i<layers;i++){
//                 pfShow("layer",cv.downloadOldStyle(i), 0, cv::Vec2d(0, 1));
// //                 usleep(1000000);
//             }

//             volatile int keep=1;
// //             while(cv.count==190&&keep){
//                 std::vector<cv::Mat> slices=cv.download();
//                 for( int i=0;i<cv.rows;i++){
//                     pfShow("cross Section",slices[i],0, cv::Vec2d(0, 1));
// //                     usleep(10000);
// //                     if(cv.count>50||cv.count<3)
// //                         gpause();
// //                 }
//             }
        }
        else{
            
            for(int i0=1;i0<=10-imagesPerCV;i0++){
                int i=((cv.fid-i0)%numImg+numImg)%numImg;
                std::cout<<"using: "<< i<<std::endl;
                cv.updateCost(images[i], Rs[i], Ts[i]);
                cudaDeviceSynchronize();
            }
            
            cudaDeviceSynchronize();
            //Attach optimizer
            cv::Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(cv.baseImageGray,cv.cvStream);
            DepthmapDenoiseWeightedHuber& denoiser=*dp;
            Optimizer optimizer(cv);
            optimizer.initOptimization();
            GpuMat a(cv.loInd.size(),cv.loInd.type());
             cv.loInd.copyTo(a,cv.cvStream);
//            cv.cvStream.enqueueCopy(cv.loInd,a);
            GpuMat d;
            denoiser.cacheGValues();
            ret=image*0;
//             pfShow("A function", ret, 0, cv::Vec2d(0, layers));
//             pfShow("D function", ret, 0, cv::Vec2d(0, layers));
//             pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));
//             pfShow("Predicted Image",ret,0,Vec2d(0,1));
//             pfShow("Actual Image",ret);
            
           
//                waitKey(0);
//                gpause();
            
            

            bool doneOptimizing; int Acount=0; int QDcount=0;
            do{
//                 std::cout<<"Theta: "<< optimizer.getTheta()<<std::endl;
//
//                 if(Acount==0)
//                     gpause();
//                a.download(ret);
//                pfShow("A function", ret, 0, cv::Vec2d(0, layers));
                
                

                for (int i = 0; i < 10; i++) {
                    d=denoiser(a,optimizer.epsilon,optimizer.getTheta());
                    QDcount++;
                    
//                    denoiser._qx.download(ret);
//                    pfShow("Q function:x direction", ret, 0, cv::Vec2d(-1, 1));
//                    denoiser._qy.download(ret);
//                    pfShow("Q function:y direction", ret, 0, cv::Vec2d(-1, 1));
//                     d.download(ret);
//                     pfShow("D function", ret, 0, cv::Vec2d(0, layers));
//                     gpause();
//                     if(QDcount<2){
//                         medianBlur(ret,ret,3);
//                         d.upload(ret);
//                     }
                }
                doneOptimizing=optimizer.optimizeA(d,a);
                Acount++;
//                 d.download(ret);
//                 pfShow("D function", ret, 0, cv::Vec2d(0, layers));
            }while(!doneOptimizing);
//             optimizer.lambda=.05;
//             optimizer.theta=10000;
//             optimizer.optimizeA(a,a);
            optimizer.cvStream.waitForCompletion();
             cv.lo.download(ret);
            pfShow("loVal", ret, 0, cv::Vec2d(0, 3));
             cv.loInd.download(ret);
            pfShow("loInd", ret, 0, cv::Vec2d(0, layers));
            std::cout<<"diffN0:"<<cv::cuda::sum(abs(D0[cv.fid]-(ret*cv.depthStep+cv.far)))<<std::endl;
            std::cout<<"diffN0Med:"<<median(abs(D0[cv.fid]-(ret*cv.depthStep+cv.far)))<<std::endl;
            medianBlur(ret,ret,3);
            medianBlur(ret,ret,3);
            pfShow("loIndMed",ret, 0, cv::Vec2d(0, layers));
            std::cout<<"diffMedN0:"<<cv::cuda::sum(abs(D0[cv.fid]-(ret*cv.depthStep+cv.far)))<<std::endl;
            std::cout<<"diffMedN0Med:"<<median(abs(D0[cv.fid]-(ret*cv.depthStep+cv.far)))<<std::endl;
            a.download(ret);
            pfShow("A function loose", ret, 0, cv::Vec2d(0, layers));
            cv::Mat diff=ret.clone();
            cv.loInd.download(ret);
            diff-=ret;
            pfShow("difference by reg", diff, 0, cv::Vec2d(-layers, layers));
            pfShow("true Depth", D0[cv.fid], 0, cv::Vec2d(0, cv.near));
            pfShow("diff", D0[cv.fid]-optimizer.depthMap(), 0, cv::Vec2d(-.005,.005));
            std::cout<<"diff:"<<cv::cuda::sum(abs(D0[cv.fid]-optimizer.depthMap()))<<std::endl;
            std::cout<<"diffMed:"<<median(abs(D0[cv.fid]-optimizer.depthMap()))<<std::endl;
            
//                gpause();
//             std::cout<<"A iterations: "<< Acount<< "  QD iterations: "<<QDcount<<std::endl;
//             pfShow("Depth Solution", optimizer.depthMap(), 0, cv::Vec2d(cv.far, cv.near));
//             imwrite("outz.png",ret);

//             Track tracker(cv);
//             cv::Mat out=optimizer.depthMap();
//             
// //             sprintf(filename,"/groundtruth/depth_%03d.png",cv.fid);
// //             cv::Mat out16;
// //             out16=1/out;
// //             out16.convertTo(out16,CV_16UC3,10);
// //             std::cout<<"Mean:"<<mean(out16)[0]<<std::endl;
// //             imwrite(filename,out16);
//             
// //             if (tcount==3){
// //                 out=cv.near-out;
// //             }
//             double m;
//             minMaxLoc(out,NULL,&m);
//             m=mean(out)[0];
//             
//                 
            double sf=1;//(.25*cv.near/m);
//             if(!(sf<100&&sf>.001)){
// //                 file<<sf<<", fail!, "<<std::endl;
//                 std::cout<<"FAIL CV #: "<<tcount<<" sf: "<<sf<<std::endl;
//                 if(sf>100||sf<.001)
//                     sf=1.0+.1-.2*(sf<1.0);
// //                 gpause();
//             }
//             tracker.depth=out;
// //             medianBlur(out,tracker.depth,3);
// //             if(imageNum>180)
// //             imageNum=((imageNum-imagesPerCV-2)%numImg+numImg)%numImg;
// //             else if (tcount>6)
            int ni=0;
//             ni=imagesPerCV;
//             ni=min(ni,imagesPerCV-3);
            ni=cv::max(ni,1);
            if(tcount==0)
                ni=1;
            imageNum=((imageNum-imagesPerCV-1+ni)%numImg+numImg)%numImg;
//             tracker.thisFrame=makeGray(images[imageNum]);
//             tracker.pose=RTToLie(Rs[cv.fid],Ts[cv.fid]);
//             tracker.pose=RTToLie(Rs[imageNum],Ts[imageNum]);
//                 if(imageNum<185)
//                 imageNum=180;
            
//             imageNum=30;
//             assert(imageNum>=0);
// //             if (imageNum>5)
// //                 if(imagesPerCV==1)
//             
//             imagesPerCV=desiredImagesPerCV;
// //                 else
// //                     imagesPerCV=1;
//             sincefail++;
            
//             for(int i0=0;i0<=imagesPerCV;i0++){
//                 int i=((imageNum+i0)%numImg+numImg)%numImg;
//                 tracker.addFrame(images[i]);
//                 if(!tracker.align()){
//                     int neg=((i-1)%numImg+numImg)%numImg;
//                     if(i0<2 && ((i-cv.fid)%numImg+numImg)%numImg==1){//failed to align the next image and at limit on frames
//                         std::cout<<"FAil: "<<i<<" on: "<<cv.fid<<std::endl;
// //                         tracker.verbose=1;
// //                         tracker.pose=RTToLie(Rs[neg],Ts[neg]);
// //                         tracker.thisFrame=images[neg];
// //                         tracker.addFrame(images[i]);
// //                         tracker.align();
// //                         tracker.verbose=0;
//                         pfShow("FAILED",images[i]);
//                     }
//                     imagesPerCV=max(i0-1,1);
// //                     if(i0==0&&sincefail>4){
// //                         std::cout<<"TRACKFAIL! RESTART RANDOM"<<std::endl;
// //                         sf=cv.near/.15;//failed so bad we need a new start
// // //                         randu(tracker.depth ,Scalar(0),Scalar(.15));
// //                         tracker.depth=.10;
// //                         tracker.pose=RTToLie(Rs[i-1],Ts[i-1]);
// //                         tracker.align();
// //                         sincefail=0;
// //                         Ts[i]=Ts[(i-1+numImg)%numImg].clone();
// //                         randu(Ts[i] ,Scalar(-1),Scalar(1));
// //                         Ts[i]=Ts[(i-1+numImg)%numImg]+Ts[i];
// //                         Rs[i]=Rs[(i-1+numImg)%numImg].clone();
// // //                         goto skip;
// //                     }
//                 }
//                
//                 LieToRT(tracker.pose,R,T);
//                 if(tracker.quality>qual[i]){
//                     Rs[i]=R.clone();
//                     Ts[i]=T.clone();
//                     qual[i]=tracker.quality;
//                     vis[i]=tracker.coverage;
//                     occ[i]=tracker.occlusion;
//                 }else{
//                     tracker.pose=RTToLie(Rs[i],Ts[i]);
//                 }
// 
//                 skip:
//                 cv::Mat p,tp;
//                 p=tracker.pose;
//                 tp=RTToLie(Rs0[i],Ts0[i]);
// // //                 {//debug
// // //                     std::cout << "True Pose: "<< tp << std::endl;
// // //                     std::cout << "True Delta: "<< LieSub(tp,tracker.basePose) << std::endl;
//                     std::cout << "Recovered Pose: "<< p << std::endl;
// // //                     std::cout << "Recovered Delta: "<< LieSub(p,tracker.basePose) << std::endl;
// // //                     std::cout << "Pose Error: "<< p-tp << std::endl;
// // //                 }
//                 
//                 std::cout<<"<"<<cv.fid<<", "<<i<<">"<<std::endl;
// //                 cv::Mat tran1=cv::Mat::eye(4,4,CV_64FC1);
// //                 ((cv::Mat)(cv::Mat_<double>(4,1) <<    0,0,-1.0/m,1)).copyTo(tran1.col(3));
// //                 cv::Mat rotor=make4x4(rodrigues((cv::Mat)(cv::Mat_<double>(3,1) << 0,-45,0)*3.1415/180.0));
// //                 cv::Mat tran2=cv::Mat::eye(4,4,CV_64FC1);
// //                 ((cv::Mat)(cv::Mat_<double>(4,1) <<    0,0,3/m,1)).copyTo(tran2.col(3));
// //                 cv::Mat view=tran2*rotor*tran1;
//                 cv::Mat basePose=make4x4(RTToP(Rs[cv.fid],Ts[cv.fid]));
//                 cv::Mat basePose0=make4x4(RTToP(Rs0[cv.fid],Ts0[cv.fid]));
//                 cv::Mat foundPose=make4x4(RTToP(R,T));
// // //                 std::cout<<"view:\n"<< fixed << setprecision(3)<< view<<std::endl;
//                 cv::Mat view=diagnosticInfo(images[i],images[cv.fid],tracker.depth,basePose,foundPose,cameraMatrix);
// //                 cv::Mat viewc=diagnosticInfo(images[i],images[cv.fid],tracker.depth,basePose0,make4x4(RTToP(Rs0[i],Ts0[i])),cameraMatrix);
// //                 for(int j=0;j<5;j++){
// //                     cv::Mat tmp;
// //                     pfShow("Predicted Image",view,0,Vec2d(0,1));
// //                     absdiff(images[i],view,tmp);
// //                     pfShow("difftrk",tmp,0,Vec2d(0,1));
// //                     if(tracker.quality<.75 &&i0==-1)
// //                         gpause();
// //                     pfShow("Predicted Image",viewc,0,Vec2d(0,1));
// //                     absdiff(images[i],viewc,tmp);
// //                     pfShow("difftrk",tmp,0,Vec2d(0,1));
// //                     if(tracker.quality<.75 &&i0==-1)
// //                         gpause();
// //                 }
// //                 tracker.pose=tp;
//             }
            
//             if (tcount>6&&imagesPerCV>20)
//             {
//                 int jump=imagesPerCV*2/3;
//                 imageNum=(imageNum+jump)%numImg;
//                 imagesPerCV-=jump;
//                 assert(imagesPerCV>0);
//             }
            
//             for (int a=0;a<360;a+=1)
//             {
//             cv::Mat tran1=cv::Mat::eye(4,4,CV_64FC1);
//             ((cv::Mat)(cv::Mat_<double>(4,1) <<    0,-100,-400,1)).copyTo(tran1.col(3));
//             cv::Mat rotor=make4x4(rodrigues((cv::Mat)(cv::Mat_<double>(3,1) << 0,a,0)*3.1415926535/180.0)*
//                 rodrigues((cv::Mat)(cv::Mat_<double>(3,1) << 90,0,0)*3.1415926535/180.0)
//             );
//             cv::Mat tran2=cv::Mat::eye(4,4,CV_64FC1);
//             ((cv::Mat)(cv::Mat_<double>(4,1) <<    0,0,3/m,1)).copyTo(tran2.col(3));
//             cv::Mat view=tran2*rotor*tran1;
//             cv::Mat basePose=make4x4(RTToP(Rs[cv.fid],Ts[cv.fid]));
//             cv::Mat foundPose=make4x4(RTToP(Rs[imageNum],Ts[imageNum]));
//             diagnosticInfo(images[imageNum],images[cv.fid],tracker.depth,basePose,view,cameraMatrix);
//             }

//             if(imageNum>numImg/2+1)
//                 goto exit;

            cv=CostVolume(images[imageNum],(FrameID)imageNum,layers,cv.near/sf,0.0,Rs[imageNum],Ts[imageNum],cameraMatrix,occlusionThreshold,norm);
            key[imageNum]=tcount;
            
            
            
            totalscale*=sf;
            file<<imageNum<<", "<<sf<<", "<<imagesPerCV<<std::endl;
//             file.sync_with_stdio();
            if(tcount==7){
                totalscale=1.0f;
            }
            tcount++;
            std::cout<<"CV #: "<<tcount<<" Total Scale: "<<totalscale<<std::endl;
            s=optimizer.cvStream;
//             for (int imageNum=0;imageNum<numImg;imageNum=imageNum+1){
//                 reprojectCloud(images[imageNum],images[0],optimizer.depthMap(),RTToP(Rs[0],Ts[0]),RTToP(Rs[imageNum],Ts[imageNum]),cameraMatrix);
//             }
            
//             gpause();
        }
        s.waitForCompletion();// so we don't lock the whole system up forever
    }
    exit:
    file<<"Key_is_keyframes"<<std::endl;
    file<<"Occlusion_is_frac_lost_by_occlusion"<<std::endl;
    file<<"Coverage_is_frac_not_lost_by_occlusion_or_out_of_frame"<<std::endl;
    file<<"Quality_is_frac_of_covered_that_matches"<<std::endl;
    file<<"Key,Quality,Coverage,Occlusion"<<std::endl;
    for(int i=0;i<numImg;i++)
        file<<key[i]<<","<<qual[i]<<","<<vis[i]<<","<<occ[i]<<std::endl;
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
    return 0;
}


