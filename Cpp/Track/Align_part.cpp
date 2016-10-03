/*
 *  Align_part.cpp
 *  
 *
 *
 */
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/utils.hpp"
#include "graphics.hpp"
#include "Track.hpp"
#include "stdio.h"
#include <iomanip>
#include <opencv2/core/core.hpp>


//debug
// #define QUIET_DTAM 1
// #include "quiet.hpp"


const static float FAIL_FRACTION=0.70;

enum alignment_modes{CV_DTAM_REV,CV_DTAM_FWD,CV_DTAM_ESM};
const double small0=.1;//~6deg, not trivial, but hopefully enough to make the translation matter

static void getGradient(const cv::Mat& image,cv::Mat & grad);




//cv::Mat& reprojectWithDepth(const cv::Mat& T,
//                        const cv::Mat& d,
//                        const cv::Mat& I,
//                        const cv::Mat& cameraMatrix,//cv::Mat_<double>
//                        const cv::Mat& _p,          //cv::Mat_<double>
//                        int mode){
//
//}

static cv::Mat paramsToProjection(const cv::Mat & p,const cv::Mat& _cameraMatrix){
    //Build the base transform
    assert(p.type()==CV_64FC1);
    cv::Mat dR=rodrigues(p.colRange(cv::Range(0,3)));
    cv::Mat dT=p.colRange(cv::Range(3,6)).t();
    cv::Mat dA;
    hconcat(dR,dT,dA);
    dA=make4x4(dA);
    cv::Mat cameraMatrix=make4x4(_cameraMatrix);
    assert(cameraMatrix.type()==CV_64FC1);
    cv::Mat proj=cameraMatrix*dA*cameraMatrix.inv();
//     std::cout<<"p: "<<"\n"<< p<< std::endl;
//     std::cout<<"Proj: "<<"\n"<< proj<< std::endl;
    //The column swap
    cv::Mat tmp=proj.colRange(2,4).clone();
    tmp.col(1).copyTo(proj.col(2));
    tmp.col(0).copyTo(proj.col(3));
    //The row drop
    proj=proj.rowRange(0,3);
    return proj;
}

static void getGradient(const cv::Mat& image,cv::Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    cv::Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cv::cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    cv::Mat grad_x(image.rows,image.cols,CV_32FC1,grad.row(0).data);
    Scharr( gray, grad_x, CV_32FC1, 1, 0, 1.0/26.0, 0, cv::BORDER_REPLICATE );
    cv::Mat grad_y(image.rows,image.cols,CV_32FC1,grad.row(1).data);
    Scharr( gray, grad_y, CV_32FC1, 0, 1, 1.0/26.0, 0, cv::BORDER_REPLICATE);
}

static void getGradient_8(const cv::Mat& image,cv::Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    cv::Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cv::cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    cv::Mat grad_x(image.rows,image.cols,CV_32FC1,grad.row(0).data);
    cv::Scharr( gray, grad_x, CV_32FC1, 1, 0, 1.0/26.0, 0, cv::BORDER_REPLICATE );
    cv::Mat grad_y(image.rows,image.cols,CV_32FC1,grad.row(1).data);
    cv::Scharr( gray, grad_y, CV_32FC1, 0, 1, 1.0/26.0, 0, cv::BORDER_REPLICATE);
}

static void getGradientInterleave(const cv::Mat& image,cv::Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    cv::Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cv::cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    cv::Mat gradX(image.rows,image.cols,CV_32FC1);
    Scharr( gray, gradX, CV_32FC1, 1, 0, 1.0/26.0, 0, cv::BORDER_REPLICATE );
    cv::Mat gradY(image.rows,image.cols,CV_32FC1);
    Scharr( gray, gradY, CV_32FC1, 0, 1, 1.0/26.0, 0, cv::BORDER_REPLICATE);
    cv::Mat src [2]={gradY,gradX};
    cv::merge(src,2,grad);
}

static void Mask(const cv::Mat& in,const cv::Mat& m,cv::Mat& out){
    cv::Mat tmp;
    
    m.convertTo(tmp,in.type());
    out=in.mul(tmp/255);
}

int Track::align_level_largedef_gray_forward(const cv::Mat& T,//Total Mem cost ~185 load/stores of image
                          const cv::Mat& d,
                          const cv::Mat& _I,
                          const cv::Mat& cameraMatrix,//cv::Mat_<double>
                          const cv::Mat& _p,                //cv::Mat_<double>
                          int mode,
                          float threshold,
                          int numParams
                                      )
{
//     cv::Mat result;
//     cv::matchTemplate( _I, T(cv::Range(5,10),cv::Range(5,15)), result, 0 );
//     
//     pfShow("soln",result);
    int ret=1;
    int r=_I.rows;
    int rows=r;
    int c=_I.cols;
    int cols=c;
    const float small=small0;
    //Build the in map (Mem cost 3 layer store:3)
    cv::Mat_<cv::Vec3f> idMap3;
    {
        idMap3.create(r,c);//[rows][cols][3]
        float* id3=(float*) (idMap3.data);
        float* dp=(float*) (d.data);
        int offset=0;
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++,offset++){
                id3[offset*3+0]=j;
                id3[offset*3+1]=i;
                id3[offset*3+2]=dp[offset];
            }
        }
    }
    
    //Build the unincremented transform: (Mem cost 2 layer store,3 load :5)
    cv::Mat baseMap(rows,cols,CV_32FC2);
    {
        cv::Mat tmp=_p.clone();
        cv::Mat baseProj=paramsToProjection(_p,cameraMatrix);
        cv::perspectiveTransform(idMap3,baseMap,baseProj);
        assert(baseMap.type()==CV_32FC2);
    }
    
    {
        //do z buffered occlusion test(approximate depth with original depth)
        cv::Mat xy;
        baseMap.convertTo(xy,CV_32SC2);
        cv::Mat_<float> zmap(r,c,-1.0/0.0);
        int* xyd=(int *)(xy.data);
        float* dp=(float*) (d.data);
        float* bp=(float*) (baseMap.data);
        for(int i=0, offset=0;i<r;i++){
            for(int j=0;j<c;j++,offset++){
                int x=xyd[offset*2+0];
                int y=xyd[offset*2+1];
                if(x>=0&&y>=0&&x<c&&y<r){
                    float oldz=zmap(y,x);
                    float newz=dp[offset];
                    if(newz>oldz){
                        zmap(y,x)=newz;
                    }
                }
            }
        }
        int fail=0;
        for(int i=0, offset=0;i<r;i++){
            for(int j=0;j<c;j++,offset++){
                int x=xyd[offset*2+0];
                int y=xyd[offset*2+1];
                if(x>=0&&y>=0&&x<c&&y<r){
                    float oldz=zmap(y,x);
                    float newz=dp[offset];
                    if(oldz-newz>0){
                        bp[offset*2+0]=-10;
                        bp[offset*2+1]=-10;
                        fail++;
                    }
                }
            }
        }
        occlusion=((double)fail)/(r*c);
    }
    
    
    // reproject the gradient and image at the same time (Mem cost >= 24)
    cv::Mat gradI;
    cv::Mat I(r,c,CV_32FC1);
    {
        getGradient(_I,gradI); //(Mem cost: min 2 load, 2 store :4)
        cv::Mat toMerge[3]={_I,
                        cv::Mat(r,c,CV_32FC1,(float*)gradI.data),
                        cv::Mat(r,c,CV_32FC1,((float*)gradI.data)+r*c)};
        cv::Mat packed;
        cv::merge(toMerge,3,packed); //(Mem cost: min 3 load, 3 store :6)
        cv::Mat pulledBack;
        
        remap( packed, pulledBack, baseMap,cv::Mat(), CV_INTER_LINEAR, cv::BORDER_CONSTANT,0.0 );//(Mem cost:?? 5load, 3 store:8)
        gradI.create(r,c,CV_32FC2);

        int from_to[] = { 0,0, 1,1, 2,2 };
        cv::Mat src[1]=pulledBack;
        cv::Mat dst[2]={I,gradI};
        
        cv::mixChannels(src,1,dst,2,from_to,3);// extract the image and the resampled gradient //(Mem cost: min 3 load, 3 store :6)
    }
    
    // Calculate the differences and build mask for operations (Mem cost ~ 8)
    cv::Mat fit;
    cv::absdiff(T,I,fit);
    cv::Mat mask=(fit<threshold)&(I>0);
    cv::Mat mask2=(I>0);
    double vis=cv::countNonZero(mask2)/(rows*cols*1.0f);
    double good=cv::countNonZero(mask)/(rows*cols*1.0f);
    double qual=good/vis;
    if(verbose){
    std::cout<<"Visibility: "<<vis;
    std::cout<<" Accepted: "<<good<<" Quality: "<<qual<<" r: "<<rows<<std::endl;
    }
    coverage=vis;
    quality=qual;
    if(!(qual>=FAIL_FRACTION)||!(vis>=.3)){//tracking failed!
        ret=0;
    }
    
    cv::Mat err=T-I;
    
    //debug
    if (verbose){

        pfShow("Before iteration",_I,0,cv::Vec2d(0,1));
        pfShow("After Iteration",I,0,cv::Vec2d(0,1));
        cv::Mat tmp;
        Mask(I,fit<threshold,tmp);
        pfShow("Tracking Stabilized With Occlusion",tmp,0,cv::Vec2d(0,1));
        pfShow("To match",T,0,cv::Vec2d(0,1));
        gpause();
    }

    
   
    
    
    
    // Build Jacobians:
    cv::Mat Jsmall;
    Jsmall.create(numParams,rows*cols,CV_32FC1);
    int OM_OFFSET=0;//128;//offset helps keep cache from being clobbered by load/stores
    cv::Mat outContainer;
    outContainer.create(numParams,rows*cols*2+OM_OFFSET,CV_32FC1);
    
    //TODO: Whole loop cacheable except J multiplies if CV_DTAM_REV (Mem cost whole loop 17/itr: 102)
    for (int paramNum=0; paramNum<numParams; paramNum++) {
        
        
        
        //Build the incremented transform
        assert(_p.type()==CV_64FC1);
        cv::Mat_<double> p=_p.clone();
        p(0,paramNum)+=small;
        cv::Mat proj=paramsToProjection(p,cameraMatrix);
        
        //get a row of dmap/dp
        cv::Mat outMap(rows,cols,CV_32FC2,((float*)outContainer.data)+rows*cols*2*paramNum+OM_OFFSET);
        
        perspectiveTransform(idMap3,outMap,proj);//outmap=baseMap+dMap/dp*small (Mem cost 5)
        
        
        //subtract off the base to make a differential  (Mem cost 6)
        //this cannot be done below in the J loop because it would need 5 pointers
        // which is bad for cache (4-way set associative)
//         cv::Mat t1,t2;
//         cv::Mat tmp[2]={t1,t2};
//         split(outMap,tmp);
//         char s[500];
//         sprintf(s,"diff0:%d",paramNum);
//         pfShow(s,tmp[0]);
//         pfShow("diff1",tmp[1]);
        outMap-=baseMap;//outmap=dMap/dp*small
//         split(outMap,tmp);
//         pfShow("diff2",tmp[0]);
//         pfShow("diff3",tmp[1]);
        //multiply by appropriate gradient
        
        
        //want:J*small=dI/dMap*dMap/dp*small
        //do: Jsmall=sumChannels((outmap-idMap2).mul(cv::merge(gradient[0],gradient[1])))
        const float * om=(const float*) (outMap.data);
        const float * bm=(const float*) (baseMap.data);
        const float * gi=(const float*) (gradI.data);
        const uchar * m = mask.data;
        float* Jp=(float*) (Jsmall.row(paramNum).data);//the row of the jacobian we're computing
        int offset = 0;
        //TODO: this loop may work faster unrolled or hand sse/avx optimized (Mem cost 6)
        //Warning this loop uses all 4 mappings in a 4 way cache!
        //Unrolling to cache line size would allow a 5th pointer to be used.
        for(float i=0;i<rows;i++){
            for(float j=0;j<cols;j++,offset++){
                float jx,jy;
                jx = om[offset*2+0]*gi[offset*2+0];
                jy = om[offset*2+1]*gi[offset*2+1];
                Jp[offset]=m[offset]?jx+jy:0;
            }
        }
        //usleep(1000000);
    }
    //now want: dp=(J'J)^-1*J'*(T-I)
    //          dp=small*(Jsmall*Jsmall')^-1*Jsmall*(T-I) since Jsmall is already transposed
    //          dp=small*Hsmallsmall^-1*Jsmall*(T-I)
    cv::Mat Hss=Jsmall*Jsmall.t(); //Hessian (numParams^2) (Mem cost 6-36 depending on cache)
    Hss.convertTo(Hss,CV_64FC1);
    cv::Mat Hinv=small*Hss.inv(cv::DECOMP_SVD);  //TODO:cacheable for CV_DTAM_REV 
    Hinv.convertTo(Hinv,CV_32FC1);
    err=err.reshape(0,r*c);

    cv::Mat dp=(Hinv*(Jsmall*err)).t();//transpose because we decided that p is row vector (Mem cost 7)
    dp.convertTo(dp,CV_64FC1);
//     std::cout<<"Je: \n"<<Jsmall*err<<std::endl;
//     std::cout<<"H: "<<"\n"<< Hss<< std::endl;
//     std::cout<<"Hinv: "<<"\n"<< Hinv<< std::endl;
//     std::cout<<"dp: "<<"\n"<< dp<< std::endl;
    
    
    //Check amount of motion
    {
        
    }
    
    //Check error
    //For the pixels that are within threshold, the average error should go down (Expensive!)
//     {
//         cv::Mat tmp=_p.clone();
//         tmp.colRange(0,numParams)+=dp;
//         cv::Mat newMap,newBack;
//         cv::Mat newProj=paramsToProjection(tmp,cameraMatrix);
//         perspectiveTransform(idMap3,newMap,newProj);
//         remap( _I, newBack, newMap, cv::Mat(), CV_INTER_LINEAR, cv::BORDER_CONSTANT,-1.0/0.0 );
//         cv::Mat newFit;
//         absdiff(T,newBack,newFit);
//         cv::Mat fitDiff;
//         subtract(fit,newFit,fitDiff,mask & (newBack>0));
//         double deltaErr=sum(fitDiff)[0];
//         std::cout<<"Delta Err: "<< deltaErr<<std::endl;
//         if (deltaErr<0)
//             return false;
//     }
    cv::Mat tmp=_p.clone()*0;
    tmp.colRange(0,numParams)+=dp;
    dp=tmp.clone();
    float dmax=.5;//assume ~1 radian field of view
    
    tmp.colRange(3,6)*=.01;
    double max;
    tmp=tmp*cameraMatrix.at<double>(0,0);
//     std::cout<<fixed<<setprecision(8)<<"bound: "<<tmp<<std::endl;
    minMaxLoc(abs(tmp),NULL,&max);
    if(max>dmax){
        dp/=max/dmax;
        ret*=2;
    }
    tmp=_p.clone()*0;
    tmp+=dp;
    dp=tmp.clone();
    
    tmp.colRange(3,6)*=.01;
    tmp=tmp*cameraMatrix.at<double>(0,0);
//     std::cout<<fixed<<setprecision(8)<<"bound: "<<tmp<<std::endl;
    
    _p+=dp;
    
    
    return ret;
}
