#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#ifndef DTAM_UTILS_HPP
#define DTAM_UTILS_HPP

static cv::Mat  makeGray(cv::Mat image){
    if (image.channels()!=1) {
        cvtColor(image, image, CV_BGR2GRAY);
    }
    return image;
}

static cv::Mat make4x4(const cv::Mat& mat){
    
    if (mat.rows!=4||mat.cols!=4){
        cv::Mat tmp=cv::Mat::eye(4,4,mat.type());
        tmp(cv::Range(0,mat.rows),cv::Range(0,mat.cols))=mat*1.0;

        return tmp;
    }else{
        return mat;
    }
}

static cv::Mat rodrigues(const cv::Mat& p){
    
    cv::Mat tmp;
    Rodrigues(p,tmp);
    return tmp;
}

static void LieToRT(cv::InputArray Lie, cv::OutputArray _R, cv::OutputArray _T){
    cv::Mat p = Lie.getMat();
    _R.create(3,3,CV_64FC1);
    cv::Mat R = _R.getMat();
    _T.create(3,1,CV_64FC1);
    cv::Mat T = _T.getMat();
    if(p.cols==1){
        p = p.t();
    }
        
    rodrigues(p.colRange(cv::Range(0,3))).copyTo(R);
    cv::Mat(p.colRange(cv::Range(3,6)).t()).copyTo(T);


}


static void RTToLie(cv::InputArray _R, cv::InputArray _T, cv::OutputArray Lie ){

    cv::Mat R = _R.getMat();
    cv::Mat T = _T.getMat();
    Lie.create(1,6,T.type());
    
    cv::Mat p = Lie.getMat(); 
    assert(p.size()==cv::Size(6,1));
    p=p.reshape(1,6);
    if(T.rows==1){
        T = T.t();
    }
    
    rodrigues(R).copyTo(p.rowRange(cv::Range(0,3)));
    T.copyTo(p.rowRange(cv::Range(3,6)));
    assert(Lie.size()==cv::Size(6,1));
}
static cv::Mat RTToLie(cv::InputArray _R, cv::InputArray _T){

    cv::Mat P;
    RTToLie(_R,_T,P);
    return P;
}
static void PToLie(cv::InputArray _P, cv::OutputArray Lie){

    cv::Mat P = _P.getMat();
    assert(P.cols == P.rows && P.rows == 4);
    cv::Mat R = P(cv::Range(0,3),cv::Range(0,3));
    cv::Mat T = P(cv::Range(0,3),cv::Range(3,4));
    RTToLie(R,T,Lie);
    assert(Lie.size()==cv::Size(6,1));
}
static void RTToP(cv::InputArray _R, cv::InputArray _T, cv::OutputArray _P ){
    
    cv::Mat R = _R.getMat();
    cv::Mat T = _T.getMat();
    cv::Mat P = _P.getMat();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}
static cv::Mat RTToP(cv::InputArray _R, cv::InputArray _T){
    
    cv::Mat R = _R.getMat();
    cv::Mat T = _T.getMat();
    cv::Mat P;
    hconcat(R,T,P);
    make4x4(P);
    return P;
}
static void LieToP(cv::InputArray Lie, cv::OutputArray _P){
    cv::Mat p = Lie.getMat();
    _P.create(4,4,p.type());
    cv::Mat P = _P.getMat();
    if(p.cols==1){
        p = p.t();
    } 
    
    cv::Mat R=rodrigues(p.colRange(cv::Range(0,3)));
    cv::Mat T=p.colRange(cv::Range(3,6)).t();
    hconcat(R,T,P);
    make4x4(P).copyTo(_P);
}
static cv::Mat LieToP(cv::InputArray Lie){
    cv::Mat P;
    LieToP(Lie,P);
    return P;
}

static cv::Mat LieSub(cv::Mat A, cv::Mat B){
    cv::Mat Pa;
    cv::Mat Pb;
    LieToP(A,Pa);
    LieToP(B,Pb);
    cv::Mat out;
    assert(A.size()==cv::Size(6,1) && B.size()==cv::Size(6,1));
    PToLie(Pa*Pb.inv(),out);
    return out;
}

static cv::Mat LieAdd(cv::Mat A, cv::Mat B){
    cv::Mat Pa;
    cv::Mat Pb;
    
    LieToP(A,Pa);
    LieToP(B,Pb);
    cv::Mat out;
    PToLie(Pa*Pb,out);
    return out;
}

template<class tp>
tp median_(const cv::Mat& _M) {
    cv::Mat M=_M.clone();
    int iSize=M.cols*M.rows;
    tp* dpSorted=(tp*)M.data;
    // Allocate an array of the same size and sort it.
    
    std::sort (dpSorted, dpSorted+iSize);

    // Middle or average of middle values in the sorted array.
    tp dMedian = 0.0;
    if ((iSize % 2) == 0) {
        dMedian = (dpSorted[iSize/2] + dpSorted[(iSize/2) - 1])/2.0;
    } else {
        dMedian = dpSorted[iSize/2];
    }
    return dMedian;
}

static double median(const cv::Mat& M) {
    if(M.type()==CV_32FC1)
        return median_<float>(M);
    if(M.type()==CV_64FC1)
        return median_<double>(M);
    if(M.type()==CV_32SC1)
        return median_<int>(M);
    if(M.type()==CV_16UC1)
        return median_<uint16_t>(M);
    assert(!"Unsupported type");
}
#endif 
