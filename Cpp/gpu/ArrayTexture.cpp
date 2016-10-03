// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#include "ArrayTexture.hpp"
//#include <opencv2/gpu/device/common.hpp>
#include <opencv2/cudev/common.hpp>

using namespace cv::cuda;
ArrayTexture::ArrayTexture(const cv::Mat& image, const Stream& cvStream) {
    refcount=&ref_count;
    ref_count=1;
    cv::Mat im2=image.clone();
    assert(image.isContinuous());
    assert(image.type()==CV_8UC4);
    
    //Describe texture
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;
    cudaChannelFormatDesc channelDesc = {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    // cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    //Fill Memory
    if(!cuArray){
        CV_CUDEV_SAFE_CALL(cudaMallocArray(&cuArray, &channelDesc, image.cols, image.rows));
    }
    CV_CUDEV_SAFE_CALL(cudaMemcpyToArray(cuArray, 0, 0, im2.datastart, im2.dataend-im2.datastart,
                                   cudaMemcpyHostToDevice/*,StreamAccessor::getStream(cvStream)*/));
    
    // Specify texture memory location
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    // Create texture object
    CV_CUDEV_SAFE_CALL(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
    
    
}
