// loads all files of a given name and extension
#include "fileLoader.hpp"
    #include <boost/filesystem.hpp>
#include <fstream>
#include <string>
    
    namespace fs = ::boost::filesystem;
    static fs::path root;
static std::vector<fs::path> txt;
static std::vector<fs::path> png;
static std::vector<fs::path> depth;


void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret);
void loadAhanda(const char * rootpath,
                double range,
                int imageNumber,
                cv::Mat& image,
                cv::Mat& d,
                cv::Mat& cameraMatrix,
                cv::Mat& R,
                cv::Mat& T){
    if(root!=std::string(rootpath)){

        root=std::string(rootpath);
        get_all(root, ".txt", txt);
        get_all(root, ".png", png);
        get_all(root, ".depth", depth);
                std::cout<<"Loading"<<std::endl;
    }
    
    convertAhandaPovRayToStandard(txt[imageNumber].c_str(),
                                      cameraMatrix,
                                      R,
                                      T);
    std::cout<<"Reading: "<<png[imageNumber].filename().string()<<std::endl;
    cv::imread(png[imageNumber].string(), -1).convertTo(image,CV_32FC3,1.0/range,1/255.0);
    int r=image.rows;
    int c=image.cols;
    if(depth.size()>0){
        std::cout<<"Depth: "<<depth[imageNumber].filename().string()<<std::endl;
        d=loadDepthAhanda(depth[imageNumber].string(), r,c,cameraMatrix);
    }
    
   
}


cv::Mat loadDepthAhanda(std::string filename, int r,int c,cv::Mat cameraMatrix){
    std::ifstream in(filename.c_str());
    int sz=r*c;
    cv::Mat_<float> out(r,c);
    float * p=(float *)out.data;
    for(int i=0;i<sz;i++){
        in>>p[i];
        assert(p[i]!=0);
    }
    cv::Mat_<double> K = cameraMatrix;
    double fx=K(0,0);
    double fy=K(1,1);
    double cx=K(0,2);
    double cy=K(1,2);
    for (int i=0;i<r;i++){
        for (int j=0;j<c;j++,p++){
            double x=j;
            double y=i;
            x=(x-cx)/fx;
            y=(y-cy)/fy;
            *p=*p/sqrt(x*x+y*y+1);
        }
    }
    
    
    
    return out;
}





































#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 


// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret)
{  
  if (!fs::exists(root)) return;

  if (fs::is_directory(root))
  {
    typedef std::set<boost::filesystem::path> Files;
    Files files;
    fs::recursive_directory_iterator it0(root);
    fs::recursive_directory_iterator endit0;
    std::copy(it0, endit0, std::inserter(files, files.begin()));
    Files::iterator it= files.begin();
    Files::iterator endit= files.end();
    while(it != endit)
    {
      if (fs::is_regular_file(*it) && (*it).extension() == ext)
      {
        ret.push_back(*it);
      }
      ++it;
    }
  }
}
