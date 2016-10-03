// This file converts the file format used at http://www.doc.ic.ac.uk/~ahanda/HighFrameRateTracking/downloads.html
// into the standard [R|T] world -> camera format used by OpenCV
// It is based on a file they provided there, but makes the world coordinate system right handed, with z up,
// x right, and y forward.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

cv::Vec3f direction;
cv::Vec3f upvector;
void convertAhandaPovRayToStandard(const char * filepath,
                                   cv::Mat& cameraMatrix,
                                   cv::Mat& R,
                                   cv::Mat& T)
{
    char text_file_name[600];
    sprintf(text_file_name,"%s",filepath);

    std::cout << "text_file_name = " << text_file_name << std::endl;

    std::ifstream cam_pars_file(text_file_name);
    if(!cam_pars_file.is_open())
    {
        std::cerr<<"Failed to open param file, check location of sample trajectory!"<<std::endl;
        exit(1);
    }

    char readlinedata[300];

    cv::Point3d direction;
    cv::Point3d upvector;
    cv::Point3d posvector;


    while(1){
        cam_pars_file.getline(readlinedata,300);
//         std::cout<<readlinedata<<std::endl;
        if ( cam_pars_file.eof())
            break;


        std::istringstream iss;


        if ( strstr(readlinedata,"cam_dir")!= NULL){


            std::string cam_dir_str(readlinedata);

            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));

            iss.str(cam_dir_str);
            iss >> direction.x ;
            iss.ignore(1,',');
            iss >> direction.z ;
            iss.ignore(1,',') ;
            iss >> direction.y;
            iss.ignore(1,',');
//             std::cout << "direction: "<< direction.x<< ", "<< direction.y << ", "<< direction.z << std::endl;

        }

        if ( strstr(readlinedata,"cam_up")!= NULL){

            std::string cam_up_str(readlinedata);

            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));


            iss.str(cam_up_str);
            iss >> upvector.x ;
            iss.ignore(1,',');
            iss >> upvector.z ;
            iss.ignore(1,',');
            iss >> upvector.y ;
            iss.ignore(1,',');



        }

        if ( strstr(readlinedata,"cam_pos")!= NULL){
//            std::cout<< "cam_pos is present!"<<std::endl;

            std::string cam_pos_str(readlinedata);

            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));

//            std::cout << "cam pose str = " << std::endl;
//            std::cout << cam_pos_str << std::endl;

            iss.str(cam_pos_str);
            iss >> posvector.x ;
            iss.ignore(1,',');
            iss >> posvector.z ;
            iss.ignore(1,',');
            iss >> posvector.y ;
            iss.ignore(1,',');
//             std::cout << "position: "<<posvector.x<< ", "<< posvector.y << ", "<< posvector.z << std::endl;

        }

    }

    R=cv::Mat(3,3,CV_64F);
    R.row(0)=cv::Mat(direction.cross(upvector)).t();
    R.row(1)=cv::Mat(-upvector).t();
    R.row(2)=cv::Mat(direction).t();

    T=-R*cv::Mat(posvector);
//     std::cout<<"T: "<<T<<std::endl<<"pos: "<<cv::Mat(posvector)<<std::endl;
   /* cameraMatrix=(cv::Mat_<double>(3,3) << 480,0.0,320.5,
										    0.0,480.0,240.5,
										    0.0,0.0,1.0);*/
    cameraMatrix=(cv::Mat_<double>(3,3) << 481.20,0.0,319.5,
                  0.0,480.0,239.5,
                  0.0,0.0,1.0);

}



