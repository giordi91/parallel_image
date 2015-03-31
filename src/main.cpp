#include <iostream>
#include <fstream>
#include <cstring>
#include <core/bitmap.h>
#include <tbb/task_scheduler_init.h>
#include "tbb/tick_count.h"
#include <time.h>
#include <stdexcept>
#include <filters/bw_filter.h>
#include <filters/gaussian_filter.h>
#include <filters/sharpen_filter.h>
#include <filters/edge_detection_filter.h>
#include <core/filter_manager.h>

#include <QtWidgets/QApplication>
#include <QtWidgets/QSplashScreen>
#include <QtCore/QTimer>
#include <QtWidgets/QPushButton>

#include <ui/mainwindow.h>

using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//TODO 
//MAKE SURE WE ARE DOING CONSEQUNETIAL ACCESS IN MEMORY (MOSTLY GPU),
//aka looping first hegiht then width

int main( int argc, char* argv[]) 
{

    cudaFree(0);
    // QApplication a(argc, argv);
    // // QSplashScreen * splash = new QSplashScreen();
    // // splash->setPixmap(QPixmap("../sandbox/misc/ui/grapichs/splashScreen.png"));
    // // splash->showMessage(QString("Initializing quantum mechanics awesomness ... ")
    // //                     ,Qt::AlignLeft ,Qt::white);

    // // splash->show();
    // // Application a(argc, argv);
    // QSplashScreen * splash = new QSplashScreen();
    // splash->setPixmap(QPixmap("misc/ui/splashScreen.png"));
    // splash->showMessage(QString("Initializing quantum mechanics awesomness ... ")
    //                     ,Qt::AlignLeft ,Qt::white);

    // splash->show();

    // MainWindow w;
    // // QTimer::singleShot(250, splash,SLOT(close()));
    // // QTimer::singleShot(250, &w,SLOT(show()));


    // // MainWindow w;
    // QTimer::singleShot(250, splash,SLOT(close()));
    // QTimer::singleShot(250, &w,SLOT(show()));

    // return a.exec();



    cout<<"Initializing ..."<<endl;
    cout<<"Reading image...."<<std::endl;
    
 

  //   Bitmap testbmp;
  //   try
  //   {
		// //E:/WORK_IN_PROGRESS/C/parallel_image/data
  //       ///user_data/WORK_IN_PROGRESS/parallel_image/data/jessy.bmp
		// ///home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp
  //       testbmp.open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp");
  //   }
  //   catch(std::runtime_error &e)
  //   {
  //       std::cout<<e.what()<<endl;
  //       #if defined(WIN32)
  //       system ("PAUSE");
  //       #endif
  //       return 0;
  //   }



    // //gather the data
    // unsigned int width = testbmp.get_width();
    // unsigned int height = testbmp.get_height();
    // unsigned int padded_size = testbmp.get_padded_size();
    // Bitmap workingBmp(width, height, padded_size);
    // std::cout<<"Image Info :"<<std::endl;
    // std::cout<<"width: "<<width<<std::endl;
    // std::cout<<"height: "<<height<<std::endl;

    // //needed buffers
    // uint8_t * src = testbmp.getRawData();
    // uint8_t * target = workingBmp.getRawData();

    // tbb::tick_count t0,t1;
     
    // blur test

    // int iterations = 19;
    // std::cout<<"Operation : 20 blur iterations \n"<<endl;
    // // //time the serial functon
    // std::cout<<"Running CPU serial..."<<endl; 
    // t0 = tbb::tick_count::now();
    // simple_blur_serial(src, target, width, height, iterations);
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" seconds \n" << endl;

    // std::cout<<"Running CPU parallel (TBB)..."<<endl; 
    // t0 = tbb::tick_count::now();
    // tbb::task_scheduler_init init(8);
    // //testing tbb
    // blur_tbb(src, target, width, height, iterations);
    // //terminating tbb
    // init.terminate();
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" seconds \n" << endl;

    // std::cout<<"Running GPU parallel (Cuda)..."<<endl; 
    // t0 = tbb::tick_count::now();
    // blur_cuda(src, target, width, height, iterations);
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" seconds" << endl;

    

    //  //BW test
    // //time the serial functon
    // Bw_filter bw( width, height);

    // t0 = tbb::tick_count::now();
    // bw.compute_serial(src, target);
    
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" s" << endl; 


    // t0 = tbb::tick_count::now();
    // tbb::task_scheduler_init init;
    // //testing tbb
    // bw.compute_tbb(src, target);
    // //terminating tbb
    // init.terminate();
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" s" << endl; 


    // t0 = tbb::tick_count::now();
    // bw.compute_cuda(src, target);
    // t1 = tbb::tick_count::now();
    // cout << (t1-t0).seconds()<<" s" << endl; 
    

    //testing the stancil


    // Gaussian_filter gf(width,height,15.0f);
    // t0 = tbb::tick_count::now();
    // gf.compute_serial(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "Computing SERIAL convolution"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 
	
    // t0 = tbb::tick_count::now();
    // gf.compute_tbb(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "Computing parallel TBB convolution"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 

    // t0 = tbb::tick_count::now();
    // gf.compute_cuda(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "Computing parallel GPU convolution"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 


    // Sharpen_filter sp(width,height);
    // t0 = tbb::tick_count::now();
    // sp.compute_serial(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "serial CPU sharpen"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 



    // t0 = tbb::tick_count::now();
    // sp.compute_tbb(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "serial TBB sharpen"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 

    // t0 = tbb::tick_count::now();
    // sp.compute_cuda(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "serial Cuda sharpen"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 


    // Edge_detection_filter ed(width,height,2);
    // t0 = tbb::tick_count::now();
    // ed.compute_serial(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "Computing serial edge detection"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 

    // t0 = tbb::tick_count::now();
    // ed.compute_tbb(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "Computing TBB edge detection"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 


    // t0 = tbb::tick_count::now();
    // ed.compute_cuda(src,target);
    // t1 = tbb::tick_count::now();
    // cout << "Computing CUDA edge detection"<< endl;
    // cout << (t1-t0).seconds()<<" s" << endl; 

    Bitmap * testbmp = new Bitmap;
    try
    {
        //E:/WORK_IN_PROGRESS/C/parallel_image/data
        ///user_data/WORK_IN_PROGRESS/parallel_image/data/jessy.bmp
        ///home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp
        testbmp->open("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessy.bmp");
    }
    catch(std::runtime_error &e)
    {
        std::cout<<e.what()<<endl;
        #if defined(WIN32)
        system ("PAUSE");
        #endif
        return 0;
    } 

    //gather the data
    unsigned int width = testbmp->get_width();
    unsigned int height = testbmp->get_height();

    Bw_filter * bw = new Bw_filter ( width, height);
    Gaussian_filter * gf = new Gaussian_filter(width,height,2.0f);
    Sharpen_filter * sp= new Sharpen_filter(width,height);
    Edge_detection_filter * ed = new Edge_detection_filter(width,height,2);

    Filter_manager fm(testbmp);
    fm.add_filter(bw);
    fm.add_filter(ed);
    fm.add_filter(sp);
    fm.add_filter(gf);
    fm.set_compute_type(Filter_manager::TBB);
    fm.evaluate_stack();



  //   try
  //   {   ///user_data/WORK_IN_PROGRESS/parallel_image/data/jessy.bmp
  //       workingBmp.save("/home/giordi/WORK_IN_PROGRESS/C/parallel_image/data/jessyBW.bmp");
  //   }
  //   catch(std::runtime_error &e)
  //   {
  //       std::cout<<e.what()<<endl;
		// #if defined(WIN32)
  //       system ("PAUSE");
		// #endif  
  //       return 0;
  //   }

  //   #if defined(WIN32)
  //   system ("PAUSE");
  //   #endif        
    // return 0;

}