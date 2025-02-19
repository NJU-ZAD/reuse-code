#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

int main()
{
    printf("Hello OpenCV\n");
    cv::String path = getenv("HOME");
    path += "/shell/icon/IEEE.jpg";
    cv::Mat img = cv::imread(path, 1);
    cv::imshow("image", img);
    cv::waitKey(0);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::imshow("grayimage", gray);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
/*
cd cpp;g++ -g -std=c++17 opencv.cpp -o opencv `pkg-config --libs --cflags opencv4`;./opencv;cd ..
cd cpp;rm -rf opencv;cd ..
*/