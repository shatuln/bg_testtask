#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>

using namespace cv;

void DrawRotatedRectangle(cv::Mat& image, cv::Point centerPoint, cv::Size rectangleSize, double rotationDegrees)
{
    cv::Scalar color = cv::Scalar(255, 0, 0);

    // Create the rotated rectangle
    cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

    // We take the edges that OpenCV calculated for us
    cv::Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);

    // Convert them so we can use them in a fillConvexPoly
    cv::Point vertices[4];    
    for(int i = 0; i < 4; ++i){
        vertices[i] = vertices2f[i];
    }

    // Now we can fill the rotated rectangle with our specified color
    cv::fillConvexPoly(image,
                       vertices,
                       4,
                       color);
}

bool GenerateSrc(int radius, float alpha) {
    if (5.0 > alpha || alpha > 15.0) {
        std::cout << "Input valid alpha (from 5.0 to 15.0)" << std::endl;
        return false;
    }

    RNG rng(static_cast<unsigned int>(time(0)));
    int v0 = rng.uniform(25, 45);
    int g = 1;

    Point2i circle_center(20, 250);
    Mat mat(Size(720,480),CV_8U);
    cvtColor(mat, mat, COLOR_GRAY2BGR);
    VideoWriter outputVideo("output.avi", VideoWriter::fourcc('D','I','V','X'), 25, Size(720,480), true);
    outputVideo.open("output.avi", VideoWriter::fourcc('D','I','V','X'), 5, Size(720,480), true);
    if (!outputVideo.isOpened()) {
        std::cout << "can't open video" << std::endl;
        return false;
    }

    float t = 0;
    //mat = Scalar(255, 255, 255);
    Point2i init_circle_center = circle_center;
    while ((circle_center.x < 730 && circle_center.x > 0) && (circle_center.y < 490 && circle_center.y > 0)) {
        mat = Scalar(255, 255, 255);
        DrawRotatedRectangle(mat, init_circle_center, Size(50, radius), 360-alpha);
        circle_center.x = init_circle_center.x + v0 * t * cos(alpha * CV_PI / 180.0);
        circle_center.y = init_circle_center.y - v0 * t * sin(alpha * CV_PI / 180.0) + (g * pow(t, 2)) / 2;
        std::cout << circle_center.x << ":" << circle_center.y << std::endl;
        circle(mat, circle_center, radius, Scalar(0,0,255), -1);
        outputVideo << mat;

        t+=0.5;
    }
    outputVideo.release();

    return true;
}

int main(int argc, char* argv[]) {
    if (argc == 3) {
        int radius = std::stoi(argv[1]);
        float alpha = std::stoi(argv[2]);
        if (!GenerateSrc(radius, alpha)) {
            std::cout << "Cannot generate source video" << std::endl;
            return -1;
        }
    } else if (argc == 1) {
        return 1;
    } else {
        std::cout << "Arguments count is invalid" << std::endl;
        return -1;
    }

    return 1;
    

    

    // //namedWindow("Display Image", WINDOW_AUTOSIZE);
    // //imshow("Display Image", mat);

    // waitKey(0);
    // return 0;
}