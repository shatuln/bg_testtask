#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>

#define FRAME_HEIGHT    768
#define FRAME_WIDTH     1024

using namespace cv;

const Point startPoint = Point(20, FRAME_HEIGHT*0.25);

void DrawRotatedRectangle(Mat& image, const Point& centerPoint, const Size& rectangleSize, const double rotationDegrees)
{
    Scalar color = Scalar(255, 0, 0);
    RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

    Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);
    Point vertices[4];    
    for(int i = 0; i < 4; ++i){
        vertices[i] = vertices2f[i];
    }
    fillConvexPoly(image, vertices, 4, color);
    return;
}

void ComputeCoef(float& a, float& b, float& c, const std::vector<Point>& points) {
    float x1 = points[0].x + FRAME_WIDTH * 0.25;
    float x2 = points[1].x + FRAME_WIDTH * 0.25;
    float x3 = points[2].x + FRAME_WIDTH * 0.25;
    float y1 = points[0].y * -1;
    float y2 = points[1].y * -1;
    float y3 = points[2].y * -1; 
    a = (y3 - (((x3*(y2-y1))+(x2*y1)-(x1*y2))/(x2-x1))) / (x3*(x3-x1-x2)+x1*x2);
    b = ((y2-y1)/(x2-x1)) - a*(x1+x2);
    c = ((x2*y1-x1*y2)/(x2-x1) + a*x1*x2);
    return;
}

void DrawTargetRect(Mat& image, const float diam, float a = 0, float b = 0, float c = 0) {
    float centerx, centery;
    centerx = FRAME_WIDTH - 20;
    if (a != 0 || b != 0 || c != 0)
        centery = (a*(pow(centerx,2)) + b*centerx + c) * -1 - diam / 4;
    else
        centery = startPoint.y;
    Rect target_rect = Rect(Point(centerx-10/2,centery-diam/2), Point(centerx+10/2,centery+diam/2));
    rectangle(image, target_rect, Scalar(0,0,255), -1, 8, 0);
    return;
}

void AnalyzeInputOneFrame(Mat& captureMat, const int radius, std::vector<Point>& points) {

    Rect captureArea(Point(FRAME_WIDTH * 0.25, 0), Size(FRAME_WIDTH * 0.5, FRAME_HEIGHT));
    int globalradius = 0, radiuscount = 0, diam = 0;
    float a = 0, b = 0, c = 0;
    Mat analyzeMat(captureArea.size(), CV_8U), tmp, gray;
    tmp = captureMat(captureArea);
    tmp.copyTo(analyzeMat);
    cvtColor(analyzeMat, gray, COLOR_BGR2GRAY);
    blur(gray, gray, Size(3,3), Point(-1,-1));
    std::vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 2, gray.rows/4, 100, 20 );
    rectangle(captureMat, captureArea, Scalar(230, 230, 230));
    if (circles.size() == 1) {
        Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
        if (center.x < (analyzeMat.cols/3) && center.x > (30) && points.size() < 1) {
            points.push_back(center);
        } else if (center.x < (analyzeMat.cols/3*2) && center.x > (analyzeMat.cols/3) && points.size() < 2) {
            points.push_back(center);
        } else if (center.x < analyzeMat.cols && center.x > (analyzeMat.cols/3*2) && points.size() < 3) {
            points.push_back(center);
        }
        // circle(gray, center, 3, Scalar(0,255,0), -1, 8, 0);
        // circle(gray, center, radius, Scalar(0,0,255), 3, 8, 0);
    }
    if (points.size() >= 3) {
        ComputeCoef(a, b, c, points);
    }
    DrawTargetRect(captureMat, radius * 2, a, b, c);
    return;
}

bool GenerateVideo(const int radius, const float alpha) {
    if (5.0 > alpha || alpha > 15.0) {
        std::cout << "Input valid alpha (from 5.0 to 15.0)" << std::endl;
        return false;
    }

    RNG rng(static_cast<unsigned int>(time(0)));
    int v0 = rng.uniform(25, 45);
    int g = 1;

    Point2i circle_center = startPoint;
    Mat mat(Size(FRAME_WIDTH, FRAME_HEIGHT),CV_8U);
    cvtColor(mat, mat, COLOR_GRAY2BGR);
    VideoWriter outputVideo("output.avi", VideoWriter::fourcc('D','I','V','X'), 25, Size(FRAME_WIDTH, FRAME_HEIGHT), true);
    if (!outputVideo.isOpened()) {
        std::cout << "can't open video" << std::endl;
        return false;
    }

    VideoWriter outputVideoFinal("output_final.avi", VideoWriter::fourcc('D','I','V','X'), 25, Size(FRAME_WIDTH, FRAME_HEIGHT), true);
    if (!outputVideoFinal.isOpened()) {
        std::cout << "can't open final output video" << std::endl;
        return false;
    }

    std::vector<Point> points;
    float t = 0;
    while ((circle_center.x < (FRAME_WIDTH - 30 - (radius*0.75)) && circle_center.x > 0) && (circle_center.y < (FRAME_HEIGHT + 10) && circle_center.y > 0)) {
        mat = Scalar(255, 255, 255);
        circle_center.x = startPoint.x + v0 * t * cos(alpha * CV_PI / 180.0);
        circle_center.y = startPoint.y - v0 * t * sin(alpha * CV_PI / 180.0) + (g * pow(t, 2)) / 2;
        std::cout << circle_center.x << ":" << circle_center.y << std::endl;
        circle(mat, circle_center, radius, Scalar(0,0,255), -1);
        DrawRotatedRectangle(mat, startPoint, Size(50, radius*2), 360-alpha);
        outputVideo << mat;

        t+=0.3;
        
        AnalyzeInputOneFrame(mat, radius, points);
        outputVideoFinal << mat;
    }
    outputVideo.release();
    outputVideoFinal.release();

    return true;
}

int main(int argc, char* argv[]) {
    if (argc == 3) {
        int radius = std::stoi(argv[1]);
        float alpha = std::stoi(argv[2]);
        if (!GenerateVideo(radius, alpha)) {
            std::cout << "Cannot generate source video" << std::endl;
            return -1;
        }
    } else {
        std::cout << "Arguments count is invalid" << std::endl;
        return -1;
    }

    return 1;

}