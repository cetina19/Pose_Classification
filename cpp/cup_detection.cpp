#include <opencv2/opencv.hpp>
#include <mediapipe/framework/formats/landmark.pb.h>
#include <cmath>
#include <vector>

float calculateAngle(const cv::Point& a, const cv::Point& b, const cv::Point& c) {
    float radians = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x);
    float angle = fabs(radians * 180.0 / M_PI);
    if (angle > 180.0)
        angle = 360.0 - angle;
    return angle;
}

int main() {
    cv::VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;

    cv::Mat frame, image;
    vector<cv::Point> landmarks;
    float angle, angle2 = -1;
    int curl_count = 0;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, image, cv::COLOR_BGR2RGB);
        landmarks = processPose(image);

        if (landmarks.size() >= 3) {
            cv::Point up = landmarks[11];
            cv::Point mid = landmarks[13];
            cv::Point down = landmarks[15];

            angle = calculateAngle(up, mid, down);
            cv::putText(image, to_string(angle),
                        cv::Point((int)(mid.x * frame.cols), (int)(mid.y * frame.rows)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

            if ((angle2 - angle) > 90) {
                curl_count++;
                angle2 = angle;
            }
        }

        drawLandmarks(image, landmarks);

        cv::putText(image, to_string(curl_count),
                    cv::Point(50, 150), cv::FONT_HERSHEY_PLAIN, 5,
                    cv::Scalar(250, 0, 0), 5);
        if (angle > 140)
            angle2 = angle;

        cv::imshow("Mediapipe Feed", image);

        if (cv::waitKey(10) == 'q')
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
