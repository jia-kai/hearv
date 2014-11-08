/*
 * $File: imgdisp.cpp
 * $Date: Sat Nov 08 12:17:58 2014 +0800
 * $Author: jiakai <jia.kai66@gmail.com>
 */

#include <cstdio>
#include <list>
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <sys/time.h>

constexpr int SCALE_FACTOR = 10;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    std::vector<cv::ogl::Texture2D> images;
    if (argc == 1) {
        fprintf(stderr, "usage: %s <images ...>\n"
                "   cycle between given images\n",
                argv[0]);
        return -1;
    }
    cv::namedWindow("img", cv::WINDOW_OPENGL);

    images.resize(argc - 1);
    for (int i = 1; i < argc; i ++) {
        auto img = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
        if (img.empty()) {
            fprintf(stderr, "failed to load %s: %m\n", argv[i]);
            return -1;
        }
        cv::Mat img1(img.rows + 2, img.cols + 2, CV_8UC3, cv::Scalar{0});
        img1.at<cv::Vec3b>(0, 0) = {0, 0, 255};
        img1.at<cv::Vec3b>(0, img.cols + 1) = {0, 255, 0};
        img1.at<cv::Vec3b>(img.rows + 1, 0) = {255, 0, 0};
        img1.at<cv::Vec3b>(img.rows + 1, img.cols + 1) = {255, 0, 0};
        img.copyTo(img1(cv::Rect(1, 1, img.cols, img.rows)));
        img = img1;
        cv::resize(img, img, {0, 0}, SCALE_FACTOR, SCALE_FACTOR,
                cv::INTER_NEAREST);
        images[i - 1].copyFrom(img);
    }
    cv::resizeWindow("img", images[0].cols(), images[0].rows());

    int nr_frame = 0;
    size_t idx = 0;
    double fps_time = get_time();
    for (; ; ) {
        nr_frame ++;
        if (nr_frame >= 100) {
            auto now = get_time();
            printf("fps: %.2f\n", nr_frame / (now - fps_time));
            fps_time = now;
            nr_frame = 0;
        }
        cv::imshow("img", images[idx ++]);
        if (idx == images.size())
            idx = 0;
        if ((cv::waitKey(1) & 0xFF) == 'q')
            break;
    }
    cv::destroyWindow("img");
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
