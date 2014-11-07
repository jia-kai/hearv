/*
 * $File: imgdisp.cpp
 * $Date: Sat Nov 08 00:57:02 2014 +0800
 * $Author: jiakai <jia.kai66@gmail.com>
 */

#include <cstdio>
#include <list>
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl_interop.hpp>
#include <sys/time.h>

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
        images[i - 1].copyFrom(img);
    }

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
