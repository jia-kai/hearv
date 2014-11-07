/*
 * $File: imgdisp.cpp
 * $Date: Fri Nov 07 23:51:57 2014 +0800
 * $Author: jiakai <jia.kai66@gmail.com>
 */

#include <cstdio>
#include <vector>
#include <GL/glut.h>
#include <opencv2/opencv.hpp>

static int width  = 640, height = 480, nr_frame, prev_fps_sample_time;
static std::vector<cv::Mat> images;
static size_t cur_img_idx = 0;

static void set_image(cv::Mat &img) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
            img.cols, img.rows,
            0,
            GL_RGB, GL_UNSIGNED_BYTE, img.ptr()); /* Texture specification */
}

/* Handler for window-repaint event. Called back when the window first appears and
   whenever the window needs to be re-painted. */
static void on_display() {
    // Clear color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
    glMatrixMode(GL_MODELVIEW);     // Operate on model-view matrix

    // set_image(images[cur_img_idx ++]);
    if (cur_img_idx == images.size())
        cur_img_idx = 0;

    /* Draw a quad */
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex2i(0, 0);
    glTexCoord2i(0, 1); glVertex2i(0, height);
    glTexCoord2i(1, 1); glVertex2i(width, height);
    glTexCoord2i(1, 0); glVertex2i(width, 0);
    glEnd();

    glutSwapBuffers();
    glFinish();

    int t = time(nullptr);
    if (t - prev_fps_sample_time >= 5) {
        printf("fps: %.2f\n", double(nr_frame) / (t - prev_fps_sample_time));
        prev_fps_sample_time = t;
        nr_frame = 0;
    }
    nr_frame ++;

    glutPostRedisplay();
} 

/* Handler for window re-size event. Called back when the window first appears and
   whenever the window is re-sized with its new width and height */
static void on_reshape(GLsizei newwidth, GLsizei newheight) {  
    // Set the viewport to cover the new window
    width = newwidth;
    height = newheight;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, width, height, 0.0, 0.0, 100.0);
    glMatrixMode(GL_MODELVIEW);

    glutPostRedisplay();
}

/* Initialize OpenGL Graphics */
static void init_gl(int w, int h) {

    glViewport(0, 0, w, h); // use a screen size of WIDTH x HEIGHT
    glEnable(GL_TEXTURE_2D);     // Enable 2D texturing

    glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
    glLoadIdentity();
    glOrtho(0.0, w, h, 0.0, 0.0, 100.0);

    glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f); 
    glClearDepth(0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear the window
}

static void on_keyboard(unsigned char ch, int, int) {
    if (ch == 'q')
        exit(0);
}

int main(int argc, char **argv) {
    if (argc == 1) {
        fprintf(stderr, "usage: %s <images ...>\n"
                "   cycle between given images\n",
                argv[0]);
        return -1;
    }
    for (int i = 1; i < argc; i ++) {
        auto img = cv::imread(argv[i], CV_LOAD_IMAGE_COLOR);
        if (img.empty()) {
            fprintf(stderr, "failed to load %s: %m\n", argv[i]);
            return -1;
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        images.push_back(img);
    }

    prev_fps_sample_time = time(nullptr);

    GLuint texid;

    /* GLUT init */
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow(argv[0]);
    glutDisplayFunc(on_display);
    glutReshapeFunc(on_reshape);
    glutKeyboardFunc(on_keyboard);

    /* OpenGL 2D generic init */
    init_gl(width, height);

    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    set_image(images.front());

    glutMainLoop();

    glDeleteTextures(1, &texid);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
