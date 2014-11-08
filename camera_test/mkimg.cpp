/*
 * $File: mkimg.cpp
 * $Date: Sat Nov 08 11:54:10 2014 +0800
 * $Author: jiakai <jia.kai66@gmail.com>
 */

#include <opencv2/opencv.hpp>
#include <cassert>
#include <vector>

constexpr int NR = 60, DIFF_THRESH = NR / 2 - 2;
using pat_t = uint64_t;
static_assert(DIFF_THRESH >= 0, "NR too small");
static_assert(NR <= sizeof(pat_t) * 8, "NR too large");

namespace {

/*!
 * pattern of ababab...ab, where a = (0 repeated n1 times) and
 * b = (1 repeated n1 times)
 */
template<size_t n1, size_t len = sizeof(pat_t) * 8>
struct sepv {
    static_assert(len % (n1 * 2) == 0 && len >= (n1 * 2), "bad param");
    static constexpr pat_t v = ((sepv<n1, len - n1 * 2>::v << n1) << n1) |
        ((pat_t(1) << n1) - 1);
};

template<size_t n1>
struct sepv<n1, 0> {
    static constexpr pat_t v = 0;
};

template<size_t reduce>
struct impl_cnt_1 {
    static pat_t f(pat_t v) {
        v = impl_cnt_1<reduce / 2>::f(v);
        return (v & sepv<reduce>::v) + ((v >> reduce) & sepv<reduce>::v);
    }
};

template<>
struct impl_cnt_1<0> {
    static pat_t f(pat_t v) {
        return v;
    }
};

int cnt_1(pat_t v) {
    return impl_cnt_1<sizeof(pat_t) * 4>::f(v);
}

std::vector<pat_t> make_pattern() {
    std::vector<pat_t> result;
    std::vector<int> cur;
    for (size_t i = 0; i < sizeof(pat_t) * 8; i ++)
        cur.push_back(i & 1);

    for (int i = 0; i < NR; i ++) {
        std::random_shuffle(cur.begin(), cur.end());
        pat_t v = 0;
        for (auto j: cur)
            v = (v << 1) | j;
        bool ok = true;
        for (auto &&prev: result)
            if (cnt_1(prev ^ v) < DIFF_THRESH) {
                ok = false;
                break;
            }
        if (ok) {
            printf("generated pattern %2d/%d: 0x%016zx\n", i + 1, NR, v);
            result.push_back(v);
        } else {
            i --;
        }
    }
    return result;
}

cv::Mat gen_img(const std::vector<pat_t> &pat) {
    cv::Mat rst(pat.size(), sizeof(pat_t) * 8, CV_8UC1);
    for (size_t i = 0; i < pat.size(); i ++) {
        auto v = pat[i];
        auto row = rst.ptr(i);
        for (int j = sizeof(pat_t) * 8 - 1; j >= 0; j --) {
            row[j] = (v & 1) * 255;
            v >>= 1;
        }
    }
    return rst;
}

} // anonymous namespace

int main() {
    srand(time(nullptr));
    auto pat = make_pattern();

    for (int i = 0; i < NR; i ++) {
        char fpath[255];
        sprintf(fpath, "data/%02d.png", i);
        cv::imwrite(fpath, gen_img(pat));
        sprintf(fpath, "data/%02d.txt", i);
        FILE *fout = fopen(fpath, "w");
        assert(fout);
        for (int r = 0; r < NR; r ++)
            fprintf(fout, "0x%zx\n", pat[r]);
        fclose(fout);
        auto v0 = pat[0];
        pat.erase(pat.begin());
        pat.push_back(v0);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

