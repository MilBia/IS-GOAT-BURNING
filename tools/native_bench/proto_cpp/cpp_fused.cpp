// C++/pybind11 prototype of the fused post-blur kernel of the classic detector (#133).
// One pass over the pixels: OpenCV-exact 8-bit BGR2HSV + inRange + BGR2GRAY + |gray - prev| > thr + both counts,
// writes the new gray image into prev_gray in place. Compiled with plain -O3 (portable baseline: SSE2 / NEON).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace py = pybind11;
using u8arr = py::array_t<uint8_t, py::array::c_style>;

namespace {
constexpr int kShift = 12;
struct Tables {
    int sdiv[256];
    int hdiv[256];
    Tables() {
        sdiv[0] = hdiv[0] = 0;
        for (int i = 1; i < 256; ++i) {
            sdiv[i] = (int)std::lround((255 << kShift) / (1.0 * i));
            hdiv[i] = (int)std::lround((180 << kShift) / (6.0 * i));
        }
    }
};
const Tables T;
}  // namespace

std::pair<long, long> fused(u8arr blurred, u8arr frame, u8arr prev, u8arr lower, u8arr upper, int thr) {
    if (blurred.ndim() != 3 || blurred.shape(2) != 3 || frame.ndim() != 3 || prev.ndim() != 2)
        throw std::invalid_argument("expected (H,W,3), (H,W,3), (H,W)");
    const long h = blurred.shape(0), w = blurred.shape(1);
    if (frame.shape(0) != h || frame.shape(1) != w || prev.shape(0) != h || prev.shape(1) != w)
        throw std::invalid_argument("shape mismatch");
    const uint8_t *bl = blurred.data(), *fr = frame.data();
    uint8_t* pv = prev.mutable_data();
    const int lo0 = lower.data()[0], lo1 = lower.data()[1], lo2 = lower.data()[2];
    const int up0 = upper.data()[0], up1 = upper.data()[1], up2 = upper.data()[2];
    long moving_warm = 0, warm = 0;
    {
        py::gil_scoped_release release;
        const int* sdiv = T.sdiv;
        const int* hdiv = T.hdiv;
        for (long y = 0; y < h; ++y) {
            const uint8_t* brow = bl + y * w * 3;
            const uint8_t* frow = fr + y * w * 3;
            uint8_t* prow = pv + y * w;
            long row_moving_warm = 0, row_warm = 0;
            for (long x = 0; x < w; ++x) {
                // --- colour: OpenCV integer BGR2HSV on the blurred frame ---
                const int b = brow[3 * x], g = brow[3 * x + 1], r = brow[3 * x + 2];
                const int v = std::max(b, std::max(g, r));
                const int vmin = std::min(b, std::min(g, r));
                const int diff = v - vmin;
                const int vr = (v == r) ? -1 : 0;
                const int vg = (v == g) ? -1 : 0;
                const int s = (diff * sdiv[v] + (1 << (kShift - 1))) >> kShift;
                int hh = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + (~vg & (r - g + 4 * diff))));
                hh = (hh * hdiv[diff] + (1 << (kShift - 1))) >> kShift;
                hh += hh < 0 ? 180 : 0;
                const int is_warm = (hh >= lo0) & (hh <= up0) & (s >= lo1) & (s <= up1) & (v >= lo2) & (v <= up2);
                // --- motion: OpenCV fixed-point BGR2GRAY on the raw frame ---
                const int fb = frow[3 * x], fg = frow[3 * x + 1], fr_ = frow[3 * x + 2];
                const int gray = (fb * 3735 + fg * 19235 + fr_ * 9798 + (1 << 14)) >> 15;
                const int d = std::abs(gray - (int)prow[x]);
                const int is_moving = d > thr;
                prow[x] = (uint8_t)gray;
                row_warm += is_warm;
                row_moving_warm += is_warm & is_moving;
            }
            warm += row_warm;
            moving_warm += row_moving_warm;
        }
    }
    return {moving_warm, warm};
}

PYBIND11_MODULE(cpp_fused, m) {
    m.doc() = "Fused colour+motion kernel prototype (C++/pybind11)";
    m.def("fused", &fused, "fused(blurred, frame, prev_gray, lower, upper, thr) -> (moving_warm, warm)");
}
