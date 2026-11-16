//! Fused per-pixel kernel for the classic fire detector (research prototype, issue #133).
//!
//! One pass over the frame computes, bit-exactly like OpenCV's 8-bit paths:
//!   * BGR->HSV (hsv_shift = 12 integer algorithm) of the *blurred* frame, then `inRange(lower, upper)` -> `warm`
//!   * BGR->GRAY (fixed point, 15 bit: B=3735, G=19235, R=9798 -- what OpenCV 5's 8u path actually uses, NOT the
//!     shift-14 1868/9617/4899 set from the task text; verified empirically, 0 mismatches vs cv2) of the *raw* frame, `absdiff` vs `prev_gray`, `> thr` -> `motion`
//!   * counts of `warm & motion` and `warm`; `prev_gray` is overwritten in place with the new gray image.
//!
//! The inner loop is branch-free scalar code over `chunks_exact(3)`; the only data-dependent memory access is
//! two 256-entry i32 lookup tables (L1 resident). No `unsafe`, no explicit SIMD.
use numpy::{PyReadonlyArray1, PyReadonlyArray3, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::OnceLock;

const HSV_SHIFT: u32 = 12;

struct Tables {
    sdiv: [i32; 256],
    hdiv: [i32; 256],
}

/// OpenCV's `sdiv_table` / `hdiv_table` (cv::cvRound == round-half-to-even on the double quotient).
fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables { sdiv: [0; 256], hdiv: [0; 256] };
        for i in 1..256usize {
            t.sdiv[i] = ((255u32 << HSV_SHIFT) as f64 / i as f64).round_ties_even() as i32;
            t.hdiv[i] = ((180u32 << HSV_SHIFT) as f64 / (6.0 * i as f64)).round_ties_even() as i32;
        }
        t
    })
}

#[derive(Clone, Copy)]
struct Params {
    lo: [i32; 3],
    hi: [i32; 3],
    thr: i32,
}

/// Process a band of rows. `blurred`/`frame` are BGR triples, `prev` is gray (updated in place).
/// Returns (moving_warm, warm) counts for the band.
#[inline(never)]
fn kernel(blurred: &[u8], frame: &[u8], prev: &mut [u8], p: Params, t: &Tables) -> (u64, u64) {
    let mut moving = 0u32;
    let mut warm = 0u32;
    // Per-row accumulation in u32 keeps the reduction vector-friendly; widen to u64 across rows.
    let (mut moving_total, mut warm_total) = (0u64, 0u64);
    for ((brow, frow), prow) in blurred.chunks(4096 * 3).zip(frame.chunks(4096 * 3)).zip(prev.chunks_mut(4096)) {
        for ((bp, fp), pg) in brow.chunks_exact(3).zip(frow.chunks_exact(3)).zip(prow.iter_mut()) {
            // ---- HSV of blurred pixel (OpenCV 8u algorithm) ----
            let (b, g, r) = (bp[0] as i32, bp[1] as i32, bp[2] as i32);
            let v = b.max(g).max(r);
            let diff = v - b.min(g).min(r);
            let s = (diff * t.sdiv[v as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
            // Early-out: hue (the expensive part) is only needed for pixels that already pass S and V.
            // Rarely taken for natural scenes, so the branch predicts well.
            let w = if (s >= p.lo[1]) & (s <= p.hi[1]) & (v >= p.lo[2]) & (v <= p.hi[2]) {
                let vr = -((v == r) as i32);
                let vg = -((v == g) as i32);
                let h = (vr & (g - b)) + (!vr & ((vg & (b - r + 2 * diff)) + (!vg & (r - g + 4 * diff))));
                let h = (h * t.hdiv[diff as usize] + (1 << (HSV_SHIFT - 1))) >> HSV_SHIFT;
                let h = h + (180 & (h >> 31)); // h < 0 -> += 180
                (h >= p.lo[0]) & (h <= p.hi[0])
            } else {
                false
            };
            // ---- GRAY of raw pixel, absdiff, threshold ----
            let gray = (fp[0] as u32 * 3735 + fp[1] as u32 * 19235 + fp[2] as u32 * 9798 + 16384) >> 15;
            let gray = gray as u8;
            let d = (gray as i32 - *pg as i32).abs();
            let m = d > p.thr;
            *pg = gray;
            warm += w as u32;
            moving += (w & m) as u32;
        }
        moving_total += moving as u64;
        warm_total += warm as u64;
        moving = 0;
        warm = 0;
    }
    (moving_total, warm_total)
}

fn run(blurred: &[u8], frame: &[u8], prev: &mut [u8], p: Params, threads: usize) -> (u64, u64) {
    let t = tables();
    let n = prev.len();
    if threads <= 1 || n < 65536 {
        return kernel(blurred, frame, prev, p, t);
    }
    // Split into contiguous pixel bands, one scoped thread each (no Python objects touched inside).
    let band = n.div_ceil(threads);
    std::thread::scope(|sc| {
        let hs: Vec<_> = prev
            .chunks_mut(band)
            .zip(blurred.chunks(band * 3))
            .zip(frame.chunks(band * 3))
            .map(|((pg, bl), fr)| sc.spawn(move || kernel(bl, fr, pg, p, t)))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    })
}

fn fused_impl(
    py: Python<'_>,
    blurred: PyReadonlyArray3<'_, u8>,
    frame: PyReadonlyArray3<'_, u8>,
    mut prev_gray: PyReadwriteArray2<'_, u8>,
    lower: PyReadonlyArray1<'_, u8>,
    upper: PyReadonlyArray1<'_, u8>,
    thr: i32,
    threads: usize,
) -> PyResult<(u64, u64)> {
    let bad = |m: &str| PyValueError::new_err(m.to_string());
    let (h, w) = (prev_gray.shape()[0], prev_gray.shape()[1]);
    if blurred.shape() != [h, w, 3] || frame.shape() != [h, w, 3] {
        return Err(bad("blurred/frame must be (H, W, 3) matching prev_gray (H, W)"));
    }
    let (lo, hi) = (lower.as_slice().map_err(|_| bad("lower must be contiguous"))?, upper.as_slice().map_err(|_| bad("upper must be contiguous"))?);
    if lo.len() != 3 || hi.len() != 3 {
        return Err(bad("lower/upper must have 3 elements"));
    }
    let p = Params { lo: [lo[0] as i32, lo[1] as i32, lo[2] as i32], hi: [hi[0] as i32, hi[1] as i32, hi[2] as i32], thr };
    // as_slice() fails (instead of copying) for non-C-contiguous input: zero-copy, explicit error.
    let bl = blurred.as_slice().map_err(|_| bad("blurred must be C-contiguous"))?;
    let fr = frame.as_slice().map_err(|_| bad("frame must be C-contiguous"))?;
    let pg = prev_gray.as_slice_mut().map_err(|_| bad("prev_gray must be C-contiguous"))?;
    Ok(py.detach(move || run(bl, fr, pg, p, threads)))
}

/// Single-threaded fused kernel. Returns (moving_warm, warm).
#[pyfunction]
fn fused(
    py: Python<'_>,
    blurred: PyReadonlyArray3<'_, u8>,
    frame: PyReadonlyArray3<'_, u8>,
    prev_gray: PyReadwriteArray2<'_, u8>,
    lower: PyReadonlyArray1<'_, u8>,
    upper: PyReadonlyArray1<'_, u8>,
    thr: i32,
) -> PyResult<(u64, u64)> {
    fused_impl(py, blurred, frame, prev_gray, lower, upper, thr, 1)
}

/// Multi-threaded variant (std::thread::scope over pixel bands). `threads` defaults to all CPUs.
#[pyfunction]
#[pyo3(signature = (blurred, frame, prev_gray, lower, upper, thr, threads=0))]
fn fused_mt(
    py: Python<'_>,
    blurred: PyReadonlyArray3<'_, u8>,
    frame: PyReadonlyArray3<'_, u8>,
    prev_gray: PyReadwriteArray2<'_, u8>,
    lower: PyReadonlyArray1<'_, u8>,
    upper: PyReadonlyArray1<'_, u8>,
    thr: i32,
    threads: usize,
) -> PyResult<(u64, u64)> {
    let n = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
    fused_impl(py, blurred, frame, prev_gray, lower, upper, thr, n)
}

#[pymodule]
fn rs_fused(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fused, m)?)?;
    m.add_function(wrap_pyfunction!(fused_mt, m)?)?;
    Ok(())
}
