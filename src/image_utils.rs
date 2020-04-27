extern crate cgmath;
extern crate image;
extern crate rand;
extern crate rustfft;

use self::cgmath::{Array, MetricSpace, Vector2, Vector3};
use self::image::{GrayImage, Luma, RgbImage};
use self::rand::Rng;
use self::rustfft::FFTplanner;
use self::rustfft::num_complex::Complex;
use self::rustfft::num_traits::Zero;

use raster::*;

/// Returns the 'k' dominant colors for a `raster` using k-mean clustering.
pub(crate) fn k_dominant_colours(raster: &Raster<f32>, k: u32) -> Vec<Vec<f32>> {
    debug_assert!(k > 0);

    let mut pixels = Vec::with_capacity(
        raster.width as usize * raster.height as usize * raster.channels as usize,
    );
    for x in 0..raster.width {
        for y in 0..raster.height {
            let mut pixel = Vec::with_capacity(raster.channels as usize);
            for c in 0..raster.channels {
                pixel.push(raster.get(x, y, c));
            }
            pixels.push(pixel);
        }
    }

    let mut centers = Vec::with_capacity(k as usize);

    // Initialize random centers.
    let mut rng = rand::thread_rng();
    for i in 0..k {
        let mut colour = Vec::with_capacity(raster.channels as usize);
        for c in 0..raster.channels {
            colour.push(rng.gen::<f32>());
        }
        centers.push(colour);
    }

    let mut converged = false;
    let mut clusters: Vec<Vec<&Vec<f32>>> = vec![Vec::new(); centers.len()];
    while !converged {
        for i in 0..centers.len() {
            clusters[i].clear();
        }

        // Assign each pixel to a cluster.
        for pixel in &pixels {
            let mut closest_cluster_index = 0;
            let mut min_distance = std::f32::MAX;
            for j in 0..centers.len() {
                let center = &centers[j];

                let mut distance = 0.0;
                for c in 0..raster.channels {
                    let diff = center[c as usize] - pixel[c as usize];
                    distance = distance + diff * diff;
                }
                // TODO: sqrt needed?
                distance = distance.sqrt();
                if distance < min_distance {
                    closest_cluster_index = j;
                    min_distance = distance;
                }
            }
            clusters[closest_cluster_index].push(&pixel);
        }

        // Calculate new centers.
        let mut new_centers = Vec::with_capacity(k as usize);
        for i in 0..clusters.len() {
            let cluster = &clusters[i];
            if cluster.len() == 0 {
                // TODO: Randomize when this fails.
                new_centers.push(vec![centers[i][0], centers[i][1], centers[i][2]]);
                continue;
            }

            let mut sum = vec![0.0; raster.channels as usize];
            for pixel in cluster {
                for c in 0..raster.channels {
                    sum[c as usize] = sum[c as usize] + pixel[c as usize];
                }
            }
            for c in 0..raster.channels {
                sum[c as usize] = sum[c as usize] / cluster.len() as f32;
            }
            new_centers.push(sum);
        }

        // Check if we've converged and update the cluster centers.
        converged = true;
        for i in 0..new_centers.len() {
            for c in 0..raster.channels {
                if new_centers[i][c as usize] != centers[i][c as usize] {
                    centers[i][c as usize] = new_centers[i][c as usize];
                    converged = false;
                }
            }
        }
    }

    return centers;
}

/// Computes the 2D DFT of the given `raster`.
pub(crate) fn fft(raster: &Raster<Complex<f32>>, inverse: bool) -> Raster<Complex<f32>> {
    let mut result = Raster::new(
        raster.width,
        raster.height,
        raster.channels,
        vec![Complex::zero(); raster.data.len()],
    );
    // FFT each individual channel.
    // TODO: Switch to column major.
    for c in 0..raster.channels {
        // Convert the raster to a vector of complex numbers.
        let mut row_input = vec![Complex::zero(); raster.width as usize * raster.height as usize];
        for y in 0..raster.height {
            for x in 0..raster.width {
                row_input[x as usize + y as usize * raster.width as usize] = raster.get(x, y, c);
            }
        }

        let mut planner = FFTplanner::new(inverse);

        // Perform 1D fourier transforms of each row.
        // TODO: Use process_multi.
        let mut row_output = vec![Complex::zero(); raster.width as usize * raster.height as usize];
        let row_fft = planner.plan_fft(raster.width as usize);
        for y in 0..raster.height as usize {
            let offset = y * raster.width as usize;
            row_fft.process(
                &mut row_input[offset..(offset + raster.width as usize)],
                &mut row_output[offset..(offset + raster.width as usize)],
            );
        }

        // Perform 1D fourier transform of each column of fourier coefficients.
        let col_fft = planner.plan_fft(raster.height as usize);
        for x in 0..raster.width {
            let mut col_input = vec![Complex::zero(); raster.height as usize];
            for y in 0..raster.height {
                col_input[y as usize] = row_output[x as usize + y as usize * raster.width as usize];
            }
            let mut col_output = vec![Complex::zero(); raster.height as usize];
            col_fft.process(&mut col_input[..], &mut col_output[..]);

            for y in 0..raster.height {
                result.set(x, y, c, col_output[y as usize]);
            }
        }
    }

    // Normalize when computing the inverse.
    if inverse {
        for y in 0..raster.height {
            for x in 0..raster.width {
                for c in 0..raster.channels {
                    result.set(
                        x,
                        y,
                        c,
                        result.get(x, y, c) / (raster.width * raster.height) as f32,
                    );
                }
            }
        }
    }

    return result;
}

/// TODO create math operators on image.

/// Converts a real valued raster into a complex valued raster.
pub fn real_to_complex(raster: &Raster<f32>) -> Raster<Complex<f32>> {
    let mut result = Raster::new(
        raster.width,
        raster.height,
        raster.channels,
        vec![Complex::zero(); raster.data.len()],
    );
    for y in 0..raster.height {
        for x in 0..raster.width {
            for c in 0..raster.channels {
                result.set(x, y, c, Complex::new(raster.get(x, y, c), 0.0));
            }
        }
    }
    return result;
}

/// Converts a complex valued raster into a real valued raster.
pub fn complex_to_real(raster: &Raster<Complex<f32>>) -> Raster<f32> {
    let mut result = Raster::new(
        raster.width,
        raster.height,
        raster.channels,
        vec![0.0; raster.data.len()],
    );
    for y in 0..raster.height {
        for x in 0..raster.width {
            for c in 0..raster.channels {
                result.set(x, y, c, raster.get(x, y, c).re);
            }
        }
    }
    return result;
}

/// Applies a Gaussian blue to the given image.
pub fn gaussian_blur(raster: &Raster<f32>, sigma: f32) -> Raster<f32> {
    let gaussian = create_gaussian(sigma, raster.width, raster.height, raster.channels);
    return convolvef(&raster, &gaussian);
}

/// Normalizes a float image s.t. all values are <= 1.0.
pub fn normalize(raster: &mut Raster<f32>) {
    let mut max_val = 0.0_f32;
    for y in 0..raster.height {
        for x in 0..raster.width {
            for c in 0..raster.channels {
                max_val = max_val.max(raster.get(x, y, c));
            }
        }
    }

    for y in 0..raster.height {
        for x in 0..raster.width {
            for c in 0..raster.channels {
                raster.set(x, y, c, raster.get(x, y, c) / max_val);
            }
        }
    }
}

/// Copy an image.
///
/// TODO: Implement the trait?
pub fn copy(raster: &Raster<f32>) -> Raster<f32> {
    // TODO: We can just memcpy to make this faster.
    let mut copy = Raster::zero(raster.width, raster.height, raster.channels);
    for y in 0..raster.height {
        for x in 0..raster.width {
            for c in 0..raster.channels {
                copy.set(x, y, c, raster.get(x, y, c));
            }
        }
    }
    return copy;
}

/// Create image pyramid with the given number of `levels`.
///
/// The first level is a copy of the input raster.
pub fn create_image_pyramid(raster: &Raster<f32>, levels: u8) -> Vec<Raster<f32>> {
    let mut pyramid = Vec::with_capacity(levels as usize);
    for level in 0..levels as usize {
        if level == 0 {
            pyramid.push(copy(raster));
            continue;
        }

        let previous = &pyramid[level - 1];

        // Blur the previous pyramid level.
        let blurred = gaussian_blur(previous, 16.0);

        // Remove every even row & column.
        let mut pyramid_level =
            Raster::zero(previous.width / 2, previous.height / 2, raster.channels);
        for y in 0..pyramid_level.height {
            for x in 0..pyramid_level.width {
                for c in 0..raster.channels {
                    pyramid_level.set(x, y, c, previous.get(x * 2, y * 2, c));
                }
            }
        }

        pyramid.push(pyramid_level);
    }
    return pyramid;
}

/// Sum all levels of image pyramid.
///
/// The final image has the same size as the image at level 0.
pub fn sum_image_pyramid(pyramid: &Vec<Raster<f32>>) -> Raster<f32> {
    debug_assert!(!pyramid.is_empty());

    let first_level = &pyramid[0];

    let mut combined = Raster::zero(first_level.width, first_level.height, first_level.channels);
    for pyramid_level in pyramid {
        // Bilinearly upsample each level.
        for y in 0..combined.height {
            let py = (y as f32 / (combined.height - 1) as f32) * (pyramid_level.height - 1) as f32;
            let lower_y = (py - 0.5) as u32;
            let alpha_y = py - lower_y as f32;
            for x in 0..combined.width {
                let px = (x as f32 / (combined.width - 1) as f32) *
                    (pyramid_level.width - 1) as f32;
                let lower_x = (px - 0.5) as u32;
                let alpha_x = px - lower_x as f32;
                for c in 0..combined.channels {
                    let lower_x_val = pyramid_level.get(lower_x, lower_y, c) * (1.0 - alpha_x) +
                        pyramid_level.get(lower_x + 1, lower_y, c) * alpha_x;
                    let upper_x_val = pyramid_level.get(lower_x, lower_y + 1, c) * (1.0 - alpha_x) +
                        pyramid_level.get(lower_x + 1, lower_y + 1, c) * alpha_x;

                    let final_value = lower_x_val * (1.0 - alpha_y) + upper_x_val * alpha_y;

                    combined.set(x, y, c, combined.get(x, y, c) + final_value);
                }
            }
        }
    }
    return combined;
}

/// Returns a saliency map, where pixel intensity indicates pixel saliency.
///
/// This uses the spectral residual method described in
/// 'Saliency Detection: A Spectral Residual Approach'
/// http://bcmi.sjtu.edu.cn/~zhangliqing/Papers/2007CVPR_Houxiaodi_04270292.pdf.
pub(crate) fn saliency(raster: &Raster<f32>) -> Raster<f32> {
    // Construct pyramid levels.
    let pyramid = create_image_pyramid(raster, 6);

    let mut saliency_pyramid = Vec::with_capacity(pyramid.len());
    for pyramid_level in pyramid {
        let complex = real_to_complex(&pyramid_level);

        // TODO: Downsample.

        let a = fft(
            &complex,
            /*inverse=*/
            false,
        );

        // TODO: Use complex_to_real or make it work for this function.
        // TODO: Rename everything.
        let mut angle = Raster::zero(
            pyramid_level.width,
            pyramid_level.height,
            pyramid_level.channels,
        );
        let mut log_amplitude = Raster::zero(
            pyramid_level.width,
            pyramid_level.height,
            pyramid_level.channels,
        );
        for y in 0..pyramid_level.height {
            for x in 0..pyramid_level.width {
                for c in 0..pyramid_level.channels {
                    let val = a.get(x, y, c);
                    angle.set(x, y, c, Complex::new(val.im.atan2(val.re), 0.0));
                    // TODO: log(10)?
                    log_amplitude.set(
                        x,
                        y,
                        c,
                        Complex::new((val.re * val.re + val.im * val.im).sqrt().ln(), 0.0),
                    );
                }
            }
        }

        // Apply a box filter to the amplitude.
        let filter = real_to_complex(&create_box_filter(
            /*size=*/
            5,
            pyramid_level.width,
            pyramid_level.height,
            pyramid_level.channels,
        ));
        let filter_freq = fft(
            &filter,
            /*inverse=*/
            false,
        );
        let amplitude_freq = fft(
            &log_amplitude,
            /*inverse=*/
            false,
        );
        // TODO: Use the convolve function but make it inversible.
        let mut mult_amplitude = Raster::zero(
            pyramid_level.width,
            pyramid_level.height,
            pyramid_level.channels,
        );
        for y in 0..pyramid_level.height {
            for x in 0..pyramid_level.width {
                for c in 0..pyramid_level.channels {
                    mult_amplitude.set(
                        x,
                        y,
                        c,
                        filter_freq.get(x, y, c) * amplitude_freq.get(x, y, c),
                    );
                }
            }
        }
        let mut filtered_amplitude = fft(
            &mult_amplitude,
            /*inverse=*/
            true,
        );

        let mut final_amplitude = Raster::zero(
            pyramid_level.width,
            pyramid_level.height,
            pyramid_level.channels,
        );
        for y in 0..pyramid_level.height {
            for x in 0..pyramid_level.width {
                for c in 0..pyramid_level.channels {
                    final_amplitude.set(
                        x,
                        y,
                        c,
                        log_amplitude.get(x, y, c) - filtered_amplitude.get(x, y, c),
                    );
                }
            }
        }

        let mut result_freq = Raster::zero(
            pyramid_level.width,
            pyramid_level.height,
            pyramid_level.channels,
        );
        for y in 0..pyramid_level.height {
            for x in 0..pyramid_level.width {
                for c in 0..pyramid_level.channels {
                    let ang = angle.get(x, y, c).re;
                    let amp = final_amplitude.get(x, y, c).re.exp();
                    result_freq.set(x, y, c, Complex::new(amp * ang.cos(), amp * ang.sin()));
                }
            }
        }

        let mut real_result = complex_to_real(&fft(
            &result_freq,
            /*inverse=*/
            true,
        ));

        // Square the results.
        for y in 0..pyramid_level.height {
            for x in 0..pyramid_level.width {
                for c in 0..pyramid_level.channels {
                    let val = real_result.get(x, y, c);
                    real_result.set(x, y, c, val * val);
                }
            }
        }

        real_result = gaussian_blur(&real_result, 4.0);

        normalize(&mut real_result);

        saliency_pyramid.push(real_result);
    }

    let mut combined_saliency = sum_image_pyramid(&saliency_pyramid);
    normalize(&mut combined_saliency);

    return combined_saliency;
}

/// Convolves two complex rasters.
fn convolve(
    raster_a: &Raster<Complex<f32>>,
    raster_b: &Raster<Complex<f32>>,
) -> Raster<Complex<f32>> {
    let mut fft_a = fft(
        &raster_a,
        /*inverse=*/
        false,
    );
    let fft_b = fft(
        &raster_b,
        /*inverse=*/
        false,
    );

    // TODO: Don't require them to be the same size.
    // TODO: Mult trait.
    for y in 0..raster_a.height {
        for x in 0..raster_a.width {
            for c in 0..raster_a.channels {
                let val_a = fft_a.get(x, y, c);
                let val_b = fft_b.get(x, y, c);
                fft_a.set(x, y, c, val_a * val_b);
            }

        }
    }

    return fft(
        &fft_a,
        /*inverse=*/
        true,
    );
}

/// Convolves two real valued rasters.
/// TODO: Use Traits to overload this?
pub(crate) fn convolvef(raster_a: &Raster<f32>, raster_b: &Raster<f32>) -> Raster<f32> {
    return complex_to_real(&convolve(
        &real_to_complex(raster_a),
        &real_to_complex(raster_b),
    ));
}

/// Returns the error residuals between two RGBs.
/*pub(crate) fn get_error_residual(img_1: &RgbImage, img_2: &RgbImage) -> RgbImage {
    debug_assert!(img_1.width() == img_2.width());
    debug_assert!(img_1.height() == img_2.height());

    let data = vec![0; (img_1.width() * img_2.height()) as usize];
    let mut residuals = RgbImage::from_raw(img_1.width(), img_1.height(), data).unwrap();
    for y in 0..img_1.height() {
        for x in 0..img_1.width() {
            let pixel_1 = img_1.get_pixel(x, y);
            let pixel_2 = img_2.get_pixel(x, y);
            let mut difference : [u8; 3];
            for c in 0..3 {
                // TODO: Maybe l2 loss but make sure it fits the data type.
                difference[c] = (pixel_1.data[c] - pixel_2.data[c]);
            }
            residuals.put_pixel(x, y, Rgb { data: difference});
        }
    }
    return residuals;
}*/
/// Returns a saliency map for the given RGB image.
///
/// Arguments:
/// `img` - The image to generate the saliency map for.
pub(crate) fn create_saliency_map(img: &RgbImage) -> GrayImage {
    let max_dimension = std::cmp::min(img.height(), img.width()) as f64;

    let mut raw_saliency = vec![0.0; (img.width() * img.height()) as usize];

    let window = 20; // pixels.

    // TODO: Use an integral image to speed this up.
    let mut max_saliency = std::f64::MIN;
    for y in 0..img.height() {
        for x in 0..img.width() {
            let pixel = {
                let pixel = img.get_pixel(x as u32, y as u32);
                Vector3::from(pixel.data).cast::<f64>().unwrap() / 255.0
            };
            let position = Vector2::new(x as f64, y as f64);

            let mut saliency = 0.1;
            for y2 in std::cmp::max(0, y - window)..std::cmp::min(y + window, img.height()) {
                for x2 in std::cmp::max(0, x - window)..std::cmp::min(x + window, img.width()) {
                    let neighbor = {
                        let pixel = img.get_pixel(x2 as u32, y2 as u32);
                        Vector3::from(pixel.data).cast::<f64>().unwrap() / 255.0
                    };
                    let neighbor_position = Vector2::new(x2 as f64, y2 as f64);

                    // TODO: Geometric distance between colors doesn't work well for RGBs.
                    let color_distance = pixel.distance(neighbor).powf(2.0);
                    let spatial_distance = position.distance(neighbor_position);

                    // Weight the color difference by the pixel distance.
                    saliency += color_distance * spatial_distance;
                }
            }
            if saliency > max_saliency {
                max_saliency = saliency;
            }

            raw_saliency[(x + y * img.width()) as usize] = saliency;
        }
    }

    let mut normalized_saliency = vec![0; (img.width() * img.height()) as usize];
    for i in 0..normalized_saliency.len() {
        normalized_saliency[i] = (raw_saliency[i] / max_saliency * 255.0) as u8;
    }
    GrayImage::from_raw(img.width(), img.height(), normalized_saliency).unwrap()
}

/// Writes a single line into the given image.
///
/// Lines are written using Xiaolin Wu's line drawing algorithm.
///
/// Arguments:
///
/// `start` - The start of the line segment to draw.
/// `end` - The end of the line segment to draw.
/// `line_colour` - The dequantized line colour.
/// `raster` - The raster to write into.
pub(crate) fn draw_line(
    start: &Vector2<u32>,
    end: &Vector2<u32>,
    line_colour: &Vector3<f32>,
    raster: &mut Raster<f32>,
) {
    debug_assert!(start.x < raster.width as u32);
    debug_assert!(start.y < raster.height as u32);
    debug_assert!(end.x < raster.width as u32);
    debug_assert!(end.y < raster.height as u32);

    let steep = is_steep(start, end);

    let mut p0 = start.map(|e| e as f32);
    let mut p1 = end.map(|e| e as f32);

    // If the line is 'steep', iterate over y rather than x.
    if steep {
        p0.swap_elements(0, 1);
        p1.swap_elements(0, 1);
    }

    // Move the start and end positions so we're always iterating forward.
    if p0.x > p1.x {
        let tmp = p0;
        p0 = p1;
        p1 = tmp;
    }

    let delta = p1 - p0;
    let gradient = if delta.x == 0.0 {
        1.0
    } else {
        delta.y / delta.x
    };

    // Iterate across each x-position. For each x position calculate
    // the contribution of two pixels above and below the y-intercept.
    let mut y_intercept = p0.y;
    for x in (p0.x.round() as u32)..(p1.x.round() as u32 + 1) {
        let fract = y_intercept.fract();

        let lower_y = y_intercept as u32;

        // Additive blend the lower-y pixel.
        // TODO: Use different blend scheme?
        if steep {
            for c in 0..3 {
                let mut val = raster.get(lower_y, x, c);
                val = val * fract + line_colour[c as usize] * (1.0 - fract);
                raster.set(lower_y, x, c, val);
            }
        } else {
            for c in 0..3 {
                let mut val = raster.get(x, lower_y, c);
                val = val * fract + line_colour[c as usize] * (1.0 - fract);
                raster.set(x, lower_y, c, val);
            }
        };

        // Additive blend the upper-y pixel. Avoid calculating if the fractional part
        // is sufficiently small since the upper pixel may overflow
        // the image bounds due to floating point imprecision.
        if fract > 1.0e-6 {
            if steep {
                for c in 0..3 {
                    let mut val = raster.get(lower_y + 1, x, c);
                    val = val * (1.0 - fract) + line_colour[c as usize] * fract;
                    raster.set(lower_y + 1, x, c, val);
                }
            } else {
                for c in 0..3 {
                    let mut val = raster.get(x, lower_y + 1, c);
                    val = val * (1.0 - fract) + line_colour[c as usize] * fract;
                    raster.set(x, lower_y + 1, c, val);
                }
            };
        }

        y_intercept += gradient;
    }
}

/// Returns the 'fitness' of rendering a given line into `current_img`,
/// relative to `target_img`.
///
/// Higher-fitness is better. Negative fitness indicates that adding a line
/// will decrease the similarity to the `target_img`.
///
/// The approximation is very similar to Xiaolin Wu's line drawing algorithm
/// but instead of rendering the line, we give higher fitness to lines which
/// cause `current_img` to more closely match `target_img`.
///
/// The fitness contributed per-pixel is equal to the change in difference
/// of the image colors with the lines contributions:
///
/// i.e. fitness = |target_image - current_image| -
///                |target_image - (current_image + line)|
///
/// Arguments:
///
/// `target_raster` - The image the integral is calculated along.
/// `current_raster` - The current line raster.
/// `start` - The start of the line segment the integral is calculated along.
/// `end` - The end of the line segment the integral is calculated along.
/// `line_color` - The dequantized line color.
pub(crate) fn line_fitness(
    target_raster: &Raster<f32>,
    current_raster: &Raster<f32>,
    start: &Vector2<u32>,
    end: &Vector2<u32>,
    line_colour: &Vector3<f32>,
) -> f32 {
    debug_assert!(start.x < target_raster.width as u32);
    debug_assert!(start.y < target_raster.height as u32);
    debug_assert!(end.x < target_raster.width as u32);
    debug_assert!(end.y < target_raster.height as u32);

    let steep = is_steep(start, end);

    let mut p0 = start.map(|e| e as f32);
    let mut p1 = end.map(|e| e as f32);

    // If the line is 'steep', iterate over y rather than x.
    if steep {
        p0.swap_elements(0, 1);
        p1.swap_elements(0, 1);
    }

    // Move the start and end positions so we're always iterating forward.
    if p0.x > p1.x {
        let tmp = p0;
        p0 = p1;
        p1 = tmp;
    }

    let delta = p1 - p0;
    let gradient = if delta.x == 0.0 {
        1.0
    } else {
        delta.y / delta.x
    };

    let mut changed = false;
    let mut total = 0_f32;

    // Iterate across each x-position. For each x position calculate
    // the contribution of two pixels above and below the y-intercept.
    let mut y_intercept = p0.y;
    for x in (p0.x.round() as u32)..(p1.x.round() as u32 + 1) {
        let fract = y_intercept.fract();

        let lower_y = y_intercept as u32;

        for c in 0..3 {
            // The lower-y pixel.
            let (target_val, current_val) = if steep {
                (
                    target_raster.get(lower_y, x, c),
                    current_raster.get(lower_y, x, c),
                )
            } else {
                (
                    target_raster.get(x, lower_y, c),
                    current_raster.get(x, lower_y, c),
                )
            };

            let new_value = (current_val * fract + line_colour[c as usize] * (1.0 - fract))
                .min(1.0);
            if (new_value - current_val).abs() > 0.01 {
                // TODO: Arbitrary.
                changed = true;
            }
            total += (target_val - current_val).abs() - (target_val - new_value).abs();
        }

        // The upper-y pixel. Avoid calculating if the fractional part
        // is sufficiently small since the upper pixel may overflow
        // the image bounds due to floating point imprecision.
        if fract > 1.0e-6 {
            for c in 0..3 {
                let (target_val, current_val) = if steep {
                    (
                        target_raster.get(lower_y + 1, x, c),
                        current_raster.get(lower_y + 1, x, c),
                    )
                } else {
                    (
                        target_raster.get(x, lower_y + 1, c),
                        current_raster.get(x, lower_y + 1, c),
                    )
                };

                let new_value = (current_val * (1.0 - fract) + line_colour[c as usize] * fract)
                    .min(1.0);
                if (new_value - current_val).abs() > 0.01 {
                    // TODO: Arbitrary.
                    changed = true;
                }
                total += (target_val - current_val).abs() - (target_val - new_value).abs();
            }
        }

        y_intercept += gradient;
    }

    if !changed {
        total = std::f32::MIN;
    }
    return total;
}

/// Approximates the discrete line integral at a single point.
///
/// Note, this function needs to use the same integral calculation as the original
/// integral calculation. It isn't sufficient to simply bilinearly interpolated intensity
/// since the line integral is dependant on the orientation of the line.
///
/// Arguments:
///
/// `img` - The image the integral is calculated along.
/// `start` - The start of the line segment the integral is calculated along.
/// `end` - The end of the line segment the integral is calculated along.
/// `point` - The point to remove from the line segment.
///
/// TODO: Probably remove this.
/// TODO: Combine with the function below and inline.
/// TODO: Multiple lines overlapping at near points will double remove values.
pub(crate) fn line_integral_at_point(
    img: &GrayImage,
    start: &Vector2<u32>,
    end: &Vector2<u32>,
    point: &Vector2<f32>,
) -> f64 {
    let steep = is_steep(start, end);

    let mut p = point.map(|e| e as f64);

    // TODO: debug assertions that point lies along start / end.
    // If the line is 'steep', iterate over y rather than x.
    if steep {
        p.swap_elements(0, 1);
    }

    let fract = p.y.fract();

    let x = p.x as u32;

    let lower_y = p.y as u32;

    let mut total = 0.0;

    let pixel = if steep {
        img.get_pixel(lower_y, x)
    } else {
        img.get_pixel(x, lower_y)
    };
    total += (pixel.data[0] as f64 / 255.0) * (1.0 - fract);

    // The upper-y pixel. Avoid calculating if the fractional part
    // is sufficiently small since the upper pixel may overflow
    // the image bounds due to floating point imprecision.
    if fract > 1.0e-6 {
        let pixel = if steep {
            img.get_pixel(lower_y + 1, x)
        } else {
            img.get_pixel(x, lower_y + 1)
        };
        total += (pixel.data[0] as f64 / 255.0) * fract;
    }

    return total;
}

/// Returns whether a line is considered "steep".
///
/// A steep line is any line with abs(slope) > 1.0.
#[inline]
fn is_steep(start: &Vector2<u32>, end: &Vector2<u32>) -> bool {
    (end.y as f32 - start.y as f32).abs() > (end.x as f32 - start.x as f32).abs()
}

/// Create the gradient of an image.
fn create_gradient_img(img: &GrayImage) -> GrayImage {
    return GrayImage::from_fn(img.width(), img.height(), |x, y| {
        if x == 0 || x == img.width() - 1 || y == 0 || y == img.height() - 1 {
            return Luma { data: [0] };
        }

        // Find the gradient magnitude image using central differencing.
        let dx = (img.get_pixel(x + 1, y)[0] as f64 - img.get_pixel(x - 1, y)[0] as f64) / 2.0;
        let dy = (img.get_pixel(x, y + 1)[0] as f64 - img.get_pixel(x, y - 1)[0] as f64) / 2.0;

        // TODO: Make this just a float image.
        return Luma { data: [(dx.powf(2.0) + dy.powf(2.0)).sqrt() as u8] };
    });
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_error_residual() {
        let img_1 = GrayImage::from_raw(3, 2, vec![1, 2, 3, 2, 4, 6]).unwrap();
        let img_2 = GrayImage::from_raw(3, 2, vec![3, 2, 1, 4, 2, 0]).unwrap();

        let residual = get_error_residual(&img_1, &img_2).into_raw();

        assert_eq!(residual, [4, 0, 4, 4, 4, 36]);
    }

    #[test]
    fn test_line_integral() {
        // 40 45 50 55
        // 20 25 30 35
        //  0  5 10 15
        let raster = Raster::new(4, 3, 1, vec![0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]);

        let assert_integral_equals = |start: &Vector2<u32>, end: &Vector2<u32>, value: f64| {
            // Ensure the integral is the same regardless of direction.
            let forward = line_integral(&raster, start, end);
            let backward = line_integral(&raster, start, end);
            assert_eq!(forward, backward);
            assert_approx_eq!(forward, value);
        };

        // 40 45 50 55
        // 20 25 30 35
        //  X  5 10 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(0, 0), 0.0);

        // 40 45 50 55
        // 20 25 30 35
        //  X  X 10 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(1, 0), 0.01960784313);

        // 40 45 50 55
        // 20 25 30 35
        //  X  X  X 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(2, 0), 0.05882352941);

        // 40 45 50 55
        //  X 25 30 35
        //  X  5 10 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(0, 1), 0.07843137254);

        //  X 45 50 55
        //  X 25 30 35
        //  X  5 10 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(0, 2), 0.23529411764);

        // 40 45 50 55
        // 20  X 30 35
        //  X  5 10 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(1, 1), 0.09803921568);

        // 40 45  X 55
        // 20  X 30 35
        //  X  5 10 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(2, 2), 0.29411764705);

        //  X 45 50 55
        // 20  X 30 35
        //  0  5 10 15
        assert_integral_equals(&Vector2::new(0, 2), &Vector2::new(1, 1), 0.25490196078);

        //  X 45 50 55
        // 20  X 30 35
        //  0  5  X 15
        assert_integral_equals(&Vector2::new(0, 2), &Vector2::new(2, 0), 0.29411764705);

        //  X 45 50 55
        //  x  x 30 35
        //  0  X 10 15
        assert_integral_equals(&Vector2::new(0, 2), &Vector2::new(1, 0), 0.26470588235);

        // 40  X 50 55
        //  x  x 30 35
        //  X  5 10 15
        assert_integral_equals(&Vector2::new(0, 0), &Vector2::new(1, 2), 0.26470588235);

        // 40 45  x  X
        // 10  x  x 35
        //  X  x 10 20
        assert_integral_equals(&Vector2::new(3, 2), &Vector2::new(0, 0), 0.43137254902);
    }

    #[test]
    fn test_line_integral_at_point() {
        // 60 70 80
        // 30 40 50
        //  0 10 20
        let img = GrayImage::from_vec(3, 3, vec![0, 10, 20, 30, 40, 50, 60, 70, 80]).unwrap();

        // TODO: Better naming.
        let assert_line_integral_at_point_equals =
            |start: &Vector2<u32>, end: &Vector2<u32>, p3: &Vector2<f32>, value: f64| {
                // Ensure the integral is the same regardless of direction.
                let forward = line_integral_at_point(&img, start, end, p3);
                let backward = line_integral_at_point(&img, start, end, p3);
                assert_eq!(forward, backward);
                assert_approx_eq!(forward, value);
            };

        // 60 70  _
        // 30  _ 50
        //  X 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 3),
            &Vector2::new(0.0, 0.0),
            0.0,
        );

        // 60 70  _
        // 30  X 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 3),
            &Vector2::new(1.0, 1.0),
            0.15686274509,
        );

        // 60 70  X
        // 30  _ 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 3),
            &Vector2::new(2.0, 2.0),
            0.31372549019,
        );

        // 60 70  _
        // 30  x 50
        //  _  x 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 3),
            &Vector2::new(1.0, 0.5),
            0.09803921568,
        );

        // 60  x  _
        // 30  x 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 3),
            &Vector2::new(1.0, 1.5),
            0.21568627451,
        );

        // 60 70  X
        // 30  _ 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 3),
            &Vector2::new(2.0, 2.0),
            0.31372549019,
        );

        // 60 70 80
        // 30 40 50
        //  X  _  _
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 0),
            &Vector2::new(0.0, 0.0),
            0.0,
        );

        // 60 70 80
        // 30 40 50
        //  _  X  _
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 0),
            &Vector2::new(1.0, 0.0),
            0.03921568627,
        );

        // 60 70 80
        // 30 40 50
        //  _  _  X
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 0),
            &Vector2::new(2.0, 0.0),
            0.07843137254,
        );

        // 60 70 80
        // 30  x 50
        //  _  x  _
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 0),
            &Vector2::new(1.0, 0.5),
            0.09803921568,
        );

        // 60 70 80
        // 30 40 50
        //  _  x  x
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(3, 0),
            &Vector2::new(1.5, 0.0),
            0.03921568627,
        );

        //  _ 70 80
        //  _ 40 50
        //  X 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(0, 3),
            &Vector2::new(0.0, 0.0),
            0.0,
        );

        //  _ 70 80
        //  X 40 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(0, 3),
            &Vector2::new(0.0, 1.0),
            0.11764705882,
        );

        //  X 70 80
        //  _ 40 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(0, 3),
            &Vector2::new(0.0, 2.0),
            0.23529411764,
        );

        //  _ 70 80
        //  x  x 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(0, 3),
            &Vector2::new(0.5, 1.0),
            0.13725490196,
        );

        //  x 70 80
        //  x  _ 50
        //  _ 10 20
        assert_line_integral_at_point_equals(
            &Vector2::new(0, 0),
            &Vector2::new(0, 3),
            &Vector2::new(0.0, 1.5),
            0.11764705882,
        );
    }
}
