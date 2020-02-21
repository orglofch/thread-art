extern crate cgmath;
extern crate image;

use self::cgmath::{Array, MetricSpace, Vector2, Vector3};
use self::image::{GrayImage, Luma, RgbImage};

/// A simple flat image.
///
/// <T> - The underlying raster type.
pub(crate) struct Raster<T> {
    pub(crate) width: usize,
    pub(crate) height: usize,

    pub(crate) data: Vec<T>,
}

impl<T> Raster<T> where T: Copy {
    pub(crate) fn new(width: usize, height: usize, data: Vec<T>) -> Raster<T> {
        assert!(data.len() == width * height * 3);
        Raster {
            width: width,
            height: height,
            data: data,
        }
    }

    pub(crate) fn get(&self, x: usize, y: usize) -> Vector3<T> {
        let i = (x + y * self.width) * 3;
        assert!(i >= 0 && i < self.data.len());
        Vector3::new(self.data[i], self.data[i + 1], self.data[i + 2])
    }
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
/// `line_color` - The colour of the line to draw.
/// `img` - The image to write into.
pub(crate) fn draw_line(
    start: &Vector2<u32>,
    end: &Vector2<u32>,
    line_colour: &Vector3<u8>,
    img: &mut RgbImage,
) {
    debug_assert!(start.x < img.width());
    debug_assert!(start.y < img.height());
    debug_assert!(end.x < img.width());
    debug_assert!(end.y < img.height());

    let steep = is_steep(start, end);

    let mut p0 = start.map(|e| e as f64);
    let mut p1 = end.map(|e| e as f64);

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
            let mut val = *img.get_pixel(lower_y, x);
            for i in 0..3 {
                val.data[i] = (val.data[i] as f64 * fract +
                                   line_colour[i] as f64 * (1.0 - fract))
                    .min(255.0) as u8;
            }
            img.put_pixel(lower_y, x, val);
        } else {
            let mut val = *img.get_pixel(x, lower_y);
            for i in 0..3 {
                val.data[i] = (val.data[i] as f64 * fract +
                                   line_colour[i] as f64 * (1.0 - fract))
                    .min(255.0) as u8;
            }
            img.put_pixel(x, lower_y, val);
        };

        // Additive blend the upper-y pixel. Avoid calculating if the fractional part
        // is sufficiently small since the upper pixel may overflow
        // the image bounds due to floating point imprecision.
        if fract > 1.0e-6 {
            if steep {
                let mut val = *img.get_pixel(lower_y + 1, x);
                for i in 0..3 {
                    val.data[i] = (val.data[i] as f64 * (1.0 - fract) +
                                       line_colour[i] as f64 * fract)
                        .min(255.0) as u8;
                }
                img.put_pixel(lower_y + 1, x, val);
            } else {
                let mut val = *img.get_pixel(x, lower_y + 1);
                for i in 0..3 {
                    val.data[i] = (val.data[i] as f64 * (1.0 - fract) +
                                       line_colour[i] as f64 * fract)
                        .min(255.0) as u8;
                }
                img.put_pixel(x, lower_y + 1, val);
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
/// `target_img` - The image the integral is calculated along.
/// `current_image` - The current line raster.
/// `line_color` - The colour of the line.
/// `start` - The start of the line segment the integral is calculated along.
/// `end` - The end of the line segment the integral is calculated along.
pub(crate) fn line_fitness(
    target_img: &RgbImage,
    current_img: &RgbImage,
    start: &Vector2<u32>,
    end: &Vector2<u32>,
    line_colour: &Vector3<u8>,
) -> f64 {
    debug_assert!(start.x < target_img.width());
    debug_assert!(start.y < target_img.height());
    debug_assert!(end.x < target_img.width());
    debug_assert!(end.y < target_img.height());

    let steep = is_steep(start, end);

    let mut p0 = start.map(|e| e as f64);
    let mut p1 = end.map(|e| e as f64);

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
    let mut total = 0_i32; // TODO: Decide on a proper type for this.

    // Iterate across each x-position. For each x position calculate
    // the contribution of two pixels above and below the y-intercept.
    let mut y_intercept = p0.y;
    for x in (p0.x.round() as u32)..(p1.x.round() as u32 + 1) {
        let fract = y_intercept.fract();

        let lower_y = y_intercept as u32;

        // The lower-y pixel.
        let (target_val, current_val) = if steep {
            (
                target_img.get_pixel(lower_y, x),
                current_img.get_pixel(lower_y, x),
            )
        } else {
            (
                target_img.get_pixel(x, lower_y),
                current_img.get_pixel(x, lower_y),
            )
        };

        for i in 0..3 {
            let new_value = (current_val[i] as f64 * fract +
                                 line_colour[i] as f64 * (1.0 - fract))
                .min(255.0) as i32;
            if (new_value - current_val[i] as i32).abs() > 1 {
                changed = true;
            }
            total += (target_val[i] as i32 - current_val[i] as i32).abs() -
                (target_val[i] as i32 - new_value).abs();
        }

        // The upper-y pixel. Avoid calculating if the fractional part
        // is sufficiently small since the upper pixel may overflow
        // the image bounds due to floating point imprecision.
        if fract > 1.0e-6 {
            let (target_val, current_val) = if steep {
                (
                    target_img.get_pixel(lower_y + 1, x),
                    current_img.get_pixel(lower_y + 1, x),
                )
            } else {
                (
                    target_img.get_pixel(x, lower_y + 1),
                    current_img.get_pixel(x, lower_y + 1),
                )
            };
            for i in 0..3 {
                let new_value = (current_val[i] as f64 * (1.0 - fract) +
                                     line_colour[i] as f64 * fract)
                    .min(255.0) as i32;
                if (new_value - current_val[i] as i32).abs() > 1 {
                    changed = true;
                }
                total += (target_val[i] as i32 - current_val[i] as i32).abs() -
                    (target_val[i] as i32 - new_value).abs();
            }
        }

        y_intercept += gradient;
    }

    if !changed {
        total = std::i32::MIN;
    }
    return total as f64;
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
        let img = GrayImage::from_vec(4, 3, vec![0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
            .unwrap();

        let assert_integral_equals = |start: &Vector2<u32>, end: &Vector2<u32>, value: f64| {
            // Ensure the integral is the same regardless of direction.
            let forward = line_integral(&img, start, end);
            let backward = line_integral(&img, start, end);
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
