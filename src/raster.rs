extern crate rustfft;

use self::rustfft::num_traits::Zero;

/// A simple flat raster image.
///
/// <T> - The underlying raster type.
pub struct Raster<T> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u32,

    pub(crate) data: Vec<T>,
}

impl<T> Raster<T>
where
    T: Copy + Zero,
{
    /// Constructs a new raster.
    pub(crate) fn new(width: u32, height: u32, channels: u32, data: Vec<T>) -> Raster<T> {
        debug_assert!(data.len() == width as usize * height as usize * channels as usize);
        Raster {
            width: width,
            height: height,
            channels: channels,
            data: data,
        }
    }

    /// Constructs a raster with all zeros.
    pub(crate) fn zero(width: u32, height: u32, channels: u32) -> Raster<T> {
        Raster {
            width: width,
            height: height,
            channels: channels,
            data: vec![T::zero(); width as usize * height as usize * channels as usize],
        }
    }

    /// Normalizes a u8 Raster into a f32 Raster.
    pub(crate) fn normalize(source: Raster<f32>) -> Raster<u8> {
        let mut result = Raster::zero(source.width, source.height, source.channels);
        for y in 0..source.height {
            for x in 0..source.width {
                for c in 0..source.channels {
                    result.set(x, y, c, (source.get(x, y, c) * 255.0) as u8);
                }
            }
        }
        return result;
    }

    /// Denormalizes a f32 Raster into a u8 Raster.
    pub(crate) fn denormalize(source: Raster<u8>) -> Raster<f32> {
        let mut result = Raster::zero(source.width, source.height, source.channels);
        for y in 0..source.height {
            for x in 0..source.width {
                for c in 0..source.channels {
                    result.set(x, y, c, source.get(x, y, c) as f32 / 255.0);
                }
            }
        }
        return result;
    }

    /// Gets a cells value.
    pub(crate) fn get(&self, x: u32, y: u32, c: u32) -> T {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        debug_assert!(c < self.channels);
        let offset = c as usize + (x as usize + y as usize * self.width as usize) * self.channels as usize;
        debug_assert!(offset >= 0 && offset < self.data.len());
        self.data[offset]
    }

    /// Sets a cells value.
    pub(crate) fn set(&mut self, x: u32, y: u32, c: u32, data: T) {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        debug_assert!(c < self.channels);
        let offset = c as usize + (x as usize + y as usize * self.width as usize) * self.channels as usize;
        debug_assert!(offset >= 0 && offset < self.data.len());
        self.data[offset] = data;
    }

    pub(crate) fn saveRgb(&self, filename: &str) {
        // TODO: Check the data is RGB.
    }
}

/// Returns a box filter, padded to the given `width` and `height`.
/// TODO: Remove channels? It doesn't make a ton of sense.
/// TODO: Support complex numbers here somehow?
/// TODO: Create a Raster method which takes a function to construct a filter.
pub(crate) fn create_box_filter(size: i32, width: u32, height: u32, channels: u32) -> Raster<f32> {
    // TODO: Checks that size in in range.
    let mut filter = Raster::zero(width, height, channels);
    let weight = 1.0 / (size as f32 * size as f32);
    let half_size = size / 2;
    for f_y in -half_size..=half_size {
        for f_x in -half_size..=half_size {
            let x = (f_x + width as i32) % width as i32;
            let y = (f_y + height as i32) % height as i32;
            for c in 0..channels {
                filter.set(x as u32, y as u32, c, weight);
            }
        }
    }
    return filter;
}

/// Returns a gaussian filter with the given `width` and `height`.
pub(crate) fn create_gaussian(sigma: f32, width: u32, height: u32, channels: u32) -> Raster<f32> {
    // TODO: This is wrong / naive?
    // TODO: Checks that sigma is in range.
    let mut filter = Raster::zero(width, height, channels);
    let constant = 1.0 / (2.0 * std::f32::consts::PI * sigma * sigma).sqrt();
    // Only populate pixels within 3 standard deviations. Everywhere else should be close to 0.
    let range = (sigma * 3.0) as i32;
    for f_y in -range..=range {
        for f_x in -range..=range {
            let x = ((f_x + width as i32) % width as i32) as f32;
            let y = ((f_y + height as i32) % height as i32) as f32;
            let dx = f_x as f32;
            let dy = f_y as f32;
            for c in 0..channels {
                // TODO: This could all be faster by factoring out divides.
                filter.set(x as u32, y as u32, c, constant * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp());
            }
        }
    }
    return filter;
}
