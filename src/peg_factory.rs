extern crate cgmath;
extern crate rand;

use cgmath::{ElementWise, Vector2};
use rand::Rng;

/// Patterns for peg layout.
pub(crate) enum PegPattern {
    Uniform,
    Rect,
    Oval,
    ConcentricOval,
    Spiral,
    Random,
}

/// Creates a collection of pegs to wrap threads around.
///
/// Arguments:
///
/// `width` - The width of the area pegs can be placed within.
/// `height` - The height of the area pegs can be placed within.
/// `num_pegs` - The number of pegs to create.
/// `pattern` - The pattern to draw pegs from.
pub(crate) fn create_pegs(
    width: u32,
    height: u32,
    num_pegs: u32,
    pattern: PegPattern,
) -> Vec<Vector2<u32>> {
    match pattern {
        PegPattern::Uniform => create_uniform_peg_pattern(width, height, num_pegs),
        PegPattern::Rect => create_rect_peg_pattern(width, height, num_pegs),
        PegPattern::Oval => create_oval_peg_pattern(width, height, num_pegs),
        PegPattern::ConcentricOval => create_concentric_oval_pattern(width, height, num_pegs),
        PegPattern::Spiral => create_spiral_peg_pattern(width, height, num_pegs),
        PegPattern::Random => create_random_peg_pattern(width, height, num_pegs),
    }
}

impl std::str::FromStr for PegPattern {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Uniform" => Ok(PegPattern::Uniform),
            "Rect" => Ok(PegPattern::Rect),
            "Oval" => Ok(PegPattern::Oval),
            "ConcentricOval" => Ok(PegPattern::ConcentricOval),
            "Spiral" => Ok(PegPattern::Spiral),
            "Random" => Ok(PegPattern::Random),
            _ => Err("Failed to parse PegPattern from string"),
        }
    }
}

/// Creates pegs in a uniform pattern within a grid of size (width, height).
///
/// Note: This doesn't necessarily have vertical or horizontal symmetry.
///  ___________
/// |+ +  +  + +|
/// |+ +  +  + +|
/// |+ +  +  + +|
/// |+ +  +  + +|
/// |+_+__+__+_+|
fn create_uniform_peg_pattern(width: u32, height: u32, num_pegs: u32) -> Vec<Vector2<u32>> {
    let area = width * height;

    let pegs_sqrt = (num_pegs as f32).sqrt();

    let peg_x_step = (width as f32 / pegs_sqrt) as usize;
    let peg_y_step = (height as f32 / pegs_sqrt) as usize;

    let mut pegs = Vec::new();
    for x in (0..width).step_by(peg_x_step) {
        for y in (0..height).step_by(peg_y_step) {
            pegs.push(Vector2::new(x, y));
        }
    }
    return pegs;
}


/// Creates pegs in a rectangular pattern within a grid of size (width, height).
///
/// Note: This doesn't necessarily have vertical or horizontal symmetry.
///  ___________
/// |+ +  +  + +|
/// |+         +|
/// |+         +|
/// |+         +|
/// |+_+__+__+_+|
fn create_rect_peg_pattern(width: u32, height: u32, num_pegs: u32) -> Vec<Vector2<u32>> {
    let perimeter = (width + height) * 2;
    let distance_between_pegs = perimeter as u32 / num_pegs;

    let mut pegs = Vec::new();
    for i in 0..num_pegs {
        let distance = distance_between_pegs * i;
        if distance < width {
            // Top.
            pegs.push(Vector2::new(distance, height - 1))
        } else if distance < width + height {
            // Left.
            pegs.push(Vector2::new(0, distance - width))
        } else if distance < width * 2 + height {
            // Bottom.
            pegs.push(Vector2::new(distance - width - height, 0));
        } else {
            // Right.
            pegs.push(Vector2::new(width - 1, distance - width * 2 - height));
        }
    }
    return pegs;
}

/// Creates pegs in an oval pattern within a grid of size (width, height).
///
/// Note: This doesn't necessarily have vertical or horizontal symmetry.
///  ___________
/// |   + + +   |
/// | +       + |
/// |+         +|
/// | +       + |
/// |___+_+_+___|
fn create_oval_peg_pattern(width: u32, height: u32, num_pegs: u32) -> Vec<Vector2<u32>> {
    let rad_per_peg = std::f32::consts::PI * 2.0 / num_pegs as f32;

    // Apply the -1 offset directly to the result rounds to a 0 indexed cell.
    let center_cell = Vector2::new(((width - 1) / 2) as f32, ((height - 1) / 2) as f32);

    let mut pegs = Vec::new();
    for i in 0..num_pegs {
        let rad = rad_per_peg * i as f32;
        let offset = Vector2::new(rad.cos(), rad.sin()).mul_element_wise(center_cell);

        pegs.push(
            Vector2::cast::<u32>(&center_cell.add_element_wise(offset)).unwrap(),
        );
    }
    return pegs;
}

/// Creats pegs in an concentric oval pattern.
///
/// Note: This doesn't necessarily have vertical or horizonta symmetry.
///  ___________
/// |   + + +   |
/// | +  +++  + |
/// |+  +   +  +|
/// | +  +++  + |
/// |___+_+_+___|
fn create_concentric_oval_pattern(width: u32, height: u32, num_pegs: u32) -> Vec<Vector2<u32>> {
    // Determine the number of ovals.
    let k_pixels_between_ovals = 50;
    let ovals = (width as f32 / 2.0 / k_pixels_between_ovals as f32) as u32;

    let pegs_per_oval = num_pegs / ovals;
    let rad_per_peg = std::f32::consts::PI * 2.0 / pegs_per_oval as f32;

    // Apply the -1 offset directly to the result rounds to a 0 indexed cell.
    let center_cell = Vector2::new(((width - 1) / 2) as f32, ((height - 1) / 2) as f32);

    let mut pegs = Vec::new();
    for i in 0..ovals {
        let axis = center_cell -
            Vector2::new(
                (k_pixels_between_ovals * i) as f32,
                (k_pixels_between_ovals * i) as f32,
            );

        for j in 0..pegs_per_oval {
            let rad = rad_per_peg * j as f32;
            let offset = Vector2::new(rad.cos(), rad.sin()).mul_element_wise(axis);

            pegs.push(
                Vector2::cast::<u32>(&center_cell.add_element_wise(offset)).unwrap(),
            );
        }
    }
    return pegs;
}


/// Creates pegs in a spiral pattern.
///  ___________
/// |  +  +  +  |
/// | +  +++  + |
/// |+  +   +  +|
/// | +      +  |
/// |___+_+_+___|
fn create_spiral_peg_pattern(width: u32, height: u32, num_pegs: u32) -> Vec<Vector2<u32>> {
    let mut pegs = Vec::new();

    return pegs;
}

/// Create pegs in a random pattern within a grid of size (width, height).
///  ___________
/// | +      + +|
/// |  +   +    |
/// |+        + |
/// |  +  +  +  |
/// |_+___+____+|
fn create_random_peg_pattern(width: u32, height: u32, num_pegs: u32) -> Vec<Vector2<u32>> {
    let mut rng = rand::thread_rng();

    let mut pegs = Vec::new();
    for _ in 0..num_pegs {
        let x = rng.gen_range(0, width);
        let y = rng.gen_range(0, height);
        pegs.push(Vector2::new(x, y));
    }
    return pegs;
}

#[cfg(test)]
mod test {
    use super::*;
    use cgmath::prelude::InnerSpace;

    #[test]
    fn test_create_rect_peg_pattern() {
        let pegs = create_rect_peg_pattern(100, 200, 100);

        assert_eq!(pegs.len(), 100);

        // Assert all of the points are within bounds and are along the border of
        // the rectangle.
        for peg in pegs {
            assert!(
                peg.x == 0 || peg.x == 99 || peg.y == 0 || peg.y == 199,
                "{:?} is outside border",
                peg
            );
        }
    }

    #[test]
    fn test_create_oval_peg_pattern() {
        let center_cell = Vector2::new(49.0, 49.0);

        let pegs = create_oval_peg_pattern(100, 100, 100);

        assert_eq!(pegs.len(), 100);

        // Assert all of the points are within bounds and are within a reasonable
        // distance from the center cell.
        for peg in pegs {
            assert_in_range!(peg.x, 0, 100);
            assert_in_range!(peg.y, 0, 200);

            let distance = (Vector2::cast::<f32>(&peg).unwrap() - center_cell).magnitude();
            assert_approx_eq_with_tolerance!(distance, 49.0, 2.0);
        }
    }

    #[test]
    fn test_create_random_peg_pattern() {
        let pegs = create_random_peg_pattern(100, 200, 100);

        assert_eq!(pegs.len(), 100);

        // Assert all of the points are within bounds.
        for peg in pegs {
            assert_in_range!(peg.x, 0, 100);
            assert_in_range!(peg.y, 0, 200);
        }
    }
}
