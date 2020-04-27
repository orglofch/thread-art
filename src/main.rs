#[macro_use]
extern crate clap;
extern crate cgmath;
extern crate hex;
extern crate image;
extern crate rand;

#[macro_use]
mod macros;

mod image_utils;
mod peg_factory;
mod raster;
mod render;
mod shader;
mod thread_art;

use std::time::Duration;

use cgmath::Vector3;
use clap::{App, Arg};
use peg_factory::{PegPattern, create_pegs};

use image_utils::*;
use raster::*;
use thread_art::*;

// TODO: Update visibility of all targets.

/// Parses a hexidecimal string into a normalized [0, 1] RGB vector.
fn parse_hex_str(hex: &str) -> Vector3<f32> {
    let hex = hex::decode(hex).unwrap();
    Vector3::new(
        hex[0] as f32 / 255.0,
        hex[1] as f32 / 255.0,
        hex[2] as f32 / 255.0,
    )
}

fn main() {
    // Flag config.
    let matches = App::new("Thread Art Builder")
        .version("1.0")
        .arg(
            Arg::with_name("image")
                .short("i")
                .long("image")
                .value_name("FILE")
                .required(true)
                .help("The target image."),
        )
        .arg(
            Arg::with_name("thread_colours")
                .short("t")
                .long("thread")
                .value_name("FFFFFF")
                .default_value("FFFFFF")
                .multiple(true)
                .help("Valid thread colours."),
        )
        .arg(
            Arg::with_name("thread_count")
                .short("tc")
                .long("thread_count")
                .help("The number of threads.")
        )
        .arg(
            Arg::with_name("background_colour")
                .short("b")
                .long("background")
                .value_name("FFFFFF")
                .default_value("000000")
                .help("The background colour."),
        )
        .arg(
            Arg::with_name("peg_pattern")
                .short("p")
                .long("pattern")
                .possible_values(
                    &[
                        "Uniform",
                        "Rect",
                        "Oval",
                        "ConcentricOval",
                        "Spiral",
                        "Random",
                    ],
                )
                .default_value("Oval")
                .help("The peg pattern to use."),
        )
        .arg(
            Arg::with_name("num_pegs")
                .short("n")
                .long("num_pegs")
                .default_value("100")
                .help("The number of pegs to place."),
        )
        .arg(
            Arg::with_name("checkpoint_file")
                .long("checkpoint_file")
                .default_value("")
                .help("The file to checkpoint the solution to."),
        )
        .arg(
            Arg::with_name("checkpoint_frequency_sec")
                .long("checkpoint_frequency_sec")
                .default_value("10")
                .help("The frequency to checkpoint the solution."),
        )
        .arg(
            Arg::with_name("saliency_map")
                .long("saliency_map")
                .value_name("FILE")
                .help(
                    "A grayscale image of the salient pixels in the target image.",
                ),
        )
        .arg(
            Arg::with_name("fitness_csv_file")
                .long("fitness_csv_file")
                .value_name("FILE")
                .default_value("")
                .help("A CSV file to output fitness samples to."),
        )
        .get_matches();

    // Parse flags.
    let target_img = image::open(matches.value_of("image").unwrap())
        .unwrap()
        .to_rgb();
    // TODO(orglofch): Move conversion to image_utils.
    let mut target_raster =
        Raster::new(
            target_img.width(),
            target_img.height(),
            3,
            vec![0.0; target_img.width() as usize * target_img.height() as usize * 3],
        );
    for y in 0..target_raster.height {
        for x in 0..target_raster.width {
            let pixel = target_img.get_pixel(x as u32, y as u32);
            for c in 0..target_raster.channels {
                target_raster.set(x, y, c, pixel[c as usize] as f32 / 255.0);
            }
        }
    }

    // TODO: Incorprate threads only keeping the furthest ones.
    /*let threads = {
        let mut colours = Vec::new();
        for colour in matches.values_of("thread_colours").unwrap() {
            colours.push(parse_hex_str(colour));
        }
        colours
};*/
    let dominant_colours = image_utils::k_dominant_colours(&target_raster, 10);
    let mut threads = Vec::new();
    for dominant_colour in dominant_colours {
        let mut colour = Vector3::new(0.0, 0.0, 0.0);
        for c in 0..3 {
            colour[c] = dominant_colour[c];
        }
        threads.push(colour);
    }

    let background_colour = parse_hex_str(matches.value_of("background_colour").unwrap());
    let peg_pattern = matches.value_of("peg_pattern").unwrap();
    let num_pegs = matches
        .value_of("num_pegs")
        .unwrap()
        .parse::<u32>()
        .unwrap();
    let checkpoint_file = matches.value_of("checkpoint_file").unwrap().to_string();
    let checkpoint_frequency = Duration::from_secs(
        matches
            .value_of("checkpoint_frequency_sec")
            .unwrap()
            .parse::<u64>()
            .unwrap(),
    );
    let saliency_raster = match matches.value_of("saliency_map") {
        Some(saliency_path) => {
            let saliency_img = image::open(saliency_path).unwrap().to_luma();
            let mut saliency_raster =
                Raster::new(
                    saliency_img.width(),
                    saliency_img.height(),
                    1,
                    vec![0.0; saliency_img.width() as usize * saliency_img.height() as usize],
                );
            for y in 0..target_raster.height {
                for x in 0..target_raster.width {
                    saliency_raster.set(
                        x,
                        y,
                        0,
                        saliency_img.get_pixel(x as u32, y as u32)[0] as f32 / 255.0,
                    );
                }
            }
            Some(saliency_raster)
        }
        None => None,
    };
    let fitness_csv_file = matches.value_of("fitness_csv_file").unwrap().to_string();

    // Generate the pegs to use.
    // TODO: Rename pegs to pegs_factory and try to use enum flags.
    let pegs = create_pegs(
        target_img.width(),
        target_img.height(),
        num_pegs,
        value_t!(matches.value_of("peg_pattern"), PegPattern).unwrap(),
    );

    let config = ThreadArtConfig {
        target_raster: target_raster,
        pegs: pegs,
        threads: threads,
        background_colour: background_colour,
        checkpoint_file: checkpoint_file,
        checkpoint_frequency: checkpoint_frequency,
        saliency_raster: saliency_raster,
        fitness_csv_file: fitness_csv_file,
    };

    let instructions = run_solver(&config);
}
