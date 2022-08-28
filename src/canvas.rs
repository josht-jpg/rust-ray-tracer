use std::{arch::x86_64::_MM_MASK_INEXACT, fmt::Write};

use crate::colors::Color;
use anyhow::Error;

struct Canvas {
    width: u16,
    height: u16,
    pixels: Vec<Color<f64>>,
}

impl Canvas {
    fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            pixels: vec![
                Color {
                    r: 0.,
                    g: 0.,
                    b: 0.,
                };
                (width * height) as usize
            ],
        }
    }

    fn write_pixel(&mut self, x: u16, y: u16, color: Color<f64>) {
        let index = ((y * self.width) + x) as usize;

        if index < self.pixels.len() {
            self.pixels[index] = color
        }
    }

    fn pixel_at(&self, x: u16, y: u16) -> Option<&Color<f64>> {
        let index = ((y * self.width) + x) as usize;
        self.pixels.get(index)
    }
}

struct PPM {
    identifier: u8,
    height: u16,
    width: u16,
    max_color_value: u8,
    pixels: Vec<Color<u8>>,
}

impl PPM {
    fn header(&self) -> String {
        format!(
            "P{}\n{} {}\n{}",
            self.identifier, self.width, self.height, self.max_color_value
        )
    }

    fn pixel_data(&self) -> String {
        // TODO: lines shouldn't be longer than 70s
        self.pixels
            .iter()
            .enumerate()
            .fold(String::new(), |mut acc, (index, pixel)| {
                write!(&mut acc, "{} {} {}", pixel.r, pixel.g, pixel.b).unwrap();

                if (index + 1) % self.width as usize == 0 {
                    acc.push('\n')
                } else {
                    acc.push(' ')
                }
                acc
            })
    }

    fn raw_string(&self) -> String {
        format!("{}\n{}", self.header(), self.pixel_data())
    }
}

fn clamp<T: PartialOrd>(n: T, min: T, max: T) -> T {
    match n {
        n if n < min => min,
        n if n > max => max,
        _ => n,
    }
}

impl From<Canvas> for PPM {
    fn from(canvas: Canvas) -> Self {
        let default_max_color_value: u8 = 255;

        let pixels = canvas
            .pixels
            .iter()
            .map(|p| Color {
                r: clamp((p.r * 255.).round() as u8, 0, default_max_color_value),
                g: clamp((p.g * 255.).round() as u8, 0, default_max_color_value),
                b: clamp((p.b * 255.).round() as u8, 0, default_max_color_value),
            })
            .collect();

        Self {
            identifier: 3,
            height: canvas.height,
            width: canvas.width,
            max_color_value: 255,
            pixels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_makes_new_canvas() {
        let canvas = Canvas::new(10, 20);

        assert_eq!(canvas.width, 10);
        assert_eq!(canvas.height, 20);

        for pixel in canvas.pixels {
            assert_eq!(
                pixel,
                Color {
                    r: 0.,
                    g: 0.,
                    b: 0.,
                }
            )
        }
    }

    #[test]
    fn it_writes_pixels() {
        let mut c = Canvas::new(10, 20);
        let red = Color {
            r: 1.,
            g: 0.,
            b: 0.,
        };

        c.write_pixel(2, 3, red);

        assert_eq!(c.pixel_at(2, 3), Some(&red))
    }

    #[test]
    fn it_constructs_ppm_header() {
        let c = Canvas::new(5, 3);
        let ppm: PPM = c.into();

        assert_eq!(ppm.header(), "P3\n5 3\n255")
    }

    #[test]
    fn it_constructs_ppm_pixel_data() {
        let mut c = Canvas::new(5, 3);

        let c1 = Color {
            r: 1.5,
            g: 0.,
            b: 0.,
        };

        let c2 = Color {
            r: 0.,
            g: 0.5,
            b: 0.,
        };

        let c3 = Color {
            r: -0.5,
            g: 0.,
            b: 1.,
        };

        c.write_pixel(0, 0, c1);
        c.write_pixel(2, 1, c2);
        c.write_pixel(4, 2, c3);

        let ppm: PPM = c.into();

        assert_eq!(
            ppm.pixel_data(),
            r#"255 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 128 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 255
"#
        )
    }
}
