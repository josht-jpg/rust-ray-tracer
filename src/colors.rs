use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::float_equal;

// TODO: change to f32
#[derive(Debug, Clone, Copy)]
pub struct Color<T> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl<T> Add for Color<T>
where
    // Need all that yo?
    T: Add<Output = T> + Sub + Mul + Div + Neg + PartialEq,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Color {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl<T: Sub<Output = T>> Sub for Color<T> {
    type Output = Color<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Color {
            r: self.r - rhs.r,
            g: self.g - rhs.g,
            b: self.b - rhs.b,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Color<T> {
    type Output = Color<T>;

    fn neg(self) -> Self::Output {
        Color {
            r: -self.r,
            g: -self.g,
            b: -self.b,
        }
    }
}

impl<T: Mul<Output = T>> Mul for Color<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            r: self.r * rhs.r,
            g: self.g * rhs.g,
            b: self.b * rhs.b,
        }
    }
}

// impl Mul<f64> for Color {
//     type Output = Self;

//     fn mul(self, rhs: f64) -> Self::Output {
//         Color {
//             r: rhs * self.r,
//             g: rhs * self.g,
//             b: rhs * self.b,
//         }
//     }
// }

// impl<T> Mul<Color<T>> for G {
//     type Output = Color<T>;

//     fn mul(self, rhs: Color<T>) -> Self::Output {
//         Color {
//             r: self * rhs.r,
//             g: self * rhs.g,
//             b: self * rhs.b,
//         }
//     }
// }

impl PartialEq for Color<f64> {
    fn eq(&self, b: &Self) -> bool {
        float_equal(self.r, b.r) && float_equal(self.g, b.g) && float_equal(self.b, b.b)
    }
}

impl PartialEq for Color<u8> {
    fn eq(&self, b: &Self) -> bool {
        self.r == b.r && self.g == b.g && self.b == b.b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adding_works() {
        let c1 = Color {
            r: 0.9,
            g: 0.6,
            b: 0.75,
        };
        let c2 = Color {
            r: 0.7,
            g: 0.1,
            b: 0.25,
        };

        assert_eq!(
            c1 + c2,
            Color {
                r: 1.6,
                g: 0.7,
                b: 1.0
            }
        )
    }

    #[test]
    fn multiplying_works() {
        let c1 = Color {
            r: 1.,
            g: 0.2,
            b: 0.4,
        };
        let c2 = Color {
            r: 0.9,
            g: 1.,
            b: 0.1,
        };

        assert_eq!(
            c1 * c2,
            Color {
                r: 0.9,
                g: 0.2,
                b: 0.04
            }
        )
    }
}
