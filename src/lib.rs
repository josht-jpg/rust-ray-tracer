use std::ops::{Add, Div, Mul, Neg, Sub};

mod colors;

#[derive(Debug)]
pub struct Tuple {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

impl Tuple {
    fn vector(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z, w: 0. }
    }

    fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2) + self.w.powi(2)).sqrt()
    }

    fn normalize(&self) -> Self {
        let magnitude = self.magnitude();
        Self {
            x: self.x / magnitude,
            y: self.y / magnitude,
            z: self.z / magnitude,
            w: self.w / magnitude,
        }
    }

    fn normalize_mut(&mut self) {
        let magnitude = self.magnitude();

        self.x /= magnitude;
        self.y /= magnitude;
        self.z /= magnitude;
        self.w /= magnitude;
    }

    fn dot(a: &Self, b: &Self) -> f64 {
        (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w)
    }

    fn cross(a: &Self, b: &Self) -> Self {
        Self::vector(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }

    fn is_point(&self) -> bool {
        self.w == 1.
    }

    fn is_vector(&self) -> bool {
        self.w == 0.
    }
}

impl PartialEq for Tuple {
    fn eq(&self, b: &Self) -> bool {
        float_equal(self.x, b.x)
            && float_equal(self.y, b.y)
            && float_equal(self.z, b.z)
            && float_equal(self.w, b.w)
    }
}

impl Add for Tuple {
    type Output = Tuple;

    fn add(self, rhs: Self) -> Self::Output {
        Tuple {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl Sub for Tuple {
    type Output = Tuple;

    fn sub(self, rhs: Self) -> Self::Output {
        Tuple {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl Neg for Tuple {
    type Output = Tuple;

    fn neg(self) -> Self::Output {
        Tuple {
            x: 0.,
            y: 0.,
            z: 0.,
            w: 0.,
        } - self
    }
}

impl Mul<f64> for Tuple {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Tuple {
            x: rhs * self.x,
            y: rhs * self.y,
            z: rhs * self.z,
            w: rhs * self.w,
        }
    }
}

impl Mul<Tuple> for f64 {
    type Output = Tuple;

    fn mul(self, rhs: Tuple) -> Self::Output {
        Tuple {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
            w: self * rhs.w,
        }
    }
}

impl Div<f64> for Tuple {
    type Output = Tuple;

    fn div(self, rhs: f64) -> Self::Output {
        Tuple {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

// TODO: delete
fn new_point(x: f64, y: f64, z: f64) -> Tuple {
    Tuple { x, y, z, w: 1. }
}

fn new_vector(x: f64, y: f64, z: f64) -> Tuple {
    Tuple { x, y, z, w: 0. }
}

const EPSILON: f64 = 0.00001;

fn float_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_tuple_woks() {
        let a = Tuple {
            x: 4.3,
            y: -4.2,
            z: 3.1,
            w: 1.0,
        };

        assert_eq!(a.x, 4.3);
        assert_eq!(a.y, -4.2);
        assert_eq!(a.z, 3.1);
        assert_eq!(a.w, 1.0);

        assert!(a.is_point());
        assert!(!a.is_vector());
    }

    #[test]
    fn vector_tuple_woks() {
        let a = Tuple {
            x: 4.3,
            y: -4.2,
            z: 3.1,
            w: 0.0,
        };

        assert_eq!(a.x, 4.3);
        assert_eq!(a.y, -4.2);
        assert_eq!(a.z, 3.1);
        assert_eq!(a.w, 0.);

        assert!(!a.is_point());
        assert!(a.is_vector());
    }

    #[test]
    fn adding_tuples_works() {
        let a1 = Tuple {
            x: 3.,
            y: -2.,
            z: 5.,
            w: 1.,
        };
        let a2 = Tuple {
            x: -2.,
            y: 3.,
            z: 1.,
            w: 0.,
        };

        assert_eq!(
            a1 + a2,
            Tuple {
                x: 1.,
                y: 1.,
                z: 6.,
                w: 1.
            }
        )
    }

    #[test]
    fn substracting_tuples_works() {
        let p1 = new_point(3., 2., 1.);
        let p2 = new_point(5., 6., 7.);

        assert_eq!(p1 - p2, new_vector(-2., -4., -6.))
    }

    #[test]
    fn negative_tuple_works() {
        let a = Tuple {
            x: 1.,
            y: -2.,
            z: 3.,
            w: -4.,
        };
        assert_eq!(
            -a,
            Tuple {
                x: -1.,
                y: 2.,
                z: -3.,
                w: 4.
            }
        )
    }

    #[test]
    fn mutliplying_tuple_by_scalar_works() {
        let a = Tuple {
            x: 1.,
            y: -2.,
            z: 3.,
            w: -4.,
        };

        assert_eq!(
            3.5 * a,
            Tuple {
                x: 3.5,
                y: -7.,
                z: 10.5,
                w: -14.
            }
        )
    }

    #[test]
    fn magnitude_works() {
        let v1 = new_vector(1., 0., 0.);
        assert_eq!(v1.magnitude(), 1.);

        let v2 = new_vector(0., 1., 0.);
        assert_eq!(v2.magnitude(), 1.);

        let v3 = new_vector(0., 0., 1.);
        assert_eq!(v3.magnitude(), 1.);

        let v4 = new_vector(-1., -2., -3.);
        assert_eq!(v4.magnitude(), 14_f64.sqrt())
    }

    #[test]
    fn noramlizing_works() {
        let v1 = new_vector(4., 0., 0.);
        assert_eq!(v1.normalize(), new_vector(1., 0., 0.));

        let v2 = new_vector(1., 2., 3.);
        assert_eq!(v2.normalize(), new_vector(0.26726, 0.53452, 0.80178));

        assert_eq!(v2.normalize().magnitude(), 1.)
    }

    #[test]
    fn dot_works() {
        let a = new_vector(1., 2., 3.);
        let b = new_vector(2., 3., 4.);
        assert_eq!(Tuple::dot(&a, &b), 20.)
    }
}
