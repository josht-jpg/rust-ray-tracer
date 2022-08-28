use std::{
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
    process::Output,
};

use crate::{float_equal, Tuple};

// TODO: maybe just make a trait?
#[derive(Debug)]
pub struct Matrix<T>(Vec<Vec<T>>);

pub trait Identity {
    fn identity() -> Self;
}

impl Identity for f64 {
    fn identity() -> Self {
        1.
    }
}

impl Identity for i32 {
    fn identity() -> Self {
        1
    }
}

impl<
        T: Mul<Output = T>
            + Div<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Copy
            + Identity
            + Default
            + Neg<Output = T>
            + PartialEq,
    > Matrix<T>
{
    fn identity(n: usize) -> Self {
        let mut result = vec![vec![T::default(); n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    result[i][j] = T::identity();
                }
            }
        }
        Matrix(result)
    }

    fn row(&self, i: usize) -> &[T] {
        &self.0[i]
    }

    fn col(&self, i: usize) -> Vec<T> {
        let mut result = vec![];
        for (i, r) in self.0.iter().enumerate() {
            result.push(r[i]);
        }

        result
    }

    fn transpose(&self) -> Self {
        let mut result = vec![vec![T::default(); self.0.len()]; self.0[0].len()];

        for i in 0..self.0.len() {
            for j in 0..self.0[i].len() {
                result[j][i] = self.0[i][j];
            }
        }

        Matrix(result)
    }

    fn submatrix(&self, row: usize, col: usize) -> Matrix<T> {
        let mut result = vec![vec![T::default(); self.0.len() - 1]; self.0.len() - 1];
        let mut i = 0;
        let mut j = 0;
        for r in 0..self.0.len() {
            if r != row {
                for c in 0..self.0[r].len() {
                    if c != col {
                        result[i][j] = self.0[r][c];
                        j += 1;
                    }
                }
                i += 1;
                j = 0;
            }
        }
        Matrix(result)
    }

    fn two_by_two_determinant(&self) -> T {
        let a = self.0[0][0];
        let b = self.0[0][1];
        let c = self.0[1][0];
        let d = self.0[1][1];

        (a * d) - (b * c)
    }

    fn minor(&self, row: usize, col: usize) -> T {
        let sub = self.submatrix(row, col);
        sub.determinant()
    }

    fn cofactor(&self, row: usize, col: usize) -> T {
        let minor = self.minor(row, col);
        if (row + col) & 1 == 1 {
            -minor
        } else {
            minor
        }
    }

    fn determinant(&self) -> T {
        if self.0.len() == 2 && self.0[0].len() == 2 {
            self.two_by_two_determinant()
        } else {
            self.0[0]
                .iter()
                .enumerate()
                .fold(T::default(), |acc, (i, a)| acc + *a * self.cofactor(0, i))
        }
    }

    fn is_invertible(&self) -> bool {
        self.determinant() != T::default()
    }

    fn inverse(&self) -> Self {
        // chill out on this one
        if !self.is_invertible() {
            panic!("Matrix is not invertible");
        }

        let determinant = self.determinant();

        let cofactor_matrix = Matrix(
            self.0
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    r.iter()
                        .enumerate()
                        .map(|(j, _)| self.cofactor(i, j))
                        .collect()
                })
                .collect(),
        );

        Matrix(
            cofactor_matrix
                .transpose()
                .0
                .iter()
                .map(|r| r.iter().map(|a| *a / determinant).collect())
                .collect(),
        )
    }
}

// impl<T>  From<Vec<Vmuec<T>>> for Matrix<T> {
//     fn from(m: Vec<Vec<T>>) -> Self {
//        Ma
//     }
// }

impl<T: Copy + Mul<Output = T> + Add + Default + AddAssign> Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let m = self.0.len();
        let n = self.0[0].len();
        let q = rhs.0[0].len();

        let mut result = vec![vec![T::default(); q]; m];
        for i in 0..m {
            for j in 0..q {
                for k in 0..n {
                    result[i][j] += self.0[i][k] * rhs.0[k][j];
                }
            }
        }

        Matrix(result)
    }
}

impl Mul<Tuple> for Matrix<f64> {
    type Output = Tuple;

    fn mul(self, rhs: Tuple) -> Self::Output {
        assert!(self.0[0].len() == 4);
        self.0
            .iter()
            .enumerate()
            .fold(
                Tuple {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                    w: 0.,
                },
                |mut acc, (i, r)| {
                    acc[i as u8] = rhs.x * r[0] + rhs.y * r[1] + rhs.z * r[2] + rhs.w * r[3];
                    acc
                },
            )
            .into()
    }
}

impl PartialEq for Matrix<f64> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(&other.0).all(|(v_i_1, v_i_2)| {
            v_i_1
                .iter()
                .zip(v_i_2)
                .all(|(a_i_1, a_i_2)| float_equal(*a_i_1, *a_i_2))
        })
    }
}

impl PartialEq for Matrix<i32> {
    fn eq(&self, other: &Self) -> bool {
        self.0.iter().zip(&other.0).all(|(v_i_1, v_i_2)| {
            v_i_1
                .iter()
                .zip(v_i_2)
                .all(|(a_i_1, a_i_2)| *a_i_1 == *a_i_2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_equality_works() {
        let A: Matrix<f64> = Matrix(vec![vec![1., 2.5], vec![0., -2.]]);
        let B: Matrix<f64> = Matrix(vec![vec![1., 2.5], vec![0., -2.]]);

        assert_eq!(A, B)
    }

    #[test]
    fn matrix_multiplication_works() {
        let A = Matrix(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 8, 7, 6],
            vec![5, 4, 3, 2],
        ]);
        let B = Matrix(vec![
            vec![-2, 1, 2, 3],
            vec![3, 2, 1, -1],
            vec![4, 3, 6, 5],
            vec![1, 2, 7, 8],
        ]);

        let expected = Matrix(vec![
            vec![20, 22, 50, 48],
            vec![44, 54, 114, 108],
            vec![40, 58, 110, 102],
            vec![16, 26, 46, 42],
        ]);

        assert_eq!(A * B, expected)
    }

    #[test]
    fn mutl_by_tuple_works() {
        let A: Matrix<f64> = Matrix(vec![
            vec![1., 2., 3., 4.],
            vec![2., 4., 4., 2.],
            vec![8., 6., 4., 1.],
            vec![0., 0., 0., 1.],
        ]);

        let b = Tuple {
            x: 1.,
            y: 2.,
            z: 3.,
            w: 1.,
        };

        assert_eq!(
            A * b,
            Tuple {
                x: 18.,
                y: 24.,
                z: 33.,
                w: 1.
            }
        )
    }

    #[test]
    fn identity_works() {
        let A = Matrix(vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]]);

        let I = Matrix::identity(3);

        assert_eq!(
            A * I,
            Matrix(vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]])
        )
    }

    #[test]
    fn identity_works_for_tuples() {
        let a = Tuple {
            x: 1.,
            y: 2.,
            z: 3.,
            w: 4.,
        };

        let I = Matrix::identity(4);

        assert_eq!(
            I * a,
            Tuple {
                x: 1.,
                y: 2.,
                z: 3.,
                w: 4.,
            }
        )
    }

    #[test]
    fn transpose_works() {
        let A = Matrix(vec![
            vec![0, 9, 3, 0],
            vec![9, 8, 0, 8],
            vec![1, 8, 5, 5],
            vec![0, 0, 3, 8],
        ]);
        let B = Matrix(vec![
            vec![0, 9, 1, 0],
            vec![9, 8, 8, 0],
            vec![3, 0, 5, 3],
            vec![0, 8, 5, 8],
        ]);

        assert_eq!(A.transpose(), B)
    }

    #[test]
    fn two_by_two_determinant_works() {
        let A = Matrix(vec![vec![1, -3], vec![5, 2]]);

        assert_eq!(A.two_by_two_determinant(), 17)
    }

    #[test]
    fn submatrix_works() {
        let A = Matrix(vec![
            vec![1, 5, 0, 1],
            vec![-3, 2, 7, -1],
            vec![0, 6, -3, 1],
            vec![5, -1, -7, 1],
        ]);

        let sub = Matrix(vec![vec![1, 0, 1], vec![0, -3, 1], vec![5, -7, 1]]);

        assert_eq!(A.submatrix(1, 1), sub)
    }

    // #[test]
    // fn minor_work() {
    //     let A = Matrix(vec![vec![3, 5, 0], vec![2, -1, -7], vec![6, -1, 5]]);
    //     assert_eq!(A.three_by_three_minor(1, 0), 25)
    // }

    // #[test]
    // fn cofactor_works() {
    //     let A = Matrix(vec![vec![3, 5, 0], vec![2, -1, -7], vec![6, -1, 5]]);
    //     assert_eq!(A.three_by_three_minor(0, 0), -12);
    //     assert_eq!(A.three_by_three_cofactor(0, 0), -12);
    //     assert_eq!(A.three_by_three_minor(1, 0), 25);
    //     assert_eq!(A.three_by_three_cofactor(1, 0), -25)
    // }

    #[test]
    fn determinant_works() {
        let A = Matrix(vec![
            vec![-2, -8, 3, 5],
            vec![-3, 1, 7, 3],
            vec![1, 2, -9, 6],
            vec![-6, 7, 7, -9],
        ]);

        assert_eq!(A.determinant(), -4071)
    }

    #[test]
    fn inverse_works() {
        let A = Matrix(vec![
            vec![-5., 2., 6., -8.],
            vec![1., -5., 1., 8.],
            vec![7., 7., -6., -7.],
            vec![1., -3., 7., 4.],
        ]);
        let B = Matrix(vec![
            vec![0.21805, 0.45113, 0.24060, -0.04511],
            vec![-0.80827, -1.45677, -0.44361, 0.52068],
            vec![-0.07895, -0.22368, -0.05263, 0.19737],
            vec![-0.52256, -0.81391, -0.30075, 0.30639],
        ]);
        assert_eq!(A.inverse(), B);

        let C = Matrix(vec![
            vec![8., -5., 9., 2.],
            vec![7., 5., 6., 1.],
            vec![-6., 0., 9., 6.],
            vec![-3., 0., -9., -4.],
        ]);

        let expected_inverse = Matrix(vec![
            vec![-0.15385, -0.15385, -0.28205, -0.53846],
            vec![-0.07692, 0.12308, 0.02564, 0.03077],
            vec![0.35897, 0.35897, 0.43590, 0.92308],
            vec![-0.69231, -0.69231, -0.76923, -1.92308],
        ]);

        assert_eq!(C.inverse(), expected_inverse);
    }

    fn multiplying_by_inverse_works() {
        let A = Matrix(vec![
            vec![3, -9, 7, 3],
            vec![3, -8, 2, -9],
            vec![-4, 4, 4, 1],
            vec![-6, 5, -1, 1],
        ]);
        let B = Matrix(vec![
            vec![8, 2, 2, 2],
            vec![3, -1, 7, 0],
            vec![7, 0, 5, 4],
            vec![6, -2, 0, 5],
        ]);

        let C = A * B;

        assert_eq!(
            C * Matrix(vec![
                vec![8, 2, 2, 2],
                vec![3, -1, 7, 0],
                vec![7, 0, 5, 4],
                vec![6, -2, 0, 5],
            ])
            .inverse(),
            Matrix(vec![
                vec![3, -9, 7, 3],
                vec![3, -8, 2, -9],
                vec![-4, 4, 4, 1],
                vec![-6, 5, -1, 1],
            ])
        );
    }
}
