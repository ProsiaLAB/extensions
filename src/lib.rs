pub mod arrays {
    use std::fmt;

    use ndarray::s;
    use ndarray::{ArrayBase, Axis, OwnedRepr};
    use ndarray::{Data, Dimension, RemoveAxis};

    use crate::types::RVector;

    /// Error type for extrema operations
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ExtremaError {
        EmptyArray,
        UndefinedOrder, // e.g., NaN encountered
    }

    impl fmt::Display for ExtremaError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                ExtremaError::EmptyArray => write!(f, "cannot compute extrema of empty array"),
                ExtremaError::UndefinedOrder => {
                    write!(f, "undefined order: encountered NaN or incomparable values")
                }
            }
        }
    }

    impl std::error::Error for ExtremaError {}

    /// Extension trait providing convenience methods for computing extrema
    /// (minima, maxima) and their indices on [ndarray::ArrayBase] values.
    pub trait ArrayExtrema<T, D>
    where
        D: Dimension,
    {
        /// Returns the maximum value in the array.
        ///
        /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
        /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
        ///
        /// # Examples
        ///
        /// ```
        /// use ndarray::array;
        /// use scorpia::utils::arrays::ArrayExtrema;
        /// use scorpia::utils::arrays::ExtremaError;
        ///
        /// let a = array![1, 3, 2];
        /// assert_eq!(a.maxval(), Ok(3));
        ///
        /// let empty: ndarray::Array1<i32> = ndarray::Array1::from_vec(vec![]);
        /// assert_eq!(empty.maxval(), Err(ExtremaError::EmptyArray));
        /// ```
        fn maxval(&self) -> Result<T, ExtremaError>;

        /// Returns the minimum value in the array.
        ///
        /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
        /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
        fn minval(&self) -> Result<T, ExtremaError>;

        /// Returns an array of maximum values along the given axis.
        ///
        /// Each element of the returned array is the maximum of the slice taken
        /// along axis. Returns `Err(ExtremaError::UndefinedOrder)` if NaN values are encountered,
        /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
        ///
        /// # Panics
        /// Panics if any subview is empty, though this cannot occur if self
        /// itself is non-empty.
        fn maxval_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError>;

        /// Returns an array of minimum values along the given axis.
        ///
        /// Each element of the returned array is the minimum of the slice taken
        /// along axis. Returns `Err(ExtremaError::UndefinedOrder)` if NaN values are encountered,
        /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
        fn minval_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError>;

        /// Returns the index of the maximum element in the array.
        ///
        /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
        /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
        ///
        /// The index is returned in [ndarray::Dimension::Pattern] form, which matches the array's dimensionality.
        fn argmax(&self) -> Result<D::Pattern, ExtremaError>;

        /// Returns the index of the minimum element in the array.
        ///
        /// Returns `Err(ExtremaError::UndefinedOrder)` if any NaN values are encountered,
        /// or `Err(ExtremaError::EmptyArray)` if the array is empty.
        fn argmin(&self) -> Result<D::Pattern, ExtremaError>;

        /// Returns an array of indices of the maximum elements along the given axis.
        ///
        /// Each element in the returned array is the index (within the axis) of the
        /// maximum value of the corresponding subview. Returns `Err(ExtremaError::UndefinedOrder)`
        /// if NaN values are encountered, or `Err(ExtremaError::EmptyArray)` if the array is empty.
        fn argmax_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError>;

        /// Returns an array of indices of the minimum elements along the given axis.
        ///
        /// Each element in the returned array is the index (within the axis) of the
        /// minimum value of the corresponding subview. Returns `Err(ExtremaError::UndefinedOrder)`
        /// if NaN values are encountered, or `Err(ExtremaError::EmptyArray)` if the array is empty.
        fn argmin_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError>;
    }

    impl<T, S, D> ArrayExtrema<T, D> for ArrayBase<S, D>
    where
        T: PartialOrd + Copy,
        S: Data<Elem = T>,
        D: Dimension + RemoveAxis,
    {
        /// See [ArrayExtrema::maxval].
        fn maxval(&self) -> Result<T, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut max_val = None;
            for &val in self.iter() {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match max_val {
                    None => max_val = Some(val),
                    Some(current_max) => {
                        match val.partial_cmp(&current_max) {
                            Some(std::cmp::Ordering::Greater) => max_val = Some(val),
                            Some(_) => {} // val <= current_max, keep current_max
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }
            Ok(max_val.unwrap()) // Safe because we checked for empty array above
        }

        /// See [ArrayExtrema::minval].
        fn minval(&self) -> Result<T, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut min_val = None;
            for &val in self.iter() {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match min_val {
                    None => min_val = Some(val),
                    Some(current_min) => {
                        match val.partial_cmp(&current_min) {
                            Some(std::cmp::Ordering::Less) => min_val = Some(val),
                            Some(_) => {} // val >= current_min, keep current_min
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }
            Ok(min_val.unwrap()) // Safe because we checked for empty array above
        }

        /// See [ArrayExtrema::maxval_along].
        fn maxval_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<T, ExtremaError> {
                let mut max_val = None;
                for &val in subview.iter() {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match max_val {
                        None => max_val = Some(val),
                        Some(current_max) => match val.partial_cmp(&current_max) {
                            Some(std::cmp::Ordering::Greater) => max_val = Some(val),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }
                Ok(max_val.unwrap()) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in result.iter_mut() {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<T, ExtremaError> elements to T
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
        }

        /// See [ArrayExtrema::minval_along].
        fn minval_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<T>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<T, ExtremaError> {
                let mut min_val = None;
                for &val in subview.iter() {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match min_val {
                        None => min_val = Some(val),
                        Some(current_min) => match val.partial_cmp(&current_min) {
                            Some(std::cmp::Ordering::Less) => min_val = Some(val),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }
                Ok(min_val.unwrap()) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in result.iter_mut() {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<T, ExtremaError> elements to T
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
        }

        /// See [ArrayExtrema::argmax].
        fn argmax(&self) -> Result<D::Pattern, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut best = None;

            for (idx, &val) in self.indexed_iter() {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match best {
                    None => best = Some((idx, val)),
                    Some((_, best_val)) => {
                        match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Greater) => best = Some((idx, val)),
                            Some(_) => {} // val <= best_val, keep current best
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }

            Ok(best.unwrap().0) // Safe because we checked for empty array above
        }

        /// See [ArrayExtrema::argmin].
        fn argmin(&self) -> Result<D::Pattern, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut best = None;

            for (idx, &val) in self.indexed_iter() {
                // Check for NaN or incomparable values by comparing with itself
                if val.partial_cmp(&val).is_none() {
                    return Err(ExtremaError::UndefinedOrder);
                }

                match best {
                    None => best = Some((idx, val)),
                    Some((_, best_val)) => {
                        match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Less) => best = Some((idx, val)),
                            Some(_) => {} // val >= best_val, keep current best
                            None => return Err(ExtremaError::UndefinedOrder), // NaN or incomparable values
                        }
                    }
                }
            }

            Ok(best.unwrap().0) // Safe because we checked for empty array above
        }

        /// See [ArrayExtrema::argmax_along].
        fn argmax_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<usize, ExtremaError> {
                let mut best = None;

                for (idx, &val) in subview.indexed_iter() {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match best {
                        None => best = Some((idx, val)),
                        Some((_, best_val)) => match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Greater) => best = Some((idx, val)),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }

                Ok(best.unwrap().0) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in result.iter_mut() {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<usize, ExtremaError> elements to usize
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
        }

        /// See [ArrayExtrema::argmin_along].
        fn argmin_along(
            &self,
            axis: Axis,
        ) -> Result<ArrayBase<OwnedRepr<usize>, D::Smaller>, ExtremaError> {
            if self.is_empty() {
                return Err(ExtremaError::EmptyArray);
            }

            let mut result = self.map_axis(axis, |subview| -> Result<usize, ExtremaError> {
                let mut best = None;

                for (idx, &val) in subview.indexed_iter() {
                    // Check for NaN or incomparable values by comparing with itself
                    if val.partial_cmp(&val).is_none() {
                        return Err(ExtremaError::UndefinedOrder);
                    }

                    match best {
                        None => best = Some((idx, val)),
                        Some((_, best_val)) => match val.partial_cmp(&best_val) {
                            Some(std::cmp::Ordering::Less) => best = Some((idx, val)),
                            Some(_) => {}
                            None => return Err(ExtremaError::UndefinedOrder),
                        },
                    }
                }

                Ok(best.unwrap().0) // Safe because subview is guaranteed non-empty
            });

            // Check if any subview computation failed
            for elem in result.iter_mut() {
                if let Err(err) = elem {
                    return Err(*err);
                }
            }

            // Convert Result<usize, ExtremaError> elements to usize
            let final_result = result.map(|res| res.unwrap());
            Ok(final_result)
        }
    }

    pub trait Integrable {
        fn trapezoid(&self, x: &RVector) -> f64;
    }

    impl Integrable for RVector {
        fn trapezoid(&self, x: &RVector) -> f64 {
            assert_eq!(self.len(), x.len(), "Arrays must have the same length");

            let y0 = self.slice(s![..-1]);
            let y1 = self.slice(s![1..]);
            let x0 = x.slice(s![..-1]);
            let x1 = x.slice(s![1..]);

            ((&y0 + &y1) / 2.0 * (&x1 - &x0)).sum()
        }
    }
}

pub mod types {
    use ndarray::{Array1, Array2, Array3, Array4, ArrayView1};

    /// n-dimensional real vector (1D array).
    pub type RVector = Array1<f64>;

    /// n-dimensional real vector view (1D view).
    pub type RVecView<'a> = ArrayView1<'a, f64>;

    /// A real matrix (2-dimensional ndarray).
    pub type RMatrix = Array2<f64>;

    /// n-dimenstional real matrix view (1d view)
    pub type RMatView<'a> = ndarray::ArrayView2<'a, f64>;

    /// A real tensor (3-dimensional ndarray).
    pub type RTensor = Array3<f64>;

    /// A 4-dimensional real tensor (4-dimensional ndarray).
    pub type RTensor4 = Array4<f64>;

    /// 1-dimensional unsigned-integer vector.
    pub type UVector = Array1<usize>;

    /// 1-dimensional signed-integer vector.
    pub type IVector = Array1<isize>;

    /// 1-dimensional boolean vector.
    pub type BVector = Array1<bool>;
}
