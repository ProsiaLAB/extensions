pub mod io {
    use std::fs::File;
    use std::io::BufRead;
    use std::io::BufReader;
    use std::io::Error;
    use std::path::Path;

    pub fn load_file_as_str<P: AsRef<Path>>(path: P) -> Result<Vec<String>, Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let all_lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

        Ok(all_lines)
    }
}

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

    /// Returns the index of the largest element in a sorted slice that is
    /// less than or equal to the given value.
    ///
    /// # Arguments
    /// * `val` - The target value to compare against.
    /// * `array` - A slice of `f64` values. Must be sorted in non-decreasing order.
    ///
    /// # Returns
    /// * `Some(index)` if an element exists in `array` such that:
    ///   - `array[index] <= val`
    ///   - and `array[index + 1] > val` (or `index` is the last valid element).
    /// * `None` if:
    ///   - the slice is empty
    ///   - `val` is smaller than the first element
    ///   - `val` is greater than or equal to the last element
    ///
    /// # Complexity
    /// Runs in O(log n) time using binary search via `partition_point`.
    ///
    /// # Example
    /// ```
    /// let arr = [1.0, 2.5, 4.0, 7.0];
    /// assert_eq!(find_index_le(3.0, &arr), Some(1)); // arr[1] = 2.5
    /// assert_eq!(find_index_le(1.0, &arr), Some(0));
    /// assert_eq!(find_index_le(7.0, &arr), None);
    /// assert_eq!(find_index_le(0.5, &arr), None);
    /// ```
    pub fn find_index_le(val: f64, array: &[f64]) -> Option<usize> {
        if array.is_empty() || val < array[0] || val >= array[array.len() - 1] {
            return None;
        }
        let idx = array.partition_point(|&x| x <= val);
        if idx > 0 { Some(idx - 1) } else { None }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use ndarray::{Array1, Array2, array};

        #[test]
        fn test_maxval_minval_nonempty() {
            let a = array![1.0, 3.5, 2.2, -5.1, 7.3];
            assert_eq!(a.maxval(), Ok(7.3));
            assert_eq!(a.minval(), Ok(-5.1));
        }

        #[test]
        fn test_maxval_minval_empty() {
            let a: Array1<f64> = Array1::from_vec(vec![]);
            assert_eq!(a.maxval(), Err(ExtremaError::EmptyArray));
            assert_eq!(a.minval(), Err(ExtremaError::EmptyArray));
        }

        #[test]
        fn test_maxval_along_axis0() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.maxval_along(Axis(0)).unwrap();
            assert_eq!(result, array![3.3, 4.2, 7.7]);
        }

        #[test]
        fn test_minval_along_axis0() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.minval_along(Axis(0)).unwrap();
            assert_eq!(result, array![1.0, -1.5, 2.1]);
        }

        #[test]
        fn test_maxval_along_axis1() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.maxval_along(Axis(1)).unwrap();
            assert_eq!(result, array![4.2, 7.7]);
        }

        #[test]
        fn test_minval_along_axis1() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.minval_along(Axis(1)).unwrap();
            assert_eq!(result, array![1.0, -1.5]);
        }

        #[test]
        fn test_argmax_argmin() {
            let a = array![10.0, 3.1, 50.5, -2.2, 50.5];
            assert_eq!(a.argmax(), Ok(2)); // first 50.5
            assert_eq!(a.argmin(), Ok(3));
        }

        #[test]
        fn test_argmax_along_axis0() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.argmax_along(Axis(0)).unwrap();
            assert_eq!(result, array![1, 0, 1]);
        }

        #[test]
        fn test_argmin_along_axis0() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.argmin_along(Axis(0)).unwrap();
            assert_eq!(result, array![0, 1, 0]);
        }

        #[test]
        fn test_argmax_along_axis1() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.argmax_along(Axis(1)).unwrap();
            assert_eq!(result, array![1, 2]);
        }

        #[test]
        fn test_argmin_along_axis1() {
            let a = array![[1.0, 4.2, 2.1], [3.3, -1.5, 7.7]];
            let result = a.argmin_along(Axis(1)).unwrap();
            assert_eq!(result, array![0, 1]);
        }

        #[test]
        fn test_maxval_minval_with_nan() {
            let a = array![1.0, f64::NAN, 3.5];
            // Now should return UndefinedOrder error when NaN is present
            assert_eq!(a.maxval(), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.minval(), Err(ExtremaError::UndefinedOrder));
        }

        #[test]
        fn test_maxval_along_axis_with_nan() {
            let a = array![[1.0, f64::NAN, 2.0], [3.0, 4.0, 5.0]];

            // Along axis 0: column 1 has a NaN, so should return UndefinedOrder error
            assert_eq!(a.maxval_along(Axis(0)), Err(ExtremaError::UndefinedOrder));

            // Along axis 1: first row has a NaN, so should return UndefinedOrder error
            assert_eq!(a.maxval_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
        }

        #[test]
        fn test_minval_along_axis_with_nan() {
            let a = array![[1.0, 4.0, 2.0], [f64::NAN, -1.5, 7.0]];

            // Both axes should return UndefinedOrder error due to NaN presence
            assert_eq!(a.minval_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.minval_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
        }

        #[test]
        fn test_argmax_argmin_with_nan() {
            let a = array![1.0, f64::NAN, 3.5];
            // Should return UndefinedOrder error when NaN is present
            assert_eq!(a.argmax(), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.argmin(), Err(ExtremaError::UndefinedOrder));
        }

        #[test]
        fn test_argmax_argmin_along_with_nan() {
            let a = array![[1.0, f64::NAN, 2.0], [3.0, 4.0, 5.0]];

            // Should return UndefinedOrder error due to NaN presence
            assert_eq!(a.argmax_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.argmin_along(Axis(0)), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.argmax_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.argmin_along(Axis(1)), Err(ExtremaError::UndefinedOrder));
        }

        #[test]
        fn test_all_methods_empty_2d() {
            let a: Array2<f64> = Array2::from_shape_vec((0, 3), vec![]).unwrap();
            assert_eq!(a.maxval(), Err(ExtremaError::EmptyArray));
            assert_eq!(a.minval(), Err(ExtremaError::EmptyArray));
            assert_eq!(a.maxval_along(Axis(0)), Err(ExtremaError::EmptyArray));
            assert_eq!(a.minval_along(Axis(1)), Err(ExtremaError::EmptyArray));
            assert_eq!(a.argmax(), Err(ExtremaError::EmptyArray));
            assert_eq!(a.argmin(), Err(ExtremaError::EmptyArray));
            assert_eq!(a.argmax_along(Axis(0)), Err(ExtremaError::EmptyArray));
            assert_eq!(a.argmin_along(Axis(1)), Err(ExtremaError::EmptyArray));
        }

        #[test]
        fn test_valid_arrays_without_nan() {
            // Test that normal arrays (without NaN) work correctly
            let a = array![1, 5, 3, 2, 4];
            assert_eq!(a.maxval(), Ok(5));
            assert_eq!(a.minval(), Ok(1));
            assert_eq!(a.argmax(), Ok(1));
            assert_eq!(a.argmin(), Ok(0));

            let b = array![[1, 2, 3], [4, 5, 6]];
            assert_eq!(b.maxval_along(Axis(0)).unwrap(), array![4, 5, 6]);
            assert_eq!(b.minval_along(Axis(0)).unwrap(), array![1, 2, 3]);
            assert_eq!(b.argmax_along(Axis(1)).unwrap(), array![2, 2]);
            assert_eq!(b.argmin_along(Axis(1)).unwrap(), array![0, 0]);
        }

        #[test]
        fn test_single_element_arrays() {
            let a = array![42.0];
            assert_eq!(a.maxval(), Ok(42.0));
            assert_eq!(a.minval(), Ok(42.0));
            assert_eq!(a.argmax(), Ok(0));
            assert_eq!(a.argmin(), Ok(0));

            let b = array![[5.0]];
            assert_eq!(b.maxval_along(Axis(0)).unwrap(), array![5.0]);
            assert_eq!(b.minval_along(Axis(1)).unwrap(), array![5.0]);
            assert_eq!(b.argmax_along(Axis(0)).unwrap(), array![0]);
            assert_eq!(b.argmin_along(Axis(1)).unwrap(), array![0]);
        }

        #[test]
        fn test_single_nan_element() {
            let a = array![f64::NAN];
            assert_eq!(a.maxval(), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.minval(), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.argmax(), Err(ExtremaError::UndefinedOrder));
            assert_eq!(a.argmin(), Err(ExtremaError::UndefinedOrder));
        }
    }
}

pub mod types {
    use ndarray::{Array1, Array2, Array3, Array4, ArrayView1};

    /// Generic Vector (1D array)
    pub type Vector<T> = Array1<T>;

    /// n-dimensional real vector (1D array).
    pub type RVector = Array1<f64>;

    /// n-dimensional real vector view (1D view).
    pub type RVecView<'a> = ArrayView1<'a, f64>;

    /// Generic matrix (2D array)
    pub type Matrix<T> = Array2<T>;

    /// A real matrix (2D ndarray).
    pub type RMatrix = Array2<f64>;

    /// n-dimensional real matrix view (2D view)
    pub type RMatView<'a> = ndarray::ArrayView2<'a, f64>;

    /// Generic tensor (3D array)
    pub type Tensor<T> = Array3<T>;

    /// A real tensor (3D ndarray).
    pub type RTensor = Array3<f64>;

    /// A 4-dimensional real tensor (4D ndarray).
    pub type RTensor4 = Array4<f64>;

    /// 1-dimensional unsigned-integer vector.
    pub type UVector = Array1<usize>;

    /// 1-dimensional signed-integer vector.
    pub type IVector = Array1<isize>;

    /// 1-dimensional boolean vector.
    pub type BVector = Array1<bool>;

    #[cfg(feature = "complex")]
    use num_complex::Complex;

    #[cfg(feature = "complex")]
    /// A fixed-length array of complex ([f64]) numbers.
    pub type CVector = Array1<Complex<f64>>;

    #[cfg(feature = "complex")]
    /// A 2-dimensional array (matrix) of complex ([f64]) numbers.
    pub type CMatrix = Array2<Complex<f64>>;

    #[cfg(feature = "complex")]
    /// A 3-dimensional array (tensor) of complex ([f64]) numbers.
    pub type CTensor = Array3<Complex<f64>>;
}
