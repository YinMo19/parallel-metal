use std::ops::Index;

use crate::{Error, Result};

/// The logical size of a rank-`D` tensor.
///
/// Axis 0 is contiguous. Spatial extents therefore use
/// `[width, height, depth, ...]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Extent<const D: usize> {
    axes: [usize; D],
}

impl<const D: usize> Extent<D> {
    pub const fn new(axes: [usize; D]) -> Self {
        Self { axes }
    }

    pub const fn axes(self) -> [usize; D] {
        self.axes
    }

    pub fn element_count(self) -> Result<usize> {
        if D == 0 || self.axes.contains(&0) {
            return Err(Error::EmptyExtent);
        }

        self.axes
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .ok_or(Error::ExtentOverflow)
    }

    pub fn point_from_linear(self, linear: usize) -> Point<D> {
        let mut remaining = linear;
        let mut axes = [0; D];
        for (axis, coordinate) in axes.iter_mut().enumerate() {
            *coordinate = remaining % self.axes[axis];
            remaining /= self.axes[axis];
        }
        Point::new(axes)
    }
}

impl<const D: usize> Index<usize> for Extent<D> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.axes[index]
    }
}

impl<const D: usize> From<[usize; D]> for Extent<D> {
    fn from(axes: [usize; D]) -> Self {
        Self::new(axes)
    }
}

/// A logical coordinate in a rank-`D` tensor.
///
/// Spatial points use `[x, y, z, ...]`, matching [`Extent`]'s axis order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point<const D: usize> {
    axes: [usize; D],
}

impl<const D: usize> Point<D> {
    pub const fn new(axes: [usize; D]) -> Self {
        Self { axes }
    }

    pub const fn axes(self) -> [usize; D] {
        self.axes
    }
}

impl<const D: usize> Index<usize> for Point<D> {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.axes[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_axis_contiguous_point_from_linear() {
        let extent = Extent::new([2, 3, 4]);
        assert_eq!(extent.element_count().unwrap(), 24);
        assert_eq!(extent.point_from_linear(0), Point::new([0, 0, 0]));
        assert_eq!(extent.point_from_linear(6), Point::new([0, 0, 1]));
        assert_eq!(extent.point_from_linear(14), Point::new([0, 1, 2]));
        assert_eq!(extent.point_from_linear(23), Point::new([1, 2, 3]));
    }

    #[test]
    fn rejects_empty_and_overflowing_extents() {
        assert!(matches!(
            Extent::new([2, 0]).element_count(),
            Err(Error::EmptyExtent)
        ));
        assert!(matches!(
            Extent::new([usize::MAX, 2]).element_count(),
            Err(Error::ExtentOverflow)
        ));
    }
}
