#[macro_export]
macro_rules! offset_of {
    ($ty:ty, $field:ident) => ({
        &(*(ptr::null() as *const $ty)).$field as *const _ as usize
    })
}

#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!((*a - *b).abs() < 1.0e-6,
                "{:?} is not approximately equal to {:?}", *a, *b);
    })
}

#[macro_export]
macro_rules! assert_approx_eq_with_tolerance {
    ($a:expr, $b:expr, $e:expr) => ({
        let (a, b, e) = (&$a, &$b, &$e);
        assert!((*a - *b).abs() < *e,
                "{:?} is not approximately equal to {:?}, with tolerance {:?}", *a, *b, *e);
    })
}

#[macro_export]
macro_rules! assert_neq {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!(*a != *b, "{:?} is equal to {:?}", *a, *b);
    })
}

#[macro_export]
macro_rules! assert_lt {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!(*a < *b, "{:?} is not less than {:?}", *a, *b);
    })
}

#[macro_export]
macro_rules! assert_lte {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!(*a <= *b, "{:?} is not less than or equal to {:?}", *a, *b);
    })
}

#[macro_export]
macro_rules! assert_gt {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!(*a > *b, "{:?} is not greater than {:?}", *a, *b);
    })
}

#[macro_export]
macro_rules! assert_gte {
    ($a:expr, $b:expr) => ({
        let (a, b) = (&$a, &$b);
        assert!(*a >= *b, "{:?} is not greater than or equal to {:?}", *a, *b);
    })
}

#[macro_export]
macro_rules! assert_in_range {
    ($a:expr, $b:expr, $c:expr) => ({
        let (a, b, c) = (&$a, &$b, &$c);
        assert!(*a >= *b && *a < *c, "{:?} is not in the range [{:?}, {:?})", *a, *b, *c);
    })
}


#[cfg(test)]
mod test {

    #[test]
    fn test_assert_approx_equal() {
        assert_approx_eq!(42_f32, 42_f32);
        assert_approx_eq!(1.0000001_f32, 1.0_f32);
        assert_approx_eq!(1.0_f32, 1.0_f32);
    }

    #[test]
    #[should_panic]
    fn test_assert_approx_equal_not_equal() {
        assert_approx_eq!(1.0_f32, 1.1_f32);
    }

    #[test]
    fn test_assert_neq() {
        assert_neq!(1.0, 2.0);
        assert_neq!(10, 42);
    }

    #[test]
    #[should_panic]
    fn test_assert_neq_equal() {
        assert_neq!(1.0, 1.0);
    }

    #[test]
    fn test_assert_lt() {
        assert_lt!(1.0, 1.1);
        assert_lt!(-42, 42);
    }

    #[test]
    #[should_panic]
    fn test_assert_lt_equal() {
        assert_lt!(1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_assert_lt_greater_than() {
        assert_lt!(2.0, 1.0);
    }

    #[test]
    fn test_assert_lte() {
        assert_lte!(1.0, 1.1);
        assert_lte!(1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_assert_lte_greater_than() {
        assert_lte!(1.1, 1.0);
    }

    #[test]
    fn test_assert_gt() {
        assert_gt!(1.1, 1.0);
        assert_gt!(42, -42);
    }

    #[test]
    #[should_panic]
    fn test_assert_gt_equal() {
        assert_gt!(1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_assert_gt_less_than() {
        assert_gt!(1.0, 2.0);
    }

    #[test]
    fn test_assert_gte() {
        assert_gte!(1.1, 1.0);
        assert_gte!(1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_assert_gte_less_than() {
        assert_gte!(1.0, 1.1);
    }

    #[test]
    fn test_assert_in_range() {
        assert_in_range!(0, -1, 1);
        assert_in_range!(1, 1, 3);
        assert_in_range!(2, 1, 3);
    }

    #[test]
    #[should_panic]
    fn test_assert_in_range_outside_range() {
        assert_in_range!(4, 1, 3);
    }

    #[test]
    #[should_panic]
    fn test_assert_in_range_exclusive_end() {
        assert_in_range!(3, 1, 3);
    }
}
