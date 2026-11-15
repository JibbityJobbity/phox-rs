/// A small set-of-flags newtype, so the RHI needs no extra dependency.
macro_rules! flags {
    ($(#[$meta:meta])* $name:ident { $($(#[$vmeta:meta])* $variant:ident = $value:expr),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug, PartialOrd, Ord)]
        pub struct $name(u32);

        impl $name {
            /// No flags set.
            pub const NONE: Self = Self(0);
            $($(#[$vmeta])* pub const $variant: Self = Self($value);)*

            pub const fn bits(self) -> u32 { self.0 }
            pub const fn from_bits(bits: u32) -> Self { Self(bits) }
            pub const fn is_empty(self) -> bool { self.0 == 0 }
            /// Whether every flag of `other` is set.
            pub const fn contains(self, other: Self) -> bool { self.0 & other.0 == other.0 }
            /// Whether any flag of `other` is set.
            pub const fn intersects(self, other: Self) -> bool { self.0 & other.0 != 0 }
        }
        impl std::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
        }
        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) { self.0 |= rhs.0; }
        }
        impl std::ops::BitAnd for $name {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self { Self(self.0 & rhs.0) }
        }
    };
}
pub(crate) use flags;
