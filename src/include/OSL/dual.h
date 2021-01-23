// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// clang-format off

#pragma once

#include <initializer_list>
#include <utility>
#include <OSL/oslconfig.h>
#include <OSL/oslversion.h>
#include <OpenImageIO/fmath.h>


OSL_NAMESPACE_ENTER


// Shortcut notation for enable_if trickery
# define DUAL_REQUIRES(...) \
  typename std::enable_if<(__VA_ARGS__), bool>::type = true



// To generically access elements or partials of a Dual, use a ConstIndex<#>.
// NOTE:  Each ConstIndex<#> (ConstIndex<0>, ConstIndex<1>, ConstIndex<2>, ...)
// is a different type and known at compile time.  A regular loop index will
// not work to access elements or partials.  For non-generic code, the methods
// .val(), .dx(), .dy(), .dz() should be used.
// For generic code, the macro "OSL_INDEX_LOOP(__INDEX_NAME, __COUNT, ...)"
// is provided to execute a code block (...) with __INDEX_NAME correctly typed
// for each ConstIndex<#> within the # range [0-__COUNT).
// In practice, this means changing
//     for(int i=0; i < elements;++i) {
//          elem(i) += value;
//     }
// to
//     OSL_INDEX_LOOP(i, elements, {
//          elem(i) += value;
//     });


// If possible (C++14+) use generic lambda based to loop to repeat code
// sequences with compile time known ConstIndex.  Otherwise for c++11,
// a more limited manually unrolled loop to a fixed maximum __COUNT
// works because dead code elimination prunes extraneous iterations.
#if (OSL_CPLUSPLUS_VERSION >= 14) && !defined(__GNUC__)
    // explanation of passing code block as macro argument to handle
    // nested comma operators that might break up the code block into
    // multiple macro arguments
    // https://mort.coffee/home/obscure-c-features/
    //
    // The macro's variable argument list populates the body of the functor lambda.
    // NOTE: generic lambda parameter __INDEX_NAME means that the functor itself
    // is a template that accepts any type.  The static_foreach will call the
    // templated functor with different types of ConstIndex<#> where # is [0-__COUNT)
    #define OSL_INDEX_LOOP(__INDEX_NAME, __COUNT, ...) \
        { \
            auto functor = [&](auto __INDEX_NAME) { \
                __VA_ARGS__; \
            }; \
            static_foreach<ConstIndex, __COUNT>(functor); \
        }

#else
    namespace pvt {
        template<typename T>
        static OSL_HOSTDEVICE constexpr T static_min(T a, T b) {
            return (b < a) ? b : a;
        }
    }
    // explanation of passing code block as macro argument to handle
    // nested comma operators that might break up the code block into
    // multiple macro arguments
    // https://mort.coffee/home/obscure-c-features/
    //
    // We rely on dead code elimination to quickly get rid of the code emitted
    // for out of range indices, but we do take care to not generate any ConstIndex<#>
    // that would be out of range using the static_min helper
    #define OSL_INDEX_LOOP(__INDEX_NAME, __COUNT, ...) \
        static_assert((__COUNT) < 4, "macro based expansion must be repeated to support higher __COUNT values"); \
        { ConstIndex<0> __INDEX_NAME; __VA_ARGS__ ; } \
        if ((__COUNT) > 1) { ConstIndex<pvt::static_min(1, (__COUNT)-1)> __INDEX_NAME; __VA_ARGS__ ; } \
        if ((__COUNT) > 2) { ConstIndex<pvt::static_min(2, (__COUNT)-1)> __INDEX_NAME; __VA_ARGS__ ; } \
        if ((__COUNT) > 3) { ConstIndex<pvt::static_min(3, (__COUNT)-1)> __INDEX_NAME; __VA_ARGS__ ; }
#endif

/// Dual numbers are used to represent values and their derivatives.
///
/// The method is summarized in:
///      Piponi, Dan, "Automatic Differentiation, C++ Templates,
///      and Photogrammetry," Journal of Graphics, GPU, and Game
///      Tools (JGT), Vol 9, No 4, pp. 41-55 (2004).
///
///

// Default DualStorage can handle any number of partials by storing elements
// in an array.  In practice the array subscript operation can require
// standard data layout so that the base address + index works.  This
// can inhibit other transformations useful for vectorization such as
// storing data members in a Structure Of Arrays (SOA) layout
template<class T, int PARTIALS>
class DualStorage
{
public:
    T m_elem[PARTIALS+1];

    template<int N>
    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<N>) const { return m_elem[N]; }
    template<int N>
    OSL_HOSTDEVICE T& elem (ConstIndex<N>)  { return m_elem[N]; }
};


// Specialize DualStorage for specific number of partials and store elements
// as individual data members.  Because each ConstIndex<#> is its own type and
// identifies a specific element, we can overload the elem function for each
// specific ConstIndex<#> and return the corresponding data member.
// The main benefit is better enabling of SROA (Scalar Replacement of Aggregates)
// and other transformations that allow each data member to be
// optimized independently, kept in vector registers vs. scatter back to the
// stack, and more.
template<class T>
class DualStorage<T, 1>
{
public:
    T m_val;
    T m_dx;

    // To better enable Scalar Replacement of Aggregates and other
    // transformations, CLANG has easier time if the per element
    // constructors declared.
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage() {}
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage(const T & val, const T & dx)
    : m_val(val)
    , m_dx(dx)
    {}
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage(const DualStorage &other)
    : m_val(other.m_val)
    , m_dx(other.m_dx)
    {}

    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<0>) const { return m_val; }
    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<1>) const { return m_dx; }
    OSL_HOSTDEVICE T& elem (ConstIndex<0>)  { return m_val; }
    OSL_HOSTDEVICE T& elem (ConstIndex<1>)  { return m_dx; }
};


// Specialize layout to be explicit data members vs. array
template<class T>
class DualStorage<T, 2>
{
public:
    T m_val;
    T m_dx;
    T m_dy;

    // To better enable Scalar Replacement of Aggregates and other
    // transformations, CLANG has easier time if the per element
    // constructors declared.
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage() {}
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage(const T & val, const T & dx, const T & dy)
    : m_val(val)
    , m_dx(dx)
    , m_dy(dy)
    {}
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage(const DualStorage &other)
    : m_val(other.m_val)
    , m_dx(other.m_dx)
    , m_dy(other.m_dy)
    {}

    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<0>) const { return m_val; }
    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<1>) const { return m_dx; }
    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<2>) const { return m_dy; }
    OSL_HOSTDEVICE T& elem (ConstIndex<0>)  { return m_val; }
    OSL_HOSTDEVICE T& elem (ConstIndex<1>)  { return m_dx; }
    OSL_HOSTDEVICE T& elem (ConstIndex<2>)  { return m_dy; }
};


template<class T>
class DualStorage<T, 3>
{
public:
    T m_val;
    T m_dx;
    T m_dy;
    T m_dz;

    // To better enable Scalar Replacement of Aggregates and other
    // transformations, CLANG has easier time if the per element
    // constructors declared.
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage() {}
    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage(const T & val, const T & dx, const T & dy, const T & dz)
    : m_val(val)
    , m_dx(dx)
    , m_dy(dy)
    , m_dz(dz)
    {}

    OSL_HOSTDEVICE OSL_CONSTEXPR14 DualStorage(const DualStorage &other)
    : m_val(other.m_val)
    , m_dx(other.m_dx)
    , m_dy(other.m_dy)
    , m_dz(other.dz)
    {}

    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<0>) const { return m_val; }
    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<1>) const { return m_dx; }
    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<2>) const { return m_dy; }
    OSL_HOSTDEVICE constexpr const T& elem (ConstIndex<3>) const { return m_dz; }
    OSL_HOSTDEVICE T& elem (ConstIndex<0>)  { return m_val; }
    OSL_HOSTDEVICE T& elem (ConstIndex<1>)  { return m_dx; }
    OSL_HOSTDEVICE T& elem (ConstIndex<2>)  { return m_dy; }
    OSL_HOSTDEVICE T& elem (ConstIndex<3>)  { return m_dz; }
};



// Single generic implementation of Dual can handle any number of partials
// delegating access to element storage to the DualStorage base class
template<
         class T,           // Base data type
         int PARTIALS=1     // Number of dimentions of partial derivs
        >
class Dual : public DualStorage<T, PARTIALS> {
    static const int elements = PARTIALS+1;   // main value + partials
    static_assert (PARTIALS>=1, "Can't have a Dual with 0 partials");
public:
    using value_type = T;
    static OSL_FORCEINLINE OSL_HOSTDEVICE OSL_CONSTEXPR14 T zero() { return T(0.0); }

    using DualStorage<T, PARTIALS>::elem;
    template<int N>
    OSL_HOSTDEVICE constexpr const T partial (ConstIndex<N>) const { return elem(ConstIndex<N+1>()); }
    template<int N>
    OSL_HOSTDEVICE T& partial (ConstIndex<N>) { return elem(ConstIndex<N+1>()); }

    /// Default ctr leaves everything uninitialized
    ///
    OSL_HOSTDEVICE OSL_CONSTEXPR14 Dual ()
    : DualStorage<T, PARTIALS>()
    {}

    /// Construct a Dual from just a real value (derivs set to 0)
    ///
    OSL_HOSTDEVICE OSL_CONSTEXPR14 Dual (const T &x)
    : DualStorage<T, PARTIALS>()
    {
        val() = x;
        OSL_INDEX_LOOP(i, PARTIALS, {
           partial(i) = zero();
        });
    }

    /// Copy constructor from another Dual of same dimension and different,
    /// but castable, data type.
    template<class F>
    OSL_HOSTDEVICE OSL_CONSTEXPR14 Dual (const Dual<F,PARTIALS> &x)
    : DualStorage<T, PARTIALS>()
    {
        OSL_INDEX_LOOP(i, elements, {
            elem(i) = T(x.elem(i));
        });
    }

    //OSL_HOSTDEVICE OSL_CONSTEXPR14 Dual (const Dual &x) = default;
    OSL_HOSTDEVICE OSL_CONSTEXPR14 Dual (const Dual &x)
    : DualStorage<T, PARTIALS>(x)
    {}

    /// Construct a Dual from a real and infinitesimals.
    ///
    template<typename... DerivListT>
    OSL_HOSTDEVICE constexpr Dual (const T &x, const DerivListT & ...derivs)
    : DualStorage<T, PARTIALS>{ x, static_cast<T>(derivs)...}
    {
        static_assert(sizeof...(DerivListT)==PARTIALS, "Wrong number of initializers");
    }

protected:
    template<int... IntListT, typename... ValueListT>
    OSL_HOSTDEVICE OSL_FORCEINLINE void
    set_expander (pvt::int_sequence<IntListT...> /*indices*/, const ValueListT & ...values)
    {
        __OSL_EXPAND_PARAMETER_PACKS( elem(ConstIndex<IntListT>()) = values );
    }
public:

    template<typename... ValueListT>
    OSL_HOSTDEVICE void set (const ValueListT & ...values)
    {
        static_assert(sizeof...(ValueListT)==elements, "Wrong number of initializers");

        set_expander(pvt::make_int_sequence<elements>(), values...);
    }

    OSL_HOSTDEVICE OSL_CONSTEXPR14 Dual (std::initializer_list<T> vals) {
#if OIIO_CPLUSPLUS_VERSION >= 14
        static_assert (vals.size() == elements, "Wrong number of initializers");
#endif
        OSL_INDEX_LOOP(i, elements, {
            elem(i) = vals.begin()[i];
        });
    }

    /// Return the real value.
    OSL_HOSTDEVICE constexpr const T& val () const { return elem(ConstIndex<0>()); }
    OSL_HOSTDEVICE T& val () { return elem(ConstIndex<0>()); }

    /// Return the partial derivative with respect to x.
    OSL_HOSTDEVICE constexpr const T& dx () const { return elem(ConstIndex<1>()); }
    OSL_HOSTDEVICE T& dx () { return elem(ConstIndex<1>()); }

    /// Return the partial derivative with respect to y.
    template<typename ThisType = Dual, typename std::enable_if<ThisType::elements==3, int>::type = 0>
    OSL_HOSTDEVICE constexpr const T &
    dy () const {
        return elem(ConstIndex<2>());
    }
    template<typename ThisType = Dual, typename std::enable_if<ThisType::elements==3, int>::type = 0>
    OSL_HOSTDEVICE T& dy () {
        return elem(ConstIndex<2>());
    }

    /// Return the partial derivative with respect to z.
    /// Only valid if there are at least 3 partial dimensions.
    template<typename ThisType = Dual, typename std::enable_if<ThisType::elements==4, int>::type = 0>
    OSL_HOSTDEVICE constexpr const T& dz () const {
        return elem(ConstIndex<3>());
    }
    template<typename ThisType = Dual, typename std::enable_if<ThisType::elements==4, int>::type = 0>
    OSL_HOSTDEVICE T& dz () {
        return elem(ConstIndex<3>());
    }

    /// Clear the derivatives; leave the value alone.
    OSL_HOSTDEVICE void clear_d () {
        OSL_INDEX_LOOP(i, PARTIALS, {
            partial(i) = zero();
        });
    }

    /// Assignment of a real (the derivs are implicitly 0).
    OSL_HOSTDEVICE const Dual & operator= (const T &x) {
        val() = x;
        OSL_INDEX_LOOP(i, PARTIALS, {
            partial(i) = zero();
        });
        return *this;
    }

    OSL_HOSTDEVICE const Dual & operator= (const Dual &other) {
        OSL_INDEX_LOOP(i, elements, {
            elem(i) = other.elem(i);
        });
        return *this;
    }


    /// Stream output.  Format as: "val[dx,dy,...]"
    ///
    friend std::ostream& operator<< (std::ostream &out, const Dual &x) {
        out << x.val() << "[";
        OSL_INDEX_LOOP(i, PARTIALS, {
            out << (x.partial(i)) << ((i < PARTIALS-1) ? ',' : ']');
        });
        return out;
    }

};



/// is_Dual<TYPE>::value is true for Dual, false for everything else.
/// You can also evaluate an is_Dual<T> as a bool.
template<class T> struct is_Dual : std::false_type {};
template<class T, int P> struct is_Dual<Dual<T,P>> : std::true_type {};

/// Dualify<T>::type returns Dual<T> if T is not a dual, or just T if it's
/// already a dual.
template<class T> struct UnDual { typedef T type; };
template<class T, int P> struct UnDual<Dual<T,P>> { typedef T type; };

/// Dualify<T,P>::type returns Dual<T,P> if T is not a dual, or just T if
/// it's already a dual.
template<class T, int P=1> struct Dualify {
    typedef Dual<typename UnDual<T>::type, P> type;
};



/// Define Dual2<T> as a Dual<T,2> -- a T-based quantity with x and y
/// partial derivatives.
template<class T> using Dual2 = Dual<T,2>;



/// Addition of duals.
///
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator+ (const Dual<T,P> &a, const Dual<T,P> &b)
{
    Dual<T,P> result = a;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) += b.elem(i);
    });
    return result;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator+ (const Dual<T,P> &a, const T &b)
{
    Dual<T,P> result = a;
    result.val() += b;
    return result;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator+ (const T &a, const Dual<T,P> &b)
{
    Dual<T,P> result = b;
    result.val() += a;
    return result;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P>& operator+= (Dual<T,P> &a, const Dual<T,P> &b)
{
    OSL_INDEX_LOOP(i, P+1, {
        a.elem(i) += b.elem(i);
    });
    return a;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P>& operator+= (Dual<T,P> &a, const T &b)
{
    a.val() += b;
    return a;
}


/// Subtraction of duals.
///
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator- (const Dual<T,P> &a, const Dual<T,P> &b)
{
    Dual<T,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = a.elem(i) - b.elem(i);
    });
    return result;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator- (const Dual<T,P> &a, const T &b)
{
    Dual<T,P> result = a;
    result.val() -= b;
    return result;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator- (const T &a, const Dual<T,P> &b)
{
    Dual<T,P> result;
    result.val() = a - b.val();
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = -b.partial(i);
    });
    return result;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P>& operator-= (Dual<T,P> &a, const Dual<T,P> &b)
{
    OSL_INDEX_LOOP(i, P+1, {
        a.elem(i) -= b.elem(i);
    });
    return a;
}


template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P>& operator-= (Dual<T,P> &a, const T &b)
{
    a.val() -= b;
    return a;
}



/// Negation of duals.
///
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator- (const Dual<T,P> &a)
{
    Dual<T,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = -a.elem(i);
    });
    return result;
}


/// Multiplication of duals. This will work for any Dual<T>*Dual<S>
/// where T*S is meaningful.
template<class T, int P, class S>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 auto
operator* (const Dual<T,P> &a, const Dual<S,P> &b) -> Dual<decltype(a.val()*b.val()),P>
{
    // Use the chain rule
    Dual<decltype(a.val()*b.val()),P> result;
    result.val() = a.val() * b.val();
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = a.val()*b.partial(i) + a.partial(i)*b.val();
    });
    return result;
}


/// Multiplication of dual by a non-dual scalar. This will work for any
/// Dual<T> * S where T*S is meaningful.
template<class T, int P, class S,
         DUAL_REQUIRES(is_Dual<S>::value == false)>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 auto
operator* (const Dual<T,P> &a, const S &b) -> Dual<decltype(a.val()*b),P>
{
    Dual<decltype(a.val()*b),P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = a.elem(i) * b;
    });
    return result;
}


/// Multiplication of dual by scalar in place.
///
template<class T, int P, class S,
         DUAL_REQUIRES(is_Dual<S>::value == false)>
OSL_HOSTDEVICE OSL_FORCEINLINE const Dual<T,P>& operator*= (Dual<T,P> &a, const S &b)
{
    OSL_INDEX_LOOP(i, P+1, {
        a.elem(i) *= b;
    });
    return a;
}



/// Multiplication of dual by a non-dual scalar. This will work for any
/// Dual<T> * S where T*S is meaningful.
template<class T, int P, class S,
         DUAL_REQUIRES(is_Dual<S>::value == false)>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 auto
operator* (const S &b, const Dual<T,P> &a) -> Dual<decltype(a.val()*b),P>
{
    return a*b;
}



/// Division of duals.
///
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator/ (const Dual<T,P> &a, const Dual<T,P> &b)
{
    T bvalinv = T(1) / b.val();
    T aval_bval = a.val() / b.val();
    Dual<T,P> result;
    result.val() = aval_bval;
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = bvalinv * (a.partial(i) - aval_bval * b.partial(i));
    });
    return result;
}


/// Division of dual by scalar.
///
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator/ (const Dual<T,P> &a, const T &b)
{
    T bvalinv = T(1) / b;
    T aval_bval = a.val() / b;
    Dual<T,P> result;
    result.val() = aval_bval;
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = bvalinv * a.partial(i);
    });
    return result;
}


/// Division of scalar by dual.
///
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> operator/ (const T &aval, const Dual<T,P> &b)
{
    T bvalinv = T(1) / b.val();
    T aval_bval = aval / b.val();
    Dual<T,P> result;
    result.val() = aval_bval;
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = bvalinv * ( - aval_bval * b.partial(i));
    });
    return result;
}




template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator< (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() < b.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator< (const Dual<T,P> &a, const T &b) {
    return a.val() < b;
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator< (const T &a, const Dual<T,P> &b) {
    return a < b.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator> (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() > b.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator> (const Dual<T,P> &a, const T &b) {
    return a.val() > b;
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator> (const T &a, const Dual<T,P> &b) {
    return a > b.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator<= (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() <= b.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator<= (const Dual<T,P> &a, const T &b) {
    return a.val() <= b;
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator<= (const T &a, const Dual<T,P> &b) {
    return a <= b.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator>= (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() >= b.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator>= (const Dual<T,P> &a, const T &b) {
    return a.val() >= b;
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool operator>= (const T &a, const Dual<T,P> &b) {
    return a >= b.val();
}



// Eliminate the derivatives of a number, works for scalar as well as Dual.
template<class T> OSL_HOSTDEVICE OSL_FORCEINLINE constexpr const T&
removeDerivatives (const T &x) { return x; }

template<class T, int P> OSL_HOSTDEVICE OSL_FORCEINLINE constexpr const T&
removeDerivatives (const Dual<T,P> &x) { return x.val(); }


// Get the x derivative (or 0 for a non-Dual)
template<class T> OSL_HOSTDEVICE OSL_FORCEINLINE constexpr const T&
getXDerivative (const T & /*x*/) { return T(0); }

template<class T, int P> OSL_HOSTDEVICE OSL_FORCEINLINE constexpr const T&
getXDerivative (const Dual<T,P> &x) { return x.dx(); }


// Get the y derivative (or 0 for a non-Dual)
template<class T> OSL_HOSTDEVICE OSL_FORCEINLINE constexpr const T&
getYDerivative (const T & /*x*/) { return T(0); }

template<class T, int P> OSL_HOSTDEVICE OSL_FORCEINLINE constexpr const T&
getYDerivative (const Dual<T,P> &x) { return x.dy(); }


// Simple templated "copy" function
template<class T> OSL_HOSTDEVICE OSL_FORCEINLINE void
assignment (T &a, const T &b) { a = b; }
template<class T, int P> OSL_HOSTDEVICE OSL_FORCEINLINE void
assignment (T &a, const Dual<T,P> &b) { a = b.val(); }

// Templated value equality. For scalars, it's the same as regular ==.
// For Dual's, this only tests the value, not the derivatives. This solves
// a pesky source of confusion about whether operator== of Duals ought to
// return if just their value is equal or if the whole struct (including
// derivs) are equal.
template<class T>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool equalVal (const T &x, const T &y) {
    return x == y;
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool equalVal (const Dual<T,P> &x, const Dual<T,P> &y) {
    return x.val() == y.val();
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool equalVal (const Dual<T,P> &x, const T &y) {
    return x.val() == y;
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr bool equalVal (const T &x, const Dual<T,P> &y) {
    return x == y.val();
}



/// equalAll is the same as equalVal generally, but for two Duals of the same
/// type, they also compare derivs.
template<class T>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 bool equalAll (const T &x, const T &y) {
    return equalVal(x, y);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 bool equalAll (const Dual<T,P> &x, const Dual<T,P> &y) {
    bool r = true;
    OSL_INDEX_LOOP(i, P+1, {
        r &= (x.elem(i) == y.elem(i));
    });
    return r;
}




// Helper for constructing the result of a Dual function of one variable,
// given the scalar version and its derivative. Suppose you have scalar
// function f(scalar), then the dual function F(<u,u'>) is defined as:
//    F(<u,u'>) = < f(u), f'(u)*u' >
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P>
dualfunc (const Dual<T,P>& u, const T& f_val, const T& df_val)
{
    Dual<T,P> result;
    result.val() = f_val;
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = df_val * u.partial(i);
    });
    return result;
}


// Helper for constructing the result of a Dual function of two variables,
// given the scalar version and its derivative. In general, the dual-form of
// the primitive function 'f(u,v)' is:
//   F(<u,u'>, <v,v'>) = < f(u,v), dfdu(u,v)u' + dfdv(u,v)v' >
// (from http://en.wikipedia.org/wiki/Automatic_differentiation)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P>
dualfunc (const Dual<T,P>& u, const Dual<T,P>& v,
          const T& f_val, const T& dfdu_val, const T& dfdv_val)
{
    Dual<T,P> result;
    result.val() = f_val;
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = dfdu_val * u.partial(i) + dfdv_val * v.partial(i);
    });
    return result;
}


// Helper for construction the result of a Dual function of three variables.
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P>
dualfunc (const Dual<T,P>& u, const Dual<T,P>& v, const Dual<T,P>& w,
          const T& f_val, const T& dfdu_val, const T& dfdv_val, const T& dfdw_val)
{
    Dual<T,P> result;
    result.val() = f_val;
    OSL_INDEX_LOOP(i, P, {
        result.partial(i) = dfdu_val * u.partial(i) + dfdv_val * v.partial(i) + dfdw_val * w.partial(i);
    });
    return result;
}



/// Fast negation of duals, in cases where you aren't too picky about the
/// difference between +0 and -0. (Courtesy of Alex Wells, Intel) Please see
/// OIIO's fast_math.h fast_neg for a more full explanation of why this is
/// faster.
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> fast_neg (const Dual<T,P> &a)
{
    Dual<T,P> result;
    OSL_INDEX_LOOP(i, P+1, {
        result.elem(i) = T(0) - a.elem(i);
    });
    return result;
}


// f(x) = cos(x), f'(x) = -sin(x)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> cos (const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::sincos(a.val(), &sina, &cosa);
    return dualfunc (a, cosa, -sina);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_cos(const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::fast_sincos (a.val(), &sina, &cosa);
    return dualfunc (a, cosa, -sina);
}

// f(x) = sin(x),  f'(x) = cos(x)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> sin (const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::sincos(a.val(), &sina, &cosa);
    return dualfunc (a, sina, cosa);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_sin(const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::fast_sincos (a.val(), &sina, &cosa);
    return dualfunc (a, sina, cosa);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE void sincos(const Dual<T,P> &a, Dual<T,P> *sine, Dual<T,P> *cosine)
{
	T sina, cosa;
	OIIO::sincos(a.val(), &sina, &cosa);
    *cosine = dualfunc (a, cosa, -sina);
    *sine   = dualfunc (a, sina, cosa);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE void fast_sincos(const Dual<T,P> &a, Dual<T,P> *sine, Dual<T,P> *cosine)
{
	T sina, cosa;
	OIIO::fast_sincos(a.val(), &sina, &cosa);
    *cosine = dualfunc (a, cosa, -sina);
    *sine   = dualfunc (a, sina, cosa);
}

// f(x) = tan(x), f'(x) = sec^2(x)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> tan (const Dual<T,P> &a)
{
    T tana  = std::tan (a.val());
    T cosa  = std::cos (a.val());
    T sec2a = T(1)/(cosa*cosa);
    return dualfunc (a, tana, sec2a);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_tan(const Dual<T,P> &a)
{
    T tana  = OIIO::fast_tan (a.val());
    T cosa  = OIIO::fast_cos (a.val());
    T sec2a = 1 / (cosa * cosa);
    return dualfunc (a, tana, sec2a);
}

// f(x) = cosh(x), f'(x) = sinh(x)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> cosh (const Dual<T,P> &a)
{
    T f = std::cosh(a.val());
    T df = std::sinh(a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_cosh(const Dual<T,P> &a)
{
    T f = OIIO::fast_cosh(a.val());
    T df = OIIO::fast_sinh(a.val());
    return dualfunc (a, f, df);
}


// f(x) = sinh(x), f'(x) = cosh(x)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> sinh (const Dual<T,P> &a)
{
    T f = std::sinh(a.val());
    T df = std::cosh(a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_sinh(const Dual<T,P> &a)
{
    T f = OIIO::fast_sinh(a.val());
    T df = OIIO::fast_cosh(a.val());
    return dualfunc (a, f, df);
}

// f(x) = tanh(x), f'(x) = sech^2(x)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> tanh (const Dual<T,P> &a)
{
    T tanha = std::tanh(a.val());
    T cosha = std::cosh(a.val());
    T sech2a = T(1)/(cosha*cosha);
    return dualfunc (a, tanha, sech2a);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_tanh(const Dual<T,P> &a)
{
    T tanha = OIIO::fast_tanh(a.val());
    T cosha = OIIO::fast_cosh(a.val());
    T sech2a = T(1) / (cosha * cosha);
    return dualfunc (a, tanha, sech2a);
}

// f(x) = acos(x), f'(x) = -1/(sqrt(1 - x^2))
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> safe_acos (const Dual<T,P> &a)
{
    if (a.val() >= T(1))
        return Dual<T,P> (T(0));
    if (a.val() <= T(-1))
        return Dual<T,P> (T(M_PI));
    T f = std::acos (a.val());
    T df = -T(1) / std::sqrt (T(1) - a.val()*a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_acos(const Dual<T,P> &a)
{
    T f = OIIO::fast_acos(a.val());
    T df = fabsf(a.val()) < 1.0f ? -1.0f / sqrtf(1.0f - a.val() * a.val()) : 0.0f;
    return dualfunc (a, f, df);
}

// f(x) = asin(x), f'(x) = 1/(sqrt(1 - x^2))
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> safe_asin (const Dual<T,P> &a)
{
    if (a.val() >= T(1))
        return Dual<T,P> (T(M_PI/2));
    if (a.val() <= T(-1))
        return Dual<T,P> (T(-M_PI/2));
    T f = std::asin (a.val());
    T df = T(1) / std::sqrt (T(1) - a.val()*a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_asin(const Dual<T,P> &a)
{
    T f = OIIO::fast_asin(a.val());
    T df = fabsf(a.val()) < 1.0f ? 1.0f / sqrtf(1.0f - a.val() * a.val()) : 0.0f;
    return dualfunc (a, f, df);
}


// f(x) = atan(x), f'(x) = 1/(1 + x^2)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> atan (const Dual<T,P> &a)
{
    T f = std::atan (a.val());
    T df = T(1) / (T(1) + a.val()*a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_atan(const Dual<T,P> &a)
{
    T f = OIIO::fast_atan(a.val());
    T df = 1.0f / (1.0f + a.val() * a.val());
    return dualfunc (a, f, df);
}

// f(x,y) = atan2(y,x); f'(x) =  y x' / (x^2 + y^2),
//                      f'(y) = -x y' / (x^2 + y^2)
// reference:  http://en.wikipedia.org/wiki/Atan2
// (above link has other formulations)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> atan2 (const Dual<T,P> &y, const Dual<T,P> &x)
{
    T atan2xy = std::atan2 (y.val(), x.val());
    T denom = (x.val() == T(0) && y.val() == T(0)) ? T(0) : T(1) / (x.val()*x.val() + y.val()*y.val());
    return dualfunc (y, x, atan2xy, -x.val()*denom, y.val()*denom);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_atan2(const Dual<T,P> &y, const Dual<T,P> &x)
{
    T atan2xy = OIIO::fast_atan2(y.val(), x.val());
    T denom = (x.val() == 0 && y.val() == 0) ? 0.0f : 1.0f / (x.val() * x.val() + y.val() * y.val());
    return dualfunc (y, x, atan2xy, -x.val()*denom, y.val()*denom);
}

// f(x) = log(a), f'(x) = 1/x
// (log base e)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> safe_log (const Dual<T,P> &a)
{
    T f = OIIO::safe_log(a.val());
    T df = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / a.val();
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_log(const Dual<T,P> &a)
{
    T f = OIIO::fast_log(a.val());
    T df = a.val() < std::numeric_limits<float>::min() ? 0.0f : 1.0f / a.val();
    return dualfunc (a, f, df);
}


// to compute pow(u,v), we need the dual-form representation of
// the pow() operator.  In general, the dual-form of the primitive
// function 'g' is:
//   g(<u,u'>, <v,v'>) = < g(u,v), dgdu(u,v)u' + dgdv(u,v)v' >
//   (from http://en.wikipedia.org/wiki/Automatic_differentiation)
// so, pow(u,v) = < u^v, vu^(v-1) u' + log(u)u^v v' >
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> safe_pow (const Dual<T,P> &u, const Dual<T,P> &v)
{
    // NOTE: this function won't return exactly the same as pow(x,y) because we
    // use the identity u^v=u * u^(v-1) which does not hold in all cases for our
    // "safe" variant (nor does it hold in general in floating point arithmetic).
    T powuvm1 = OIIO::safe_pow(u.val(), v.val() - T(1));
    T powuv   = powuvm1 * u.val();
    T logu    = u.val() > 0 ? OIIO::safe_log(u.val()) : T(0);
    return dualfunc (u, v, powuv, v.val()*powuvm1, logu*powuv);
}
// Fallthrough to OIIO::safe_pow for floats and vectors
using OIIO::safe_pow;

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_safe_pow(const Dual<T,P> &u, const Dual<T,P> &v)
{
    // NOTE: same issue as above (fast_safe_pow does even more clamping)
    T powuvm1 = OIIO::fast_safe_pow (u.val(), v.val() - 1.0f);
    T powuv   = powuvm1 * u.val();
    T logu    = u.val() > 0 ? OIIO::fast_log(u.val()) : 0.0f;
    return dualfunc (u, v, powuv, v.val()*powuvm1, logu*powuv);
}

// f(x) = log2(x), f'(x) = 1/(x*log2)
// (log base 2)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> safe_log2 (const Dual<T,P> &a)
{
    T f = safe_log2(a.val());
    T df = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / (a.val() * T(M_LN2));
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_log2(const Dual<T,P> &a)
{
    T f = OIIO::fast_log2(a.val());
    T aln2 = a.val() * float(M_LN2);
    T df = aln2 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln2;
    return dualfunc (a, f, df);
}

// f(x) = log10(x), f'(x) = 1/(x*log10)
// (log base 10)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> safe_log10 (const Dual<T,P> &a)
{
    T f = safe_log10(a.val());
    T df = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / (a.val() * T(M_LN10));
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_log10(const Dual<T,P> &a)
{
    T f  = OIIO::fast_log10(a.val());
    T aln10 = a.val() * float(M_LN10);
    T df  = aln10 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln10;
    return dualfunc (a, f, df);
}

// f(x) = e^x, f'(x) = e^x
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> exp (const Dual<T,P> &a)
{
    T f = std::exp(a.val());
    return dualfunc (a, f, f);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_exp(const Dual<T,P> &a)
{
    T f = OIIO::fast_exp(a.val());
    return dualfunc (a, f, f);
}

// f(x) = 2^x, f'(x) = (2^x)*log(2)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> exp2 (const Dual<T,P> &a)
{
    using std::exp2;
    T f = exp2(a.val());
    return dualfunc (a, f, f*T(M_LN2));
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_exp2(const Dual<T,P> &a)
{
    T f = OIIO::fast_exp2(float(a.val()));
    return dualfunc (a, f, f*T(M_LN2));
}


// f(x) = e^x - 1, f'(x) = e^x
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> expm1 (const Dual<T,P> &a)
{
    using std::expm1;
    T f  = expm1(a.val());
    T df = std::exp  (a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_expm1(const Dual<T,P> &a)
{
    T f  = OIIO::fast_expm1(a.val());
    T df = OIIO::fast_exp  (a.val());
    return dualfunc (a, f, df);
}

// f(x) = erf(x), f'(x) = (2e^(-x^2))/sqrt(pi)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> erf (const Dual<T,P> &a)
{
    using std::erf;
    T f = erf (a.val());
    const T two_over_sqrt_pi = T(1.128379167095512573896158903);
    T df = std::exp (-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_erf(const Dual<T,P> &a)
{
    T f = OIIO::fast_erf (float(a.val())); // float version!
    const T two_over_sqrt_pi = 1.128379167095512573896158903f;
    T df = OIIO::fast_exp(-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}

// f(x) = erfc(x), f'(x) = -(2e^(-x^2))/sqrt(pi)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> erfc (const Dual<T,P> &a)
{
    using std::erfc;
    T f = erfc (a.val()); // float version!
    const T two_over_sqrt_pi = -T(1.128379167095512573896158903);
    T df = std::exp (-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_erfc(const Dual<T,P> &a)
{
    T f = OIIO::fast_erfc (float(a.val())); // float version!
    const T two_over_sqrt_pi = -1.128379167095512573896158903f;
    T df = OIIO::fast_exp(-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}


// f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE OSL_CONSTEXPR14 Dual<T,P> sqrt (const Dual<T,P> &a)
{
    Dual<T,P> result;
    if (OSL_LIKELY(a.val() > T(0))) {
        T f  = std::sqrt(a.val());
        T df = T(0.5) / f;
        result = dualfunc (a, f, df);
    } else {
        result = Dual<T,P> (T(0));
    }
    return result;
}

// f(x) = 1/sqrt(x), f'(x) = -1/(2*x^(3/2))
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> inversesqrt (const Dual<T,P> &a)
{
    // do we want to print an error message?
    Dual<T,P> result;
    if (OSL_LIKELY(a.val() > T(0))) {
        T f  = T(1)/std::sqrt(a.val());
        T df = T(-0.5)*f/a.val();
        result = dualfunc (a, f, df);
    } else {
        result = Dual<T,P> (T(0));
    }
    return result;
}

// f(x) = cbrt(x), f'(x) = 1/(3*x^(2/3))
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> cbrt (const Dual<T,P> &a)
{
    if (OSL_LIKELY(a.val() != T(0))) {
        T f = std::cbrt(a.val());
        T df = T(1) / (T(3) * f * f);
        return dualfunc(a, f, df);
    }
    return Dual<T,P> (T(0));
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fast_cbrt (const Dual<T,P> &a)
{
    if (OSL_LIKELY(a.val() != T(0))) {
        T f = OIIO::fast_cbrt(float(a.val())); // float version!
        T df = T(1) / (T(3) * f * f);
        return dualfunc(a, f, df);
    }
    return Dual<T,P> (T(0));
}

// (fx) = x*(1-a) + y*a, f'(x) = (1-a)x' + (y - x)*a' + a*y'
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> mix (const Dual<T,P> &x, const Dual<T,P> &y, const Dual<T,P> &a)
{
    T mixval = x.val()*(T(1)-a.val()) + y.val()*a.val();
    return dualfunc (x, y, a, mixval, T(1)-a.val(), a.val(), y.val() - x.val());
}

template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> fabs (const Dual<T,P> &x)
{
    return x.val() >= T(0) ? x : -x;
}


#ifdef OIIO_FMATH_HAS_SAFE_FMOD
using OIIO::safe_fmod;
#else
OSL_FORCEINLINE OSL_HOSTDEVICE float safe_fmod (float a, float b)
{
    if (OSL_LIKELY(b != 0.0f)) {
#if 0
        return std::fmod (a,b);
        // std::fmod was getting called serially instead of vectorizing, so
        // we will just do the calculation ourselves
#else
        // This formulation is equivalent but much faster in our benchmarks,
        // also vectorizes better.
        // The floating-point remainder of the division operation
        // a/b is a - N*b, where N = a/b with its fractional part truncated.
        int N = static_cast<int>(a/b);
        return a - N*b;
#endif
    }
    return 0.0f;
}
#endif

template<class T, int P>
OSL_FORCEINLINE OSL_HOSTDEVICE Dual<T,P>
safe_fmod (const Dual<T,P>& a, const Dual<T,P>& b)
{
    Dual<T,P> result = a;
    result.val() = safe_fmod(a.val(), b.val());
    return result;
}



OSL_HOSTDEVICE OSL_FORCEINLINE float smoothstep(float e0, float e1, float x) {
    if (x < e0) return 0.0f;
    else if (x >= e1) return 1.0f;
    else {
        float t = (x - e0)/(e1 - e0);
        return (3.0f-2.0f*t)*(t*t);
    }
}

// f(t) = (3-2t)t^2,   t = (x-e0)/(e1-e0)
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE Dual<T,P> smoothstep (const Dual<T,P> &e0, const Dual<T,P> &e1, const Dual<T,P> &x)
{
   if (x.val() < e0.val()) {
      return Dual<T,P> (T(0));
   }
   else if (x.val() >= e1.val()) {
      return Dual<T,P> (T(1));
   }
   Dual<T,P> t = (x - e0)/(e1-e0);
   return  (T(3) - T(2)*t)*t*t;
}


#ifdef __CUDA_ARCH__
template<> inline OSL_HOSTDEVICE OSL_CONSTEXPR14 float3 Dual<float3, 2>::zero() {
    return float3{0.f, 0.f, 0.f};
}
#endif


// ceil(Dual) loses derivatives
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr float ceil (const Dual<T,P> &x)
{
    return std::ceil(x.val());
}


// floor(Dual) loses derivatives
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr float floor (const Dual<T,P> &x)
{
    return std::floor(x.val());
}


// floor, cast to an int.
template<class T, int P>
OSL_HOSTDEVICE OSL_FORCEINLINE constexpr int
ifloor (const Dual<T,P> &x)
{
    return (int)floor(x);
}


OSL_NAMESPACE_EXIT
