/*
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <initializer_list>
#include <OSL/oslversion.h>
#include <OpenImageIO/fmath.h>
OSL_NAMESPACE_ENTER


// Shortcut notation for enable_if trickery
# define DUAL_REQUIRES(...) \
  typename std::enable_if<(__VA_ARGS__), bool>::type = true



/// Dual numbers are used to represent values and their derivatives.
///
/// The method is summarized in:
///      Piponi, Dan, "Automatic Differentiation, C++ Templates,
///      and Photogrammetry," Journal of Graphics, GPU, and Game
///      Tools (JGT), Vol 9, No 4, pp. 41-55 (2004).
///
///

template<
         class T,           // Base data type
         int PARTIALS=1     // Number of dimentions of partial derivs
        >
class Dual {
    static const int elements = PARTIALS+1;   // main value + partials
    static_assert (PARTIALS>=1, "Can't have a Dual with 0 partials");
public:
    using value_type = T;

    /// Default ctr leaves everything uninitialized
    ///
    constexpr Dual () { }

    /// Construct a Dual from just a real value (derivs set to 0)
    ///
    OIIO_CONSTEXPR14 Dual (const T &x) {
        m_data[0] = x;
        for (int i = 1; i <= PARTIALS; ++i)
            m_data[i] = T(0.0);
    }

    /// Copy constructor from another Dual of same type and dimension.
    OIIO_CONSTEXPR14 Dual (const Dual &x) {
        for (int i = 0; i <= PARTIALS; ++i)
            m_data[i] = T(x.m_data[i]);
    }

#if 1
    /// Copy constructor from another Dual of same dimension and different,
    /// but castable, data type.
    template<class F>
    OIIO_CONSTEXPR14 Dual (const Dual<F,PARTIALS> &x) {
        for (int i = 0; i <= PARTIALS; ++i)
            m_data[i] = T(x.elem(i));
    }
#endif

    /// Construct a Dual from a real and infinitesimals.
    ///
    constexpr Dual (const T &x, const T &dx) : m_data{ x, dx } {
        static_assert(PARTIALS==1, "Wrong number of initializers");
    }
    constexpr Dual (const T &x, const T &dx, const T &dy)
        : m_data{ x, dx, dy }
    {
        static_assert(PARTIALS==2, "Wrong number of initializers");
    }
    void set (const T &x, const T &dx) {
        static_assert(PARTIALS==1, "Wrong number of initializers");
        m_data[0] = x;
        m_data[1] = dx;
    }
    void set (const T &x, const T &dx, const T &dy) {
        static_assert(PARTIALS==2, "Wrong number of initializers");
        m_data[0] = x;
        m_data[1] = dx;
        m_data[2] = dy;
    }

    OIIO_CONSTEXPR14 Dual (std::initializer_list<T> vals) {
        static_assert (vals.size() == elements, "Wrong number of initializers");
        for (int i = 0; i < elements; ++i)
            m_data[i] = vals.begin()[i];
    }

    /// Return the real value.
    constexpr const T& val () const { return m_data[0]; }
    T& val () { return m_data[0]; }

    /// Return the partial derivative with respect to x.
    constexpr const T& dx () const { return m_data[1]; }
    T& dx () { return m_data[1]; }

    /// Return the partial derivative with respect to y.
    /// Only valid if there are at least 2 partial dimensions.
    constexpr const T& dy () const {
        static_assert(PARTIALS>=2, "Cannot call dy without at least 2 partials");
        return m_data[2];
    }
    T& dy () {
        static_assert(PARTIALS>=2, "Cannot call dy without at least 2 partials");
        return m_data[2];
    }

    /// Return the partial derivative with respect to z.
    /// Only valid if there are at least 3 partial dimensions.
    constexpr const T& dz () const {
        static_assert(PARTIALS>=3, "Cannot call dz without at least 3 partials");
        return m_data[3];
    }
    T& dz () {
        static_assert(PARTIALS>=3, "Cannot call dz without at least 3 partials");
        return m_data[3];
    }

    /// Clear the derivatives; leave the value alone.
    void clear_d () {
        for (int i = 1; i <= PARTIALS; ++i)
            m_data[i] = T(0.0);
    }

    /// Return the i-th partial derivative.
    constexpr const T& partial (int i) const {
        return m_data[i+1];
    }

    /// Return a mutable reference to the i-th partial derivative.
    /// Use with caution!
    T& partial (int i) {
        return m_data[i+1];
    }

    /// Assignment of a real (the derivs are implicitly 0).
    const Dual & operator= (const T &x) {
        m_data[0] = x;
        for (int i = 1; i <= PARTIALS; ++i)
            m_data[i] = T(0);
        return *this;
    }

    /// Access like an array -- be careful! Element 0 is the main value,
    /// elements [1..PARTIALS] are the infinitessimals.
    constexpr const T& elem (int i) const { return m_data[i]; }
    T& elem (int i) { return m_data[i]; }

    /// Stream output.  Format as: "val[dx,dy,...]"
    ///
    friend std::ostream& operator<< (std::ostream &out, const Dual &x) {
        out << x.m_data[0] << "[";
        for (int i = 1; i < PARTIALS; ++i)
            out << x.m_data[i] << ',';
        return out << x.m_data[PARTIALS] << "]";
    }

private:
    T m_data[elements];  ///< [0] is the value, [1..PARTIALS] are derivs
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
inline OIIO_CONSTEXPR14 Dual<T,P> operator+ (const Dual<T,P> &a, const Dual<T,P> &b)
{
    Dual<T,P> result = a;
    for (int i = 0; i <= P; ++i)
        result.elem(i) += b.elem(i);
    return result;
}


template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator+ (const Dual<T,P> &a, const T &b)
{
    Dual<T,P> result = a;
    result.val() += b;
    return result;
}


template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator+ (const T &a, const Dual<T,P> &b)
{
    Dual<T,P> result = b;
    result.val() += a;
    return result;
}


template<class T, int P>
inline Dual<T,P>& operator+= (Dual<T,P> &a, const Dual<T,P> &b)
{
    for (int i = 0; i <= P; ++i)
        a.elem(i) += b.elem(i);
    return a;
}


template<class T, int P>
inline Dual<T,P>& operator+= (Dual<T,P> &a, const T &b)
{
    a.val() += b;
    return a;
}


/// Subtraction of duals.
///
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator- (const Dual<T,P> &a, const Dual<T,P> &b)
{
    Dual<T,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i) = a.elem(i) - b.elem(i);
    return result;
}


template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator- (const Dual<T,P> &a, const T &b)
{
    Dual<T,P> result = a;
    result.val() -= b;
    return result;
}


template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator- (const T &a, const Dual<T,P> &b)
{
    Dual<T,P> result;
    result.val() = a - b.val();
    for (int i = 1; i <= P; ++i)
        result.elem(i) = -b.elem(i);
    return result;
}


template<class T, int P>
inline Dual<T,P>& operator-= (Dual<T,P> &a, const Dual<T,P> &b)
{
    for (int i = 0; i <= P; ++i)
        a.elem(i) -= b.elem(i);
    return a;
}


template<class T, int P>
inline Dual<T,P>& operator-= (Dual<T,P> &a, const T &b)
{
    a.val() -= b.val();
    return a;
}



/// Negation of duals.
///
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator- (const Dual<T,P> &a)
{
    Dual<T,P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i) = -a.elem(i);
    return result;
}


/// Multiplication of duals. This will work for any Dual<T>*Dual<S>
/// where T*S is meaningful.
template<class T, int P, class S>
inline OIIO_CONSTEXPR14 auto
operator* (const Dual<T,P> &a, const Dual<S,P> &b) -> Dual<decltype(a.elem(0)*b.elem(0)),P>
{
    // Use the chain rule
    Dual<decltype(a.elem(0)*b.elem(0)),P> result;
    result.val() = a.val() * b.val();
    for (int i = 1; i <= P; ++i)
        result.elem(i) = a.val()*b.elem(i) + a.elem(i)*b.val();
    return result;
}


/// Multiplication of dual by a non-dual scalar. This will work for any
/// Dual<T> * S where T*S is meaningful.
template<class T, int P, class S,
         DUAL_REQUIRES(is_Dual<S>::value == false)>
inline OIIO_CONSTEXPR14 auto
operator* (const Dual<T,P> &a, const S &b) -> Dual<decltype(a.elem(0)*b),P>
{
    Dual<decltype(a.elem(0)*b),P> result;
    for (int i = 0; i <= P; ++i)
        result.elem(i) = a.elem(i) * b;
    return result;
}


/// Multiplication of dual by scalar in place.
///
template<class T, int P, class S,
         DUAL_REQUIRES(is_Dual<S>::value == false)>
inline const Dual<T,P>& operator*= (Dual<T,P> &a, const S &b)
{
    for (int i = 0; i <= P; ++i)
        a.elem(i) *= b;
    return a;
}



/// Multiplication of dual by a non-dual scalar. This will work for any
/// Dual<T> * S where T*S is meaningful.
template<class T, int P, class S,
         DUAL_REQUIRES(is_Dual<S>::value == false)>
inline OIIO_CONSTEXPR14 auto
operator* (const S &b, const Dual<T,P> &a) -> Dual<decltype(a.elem(0)*b),P>
{
    return a*b;
}



/// Division of duals.
///
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator/ (const Dual<T,P> &a, const Dual<T,P> &b)
{
    T bvalinv = 1.0f / b.val();
    T aval_bval = a.val() * bvalinv;
    Dual<T,P> result;
    result.val() = aval_bval;
    for (int i = 1; i <= P; ++i)
        result.elem(i) = bvalinv * (a.elem(i) - aval_bval * b.elem(i));
    return result;
}


/// Division of dual by scalar.
///
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator/ (const Dual<T,P> &a, const T &b)
{
    T binv = 1.0f / b;
    return a * binv;
}


/// Division of scalar by dual.
///
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> operator/ (const T &aval, const Dual<T,P> &b)
{
    T bvalinv = 1.0f / b.val();
    T aval_bval = aval * bvalinv;
    Dual<T,P> result;
    result.val() = aval_bval;
    for (int i = 1; i <= P; ++i)
        result.elem(i) = bvalinv * ( - aval_bval * b.elem(i));
    return result;
}




template<class T, int P>
inline constexpr bool operator< (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() < b.val();
}

template<class T, int P>
inline constexpr bool operator< (const Dual<T,P> &a, const T &b) {
    return a.val() < b;
}

template<class T, int P>
inline constexpr bool operator> (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() > b.val();
}

template<class T, int P>
inline constexpr bool operator> (const Dual<T,P> &a, const T &b) {
    return a.val() > b;
}

template<class T, int P>
inline constexpr bool operator<= (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() <= b.val();
}

template<class T, int P>
inline constexpr bool operator<= (const Dual<T,P> &a, const T &b) {
    return a.val() <= b;
}

template<class T, int P>
inline constexpr bool operator>= (const Dual<T,P> &a, const Dual<T,P> &b) {
    return a.val() >= b.val();
}

template<class T, int P>
inline constexpr bool operator>= (const Dual<T,P> &a, const T &b) {
    return a.val() >= b;
}



// Eliminate the derivatives of a number, works for scalar as well as Dual.
template<class T> inline constexpr const T&
removeDerivatives (const T &x) { return x; }

template<class T, int P> inline constexpr const T&
removeDerivatives (const Dual<T,P> &x) { return x.val(); }


// Get the x derivative (or 0 for a non-Dual)
template<class T> inline constexpr const T&
getXDerivative (const T &x) { return T(0); }

template<class T, int P> inline constexpr const T&
getXDerivative (const Dual<T,P> &x) { return x.partial(0); }


// Get the y derivative (or 0 for a non-Dual)
template<class T> inline constexpr const T&
getYDerivative (const T &x) { return T(0); }

template<class T, int P> inline constexpr const T&
getYDerivative (const Dual<T,P> &x) { return x.partial(1); }


// Simple templated "copy" function
template<class T> inline void
assignment (T &a, T &b) { a = b; }
template<class T, int P> inline void
assignment (T &a, Dual<T,P> &b) { a = b.val(); }

// Templated value equality. For scalars, it's the same as regular ==.
// For Dual's, this only tests the value, not the derivatives. This solves
// a pesky source of confusion about whether operator== of Duals ought to
// return if just their value is equal or if the whole struct (including
// derivs) are equal.
template<class T>
inline constexpr bool equalVal (const T &x, const T &y) {
    return x == y;
}

template<class T, int P>
inline constexpr bool equalVal (const Dual<T,P> &x, const Dual<T,P> &y) {
    return x.val() == y.val();
}

template<class T, int P>
inline constexpr bool equalVal (const Dual<T,P> &x, const T &y) {
    return x.val() == y;
}

template<class T, int P>
inline constexpr bool equalVal (const T &x, const Dual<T,P> &y) {
    return x == y.val();
}



// Helper for constructing the result of a Dual function of one variable,
// given the scalar version and its derivative. Suppose you have scalar
// function f(scalar), then the dual function F(<u,u'>) is defined as:
//    F(<u,u'>) = < f(u), f'(u)*u' >
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P>
dualfunc (const Dual<T,P>& u, const T& f_val, const T& df_val)
{
    Dual<T,P> result;
    result.val() = f_val;
    for (int i = 1; i <= P; ++i)
        result.elem(i) = df_val * u.elem(i);
    return result;
}

// Helper for constructing the result of a Dual function of two variables,
// given the scalar version and its derivative. In general, the dual-form of
// the primitive function 'f(u,v)' is:
//   F(<u,u'>, <v,v'>) = < f(u,v), dfdu(u,v)u' + dfdv(u,v)v' >
// (from http://en.wikipedia.org/wiki/Automatic_differentiation)
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P>
dualfunc (const Dual<T,P>& u, const Dual<T,P>& v,
          const T& f_val, const T& dfdu_val, const T& dfdv_val)
{
    Dual<T,P> result;
    result.val() = f_val;
    for (int i = 1; i <= P; ++i)
        result.elem(i) = dfdu_val * u.elem(i) + dfdv_val * v.elem(i);
    return result;
}


// Helper for construction the result of a Dual function of three variables.
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P>
dualfunc (const Dual<T,P>& u, const Dual<T,P>& v, const Dual<T,P>& w,
          const T& f_val, const T& dfdu_val, const T& dfdv_val, const T& dfdw_val)
{
    Dual<T,P> result;
    result.val() = f_val;
    for (int i = 1; i <= P; ++i)
        result.elem(i) = dfdu_val * u.elem(i) + dfdv_val * v.elem(i) + dfdw_val * w.elem(i);
    return result;
}



// f(x) = cos(x), f'(x) = -sin(x)
template<class T, int P>
inline Dual<T,P> cos (const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::sincos(a.val(), &sina, &cosa);
    return dualfunc (a, cosa, -sina);
}

template<class T, int P>
inline Dual<T,P> fast_cos(const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::fast_sincos (a.val(), &sina, &cosa);
    return dualfunc (a, cosa, -sina);
}

// f(x) = sin(x),  f'(x) = cos(x)
template<class T, int P>
inline Dual<T,P> sin (const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::sincos(a.val(), &sina, &cosa);
    return dualfunc (a, sina, cosa);
}

template<class T, int P>
inline Dual<T,P> fast_sin(const Dual<T,P> &a)
{
    T sina, cosa;
    OIIO::fast_sincos (a.val(), &sina, &cosa);
    return dualfunc (a, sina, cosa);
}

template<class T, int P>
inline void sincos(const Dual<T,P> &a, Dual<T,P> *sine, Dual<T,P> *cosine)
{
	T sina, cosa;
	OIIO::sincos(a.val(), &sina, &cosa);
    *cosine = dualfunc (a, cosa, -sina);
    *sine   = dualfunc (a, sina, cosa);
}

template<class T, int P>
inline void fast_sincos(const Dual<T,P> &a, Dual<T,P> *sine, Dual<T,P> *cosine)
{
	T sina, cosa;
	OIIO::fast_sincos(a.val(), &sina, &cosa);
    *cosine = dualfunc (a, cosa, -sina);
    *sine   = dualfunc (a, sina, cosa);
}

// f(x) = tan(x), f'(x) = sec^2(x)
template<class T, int P>
inline Dual<T,P> tan (const Dual<T,P> &a)
{
    T tana  = std::tan (a.val());
    T cosa  = std::cos (a.val());
    T sec2a = T(1)/(cosa*cosa);
    return dualfunc (a, tana, sec2a);
}

template<class T, int P>
inline Dual<T,P> fast_tan(const Dual<T,P> &a)
{
    T tana  = OIIO::fast_tan (a.val());
    T cosa  = OIIO::fast_cos (a.val());
    T sec2a = 1 / (cosa * cosa);
    return dualfunc (a, tana, sec2a);
}

// f(x) = cosh(x), f'(x) = sinh(x)
template<class T, int P>
inline Dual<T,P> cosh (const Dual<T,P> &a)
{
    T f = std::cosh(a.val());
    T df = std::sinh(a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_cosh(const Dual<T,P> &a)
{
    T f = OIIO::fast_cosh(a.val());
    T df = OIIO::fast_sinh(a.val());
    return dualfunc (a, f, df);
}


// f(x) = sinh(x), f'(x) = cosh(x)
template<class T, int P>
inline Dual<T,P> sinh (const Dual<T,P> &a)
{
    T f = std::sinh(a.val());
    T df = std::cosh(a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_sinh(const Dual<T,P> &a)
{
    T f = OIIO::fast_sinh(a.val());
    T df = OIIO::fast_cosh(a.val());
    return dualfunc (a, f, df);
}

// f(x) = tanh(x), f'(x) = sech^2(x)
template<class T, int P>
inline Dual<T,P> tanh (const Dual<T,P> &a)
{
    T tanha = std::tanh(a.val());
    T cosha = std::cosh(a.val());
    T sech2a = T(1)/(cosha*cosha);
    return dualfunc (a, tanha, sech2a);
}

template<class T, int P>
inline Dual<T,P> fast_tanh(const Dual<T,P> &a)
{
    T tanha = OIIO::fast_tanh(a.val());
    T cosha = OIIO::fast_cosh(a.val());
    T sech2a = T(1) / (cosha * cosha);
    return dualfunc (a, tanha, sech2a);
}

// f(x) = acos(x), f'(x) = -1/(sqrt(1 - x^2))
template<class T, int P>
inline Dual<T,P> safe_acos (const Dual<T,P> &a)
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
inline Dual<T,P> fast_acos(const Dual<T,P> &a)
{
    T f = OIIO::fast_acos(a.val());
    T df = fabsf(a.val()) < 1.0f ? -1.0f / sqrtf(1.0f - a.val() * a.val()) : 0.0f;
    return dualfunc (a, f, df);
}

// f(x) = asin(x), f'(x) = 1/(sqrt(1 - x^2))
template<class T, int P>
inline Dual<T,P> safe_asin (const Dual<T,P> &a)
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
inline Dual<T,P> fast_asin(const Dual<T,P> &a)
{
    T f = OIIO::fast_asin(a.val());
    T df = fabsf(a.val()) < 1.0f ? 1.0f / sqrtf(1.0f - a.val() * a.val()) : 0.0f;
    return dualfunc (a, f, df);
}


// f(x) = atan(x), f'(x) = 1/(1 + x^2)
template<class T, int P>
inline Dual<T,P> atan (const Dual<T,P> &a)
{
    T f = std::atan (a.val());
    T df = T(1) / (T(1) + a.val()*a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_atan(const Dual<T,P> &a)
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
inline Dual<T,P> atan2 (const Dual<T,P> &y, const Dual<T,P> &x)
{
    T atan2xy = std::atan2 (y.val(), x.val());
    T denom = (x.val() == T(0) && y.val() == T(0)) ? T(0) : T(1) / (x.val()*x.val() + y.val()*y.val());
    return dualfunc (y, x, atan2xy, -x.val()*denom, y.val()*denom);
}

template<class T, int P>
inline Dual<T,P> fast_atan2(const Dual<T,P> &y, const Dual<T,P> &x)
{
    T atan2xy = OIIO::fast_atan2(y.val(), x.val());
    T denom = (x.val() == 0 && y.val() == 0) ? 0.0f : 1.0f / (x.val() * x.val() + y.val() * y.val());
    return dualfunc (y, x, atan2xy, -x.val()*denom, y.val()*denom);
}


// to compute pow(u,v), we need the dual-form representation of
// the pow() operator.  In general, the dual-form of the primitive
// function 'g' is:
//   g(<u,u'>, <v,v'>) = < g(u,v), dgdu(u,v)u' + dgdv(u,v)v' >
//   (from http://en.wikipedia.org/wiki/Automatic_differentiation)
// so, pow(u,v) = < u^v, vu^(v-1) u' + log(u)u^v v' >
template<class T, int P>
inline Dual<T,P> safe_pow (const Dual<T,P> &u, const Dual<T,P> &v)
{
    // NOTE: this function won't return exactly the same as pow(x,y) because we
    // use the identity u^v=u * u^(v-1) which does not hold in all cases for our
    // "safe" variant (nor does it hold in general in floating point arithmetic).
    T powuvm1 = safe_pow(u.val(), v.val() - T(1));
    T powuv   = powuvm1 * u.val();
    T logu    = u.val() > 0 ? safe_log(u.val()) : T(0);
    return dualfunc (u, v, powuv, v.val()*powuvm1, logu*powuv);
}

template<class T, int P>
inline Dual<T,P> fast_safe_pow(const Dual<T,P> &u, const Dual<T,P> &v)
{
    // NOTE: same issue as above (fast_safe_pow does even more clamping)
    T powuvm1 = OIIO::fast_safe_pow (u.val(), v.val() - 1.0f);
    T powuv   = powuvm1 * u.val();
    T logu    = u.val() > 0 ? OIIO::fast_log(u.val()) : 0.0f;
    return dualfunc (u, v, powuv, v.val()*powuvm1, logu*powuv);
}

// f(x) = log(a), f'(x) = 1/x
// (log base e)
template<class T, int P>
inline Dual<T,P> safe_log (const Dual<T,P> &a)
{
    T f = safe_log(a.val());
    T df = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / a.val();
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_log(const Dual<T,P> &a)
{
    T f = OIIO::fast_log(a.val());
    T df = a.val() < std::numeric_limits<float>::min() ? 0.0f : 1.0f / a.val();
    return dualfunc (a, f, df);
}

// f(x) = log2(x), f'(x) = 1/(x*log2)
// (log base 2)
template<class T, int P>
inline Dual<T,P> safe_log2 (const Dual<T,P> &a)
{
    T f = safe_log2(a.val());
    T df = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / (a.val() * T(M_LN2));
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_log2(const Dual<T,P> &a)
{
    T f = OIIO::fast_log2(a.val());
    T aln2 = a.val() * float(M_LN2);
    T df = aln2 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln2;
    return dualfunc (a, f, df);
}

// f(x) = log10(x), f'(x) = 1/(x*log10)
// (log base 10)
template<class T, int P>
inline Dual<T,P> safe_log10 (const Dual<T,P> &a)
{
    T f = safe_log10(a.val());
    T df = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / (a.val() * T(M_LN10));
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_log10(const Dual<T,P> &a)
{
    T f  = OIIO::fast_log10(a.val());
    T aln10 = a.val() * float(M_LN10);
    T df  = aln10 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln10;
    return dualfunc (a, f, df);
}

// f(x) = e^x, f'(x) = e^x
template<class T, int P>
inline Dual<T,P> exp (const Dual<T,P> &a)
{
    T f = std::exp(a.val());
    return dualfunc (a, f, f);
}

template<class T, int P>
inline Dual<T,P> fast_exp(const Dual<T,P> &a)
{
    T f = OIIO::fast_exp(a.val());
    return dualfunc (a, f, f);
}

// f(x) = 2^x, f'(x) = (2^x)*log(2)
template<class T, int P>
inline Dual<T,P> exp2 (const Dual<T,P> &a)
{
    // FIXME: std::exp2 is only available in C++11
    T f = exp2f(float(a.val()));
    return dualfunc (a, f, f*T(M_LN2));
}

template<class T, int P>
inline Dual<T,P> fast_exp2(const Dual<T,P> &a)
{
    T f = OIIO::fast_exp2(float(a.val()));
    return dualfunc (a, f, f*T(M_LN2));
}


// f(x) = e^x - 1, f'(x) = e^x
template<class T, int P>
inline Dual<T,P> expm1 (const Dual<T,P> &a)
{
    // FIXME: std::expm1 is only available in C++11
    T f  = expm1f(float(a.val())); // float version!
    T df = std::exp  (a.val());
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_expm1(const Dual<T,P> &a)
{
    T f  = OIIO::fast_expm1(a.val());
    T df = OIIO::fast_exp  (a.val());
    return dualfunc (a, f, df);
}

// f(x) = erf(x), f'(x) = (2e^(-x^2))/sqrt(pi)
template<class T, int P>
inline Dual<T,P> erf (const Dual<T,P> &a)
{
    // FIXME: std::erf is only defined in C++11
    T f = erff (float(a.val())); // float version!
    const T two_over_sqrt_pi = T(1.128379167095512573896158903);
    T df = std::exp (-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_erf(const Dual<T,P> &a)
{
    T f = OIIO::fast_erf (float(a.val())); // float version!
    const T two_over_sqrt_pi = 1.128379167095512573896158903f;
    T df = OIIO::fast_exp(-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}

// f(x) = erfc(x), f'(x) = -(2e^(-x^2))/sqrt(pi)
template<class T, int P>
inline Dual<T,P> erfc (const Dual<T,P> &a)
{
    // FIXME: std::erfc is only defined in C++11
    T f = erfcf (float(a.val())); // float version!
    const T two_over_sqrt_pi = -T(1.128379167095512573896158903);
    T df = std::exp (-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}

template<class T, int P>
inline Dual<T,P> fast_erfc(const Dual<T,P> &a)
{
    T f = OIIO::fast_erfc (float(a.val())); // float version!
    const T two_over_sqrt_pi = -1.128379167095512573896158903f;
    T df = OIIO::fast_exp(-a.val() * a.val()) * two_over_sqrt_pi;
    return dualfunc (a, f, df);
}


// f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
template<class T, int P>
inline OIIO_CONSTEXPR14 Dual<T,P> sqrt (const Dual<T,P> &a)
{
    if (a.val() <= T(0))
        return Dual<T,P> (T(0));
    T f  = std::sqrt(a.val());
    T df = T(0.5) / f;
    return dualfunc (a, f, df);
}

// f(x) = 1/sqrt(x), f'(x) = -1/(2*x^(3/2))
template<class T, int P>
inline Dual<T,P> inversesqrt (const Dual<T,P> &a)
{
    // do we want to print an error message?
    if (a.val() <= T(0))
        return Dual<T,P> (T(0));
    T f  = T(1)/std::sqrt(a.val());
    T df = T(-0.5)*f/a.val();
    return dualfunc (a, f, df);
}

// (fx) = x*(1-a) + y*a, f'(x) = (1-a)x' + (y - x)*a' + a*y'
template<class T, int P>
inline Dual<T,P> mix (const Dual<T,P> &x, const Dual<T,P> &y, const Dual<T,P> &a)
{
    T mixval = x.val()*(T(1)-a.val()) + y.val()*a.val();
    return dualfunc (x, y, a, mixval, T(1)-a.val(), a.val(), y.val() - x.val());
}

template<class T, int P>
inline Dual<T,P> fabs (const Dual<T,P> &x)
{
    return x.val() >= T(0) ? x : -x;
}



inline float smoothstep(float e0, float e1, float x) { 
    if (x < e0) return 0.0f;
    else if (x >= e1) return 1.0f;
    else {
        float t = (x - e0)/(e1 - e0);
        return (3.0f-2.0f*t)*(t*t);
    }
}

// f(t) = (3-2t)t^2,   t = (x-e0)/(e1-e0)
template<class T, int P>
inline Dual<T,P> smoothstep (const Dual<T,P> &e0, const Dual<T,P> &e1, const Dual<T,P> &x)
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



// ceil(Dual) loses derivatives
template<class T, int P>
inline constexpr float ceil (const Dual<T,P> &x)
{
    return std::ceil(x.val());
}


// floor(Dual) loses derivatives
template<class T, int P>
inline constexpr float floor (const Dual<T,P> &x)
{
    return std::floor(x.val());
}


// floor, cast to an int.
template<class T, int P>
inline constexpr int
ifloor (const Dual<T,P> &x)
{
    return (int)floor(x);
}


OSL_NAMESPACE_EXIT
