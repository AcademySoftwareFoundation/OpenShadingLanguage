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

#include <OSL/oslversion.h>
#include <OpenImageIO/fmath.h>
OSL_NAMESPACE_ENTER


/// Dual numbers are used to represent values and their derivatives.
///
/// The method is summarized in:
///      Piponi, Dan, "Automatic Differentiation, C++ Templates,
///      and Photogrammetry," Journal of Graphics, GPU, and Game 
///      Tools (JGT), Vol 9, No 4, pp. 41-55 (2004).
///
/// See http://jgt.akpeters.com/papers/Piponi04/ for downloadable
/// software that attempts to be general to N dimensions of partial
/// derivatives.  But for OSL, we only are interested in partials with
/// respect to "x" and "y", and don't need to be more general, so for
/// speed and simplicity we are hard-coding our implementation to
/// dimensionality 2.  We also change some of his nomenclature.
///

template<class T>
class Dual2 {
public:
    /// Default ctr leaves everything uninitialized
    ///
    Dual2 () { }

    /// Construct a Dual from just a real value (derivs set to 0)
    ///
    Dual2 (const T &x) : m_val(x), m_dx(T(0.0)), m_dy(T(0.0)) { }

    template <class F>
    Dual2 (const Dual2<F> &x) : m_val(T(x.val())), m_dx(T(x.dx())), m_dy(T(x.dy())) { }

    /// Construct a Dual from a real and both infinitesimal.
    ///
    Dual2 (const T &x, const T &dx, const T &dy) 
        : m_val(x), m_dx(dx), m_dy(dy) { }

    void set (const T &x, const T &dx, const T &dy) {
        m_val = x;  m_dx = dx;  m_dy = dy;
    }

    /// Return the real value of *this.
    ///
    const T& val () const { return m_val; }
    T& val () { return m_val; }

    /// Return the partial derivative with respect to x
    ///
    const T& dx () const { return m_dx; }
    T& dx () { return m_dx; }

    /// Return the partial derivative with respect to y
    ///
    const T& dy () const { return m_dy; }
    T& dy () { return m_dy; }

    void set_val (const T &val) { m_val = val; }
    void set_dx  (const T &dx)  { m_dx  = dx;  }
    void set_dy  (const T &dy)  { m_dy  = dy;  }

    /// Clear the derivatives; leave the value alone.
    ///
    void clear_d () { m_dx = T(0);  m_dy = T(0); }

    /// Return the special dual number (i == 0 is the dx imaginary
    /// number, i == 1 is the dy imaginary number).
    static Dual2<T> d (int i) {
        return i==0 ? Dual2<T> (T(0),T(1),T(0)) : Dual2<T> (T(0),T(0),T(1));
    }

    const Dual2<T> & operator= (const T &x) {
        set (x, T(0), T(0));
        return *this;
    }

    /// Stream output.  Format as: "val[dx,dy]"
    ///
    friend std::ostream& operator<< (std::ostream &out, const Dual2<T> &x) {
        return out << x.val() << "[" << x.dx() << "," << x.dy() << "]";
    }

private:
    T m_val;   ///< The value
    T m_dx;    ///< Infinitesimal partial differential with respect to x
    T m_dy;    ///< Infinitesimal partial differential with respect to y
};




/// Addition of duals.
///
template<class T>
inline Dual2<T> operator+ (const Dual2<T> &a, const Dual2<T> &b)
{
    return Dual2<T> (a.val()+b.val(), a.dx()+b.dx(), a.dy()+b.dy());
}


template<class T>
inline Dual2<T> operator+ (const Dual2<T> &a, const T &b)
{
    return Dual2<T> (a.val()+b, a.dx(), a.dy());
}


template<class T>
inline Dual2<T> operator+ (const T &a, const Dual2<T> &b)
{
    return Dual2<T> (a+b.val(), b.dx(), b.dy());
}


template<class T>
inline Dual2<T>& operator+= (Dual2<T> &a, const Dual2<T> &b)
{
    a.val() += b.val();
    a.dx()  += b.dx();
    a.dy()  += b.dy();
    return a;
}


template<class T>
inline Dual2<T>& operator+= (Dual2<T> &a, const T &b)
{
    a.val() += b;
    return a;
}


/// Subtraction of duals.
///
template<class T>
inline Dual2<T> operator- (const Dual2<T> &a, const Dual2<T> &b)
{
    return Dual2<T> (a.val()-b.val(), a.dx()-b.dx(), a.dy()-b.dy());
}


template<class T>
inline Dual2<T> operator- (const Dual2<T> &a, const T &b)
{
    return Dual2<T> (a.val()-b, a.dx(), a.dy());
}


template<class T>
inline Dual2<T> operator- (const T &a, const Dual2<T> &b)
{
    return Dual2<T> (a-b.val(), -b.dx(), -b.dy());
}


template<class T>
inline Dual2<T>& operator-= (Dual2<T> &a, const Dual2<T> &b)
{
    a.val() -= b.val();
    a.dx()  -= b.dx();
    a.dy()  -= b.dy();
    return a;
}


template<class T>
inline Dual2<T>& operator-= (Dual2<T> &a, const T &b)
{
    a.val() -= b.val();
    return a;
}



/// Negation of duals.
///
template<class T>
inline Dual2<T> operator- (const Dual2<T> &a)
{
    return Dual2<T> (-a.val(), -a.dx(), -a.dy());
}


/// Multiplication of duals.
///
template<class T>
inline Dual2<T> operator* (const Dual2<T> &a, const Dual2<T> &b)
{
    // Use the chain rule
    return Dual2<T> (a.val()*b.val(),
                     a.val()*b.dx() + a.dx()*b.val(),
                     a.val()*b.dy() + a.dy()*b.val());
}


/// Multiplication of dual by scalar.
///
template<class T>
inline Dual2<T> operator* (const Dual2<T> &a, const T &b)
{
    return Dual2<T> (a.val()*b, a.dx()*b, a.dy()*b);
}


/// Multiplication of dual by scalar in place.
///
template<class T>
inline const Dual2<T>& operator*= (Dual2<T> &a, const T &b)
{
    a.val() *= b;
    a.dx()  *= b;
    a.dy()  *= b;
    return a;
}



/// Multiplication of dual by scalar.
///
template<class T>
inline Dual2<T> operator* (const T &b, const Dual2<T> &a)
{
    return Dual2<T> (a.val()*b, a.dx()*b, a.dy()*b);
}

template<class T, class S>
inline Dual2<T> operator* (const S &b, const Dual2<T> &a)
{
    return Dual2<T> (a.val()*T(b), a.dx()*T(b), a.dy()*T(b));
}



/// Division of duals.
///
template<class T>
inline Dual2<T> operator/ (const Dual2<T> &a, const Dual2<T> &b)
{
    T bvalinv = 1.0f / b.val();
    T aval_bval = a.val() * bvalinv;
    return Dual2<T> (aval_bval,
                     bvalinv * (a.dx() - aval_bval * b.dx()),
                     bvalinv * (a.dy() - aval_bval * b.dy()));
}


/// Division of dual by scalar.
///
template<class T>
inline Dual2<T> operator/ (const Dual2<T> &a, const T &b)
{
    T binv = 1.0f / b;
    return a * binv;
}


/// Division of scalar by dual.
///
template<class T>
inline Dual2<T> operator/ (const T &aval, const Dual2<T> &b)
{
    T bvalinv = 1.0f / b.val();
    T aval_bval = aval * bvalinv;
    return Dual2<T> (aval_bval,
                     bvalinv * ( - aval_bval * b.dx()),
                     bvalinv * ( - aval_bval * b.dy()));
}




template<class T>
inline bool operator< (const Dual2<T> &a, const Dual2<T> &b) {
    return a.val() < b.val();
}

template<class T>
inline bool operator< (const Dual2<T> &a, const T &b) {
    return a.val() < b;
}

template<class T>
inline bool operator> (const Dual2<T> &a, const Dual2<T> &b) {
    return a.val() > b.val();
}

template<class T>
inline bool operator> (const Dual2<T> &a, const T &b) {
    return a.val() > b;
}

template<class T>
inline bool operator<= (const Dual2<T> &a, const Dual2<T> &b) {
    return a.val() <= b.val();
}

template<class T>
inline bool operator<= (const Dual2<T> &a, const T &b) {
    return a.val() <= b;
}

template<class T>
inline bool operator>= (const Dual2<T> &a, const Dual2<T> &b) {
    return a.val() >= b.val();
}

template<class T>
inline bool operator>= (const Dual2<T> &a, const T &b) {
    return a.val() >= b;
}



// Eliminate the derivatives of a number
template<class T> inline T removeDerivatives (const T &x)        { return x;       }
template<class T> inline T removeDerivatives (const Dual2<T> &x) { return x.val(); }

// Get the x derivative (or 0 for a non-Dual)
template<class T> inline T getXDerivative (const T &x)        { return T(0);   }
template<class T> inline T getXDerivative (const Dual2<T> &x) { return x.dx(); }

// Get the y derivative (or 0 for a non-Dual)
template<class T> inline T getYDerivative (const T &x)        { return T(0);   }
template<class T> inline T getYDerivative (const Dual2<T> &x) { return x.dy(); }

// Simple templated "copy" function
template <class T> inline void assignment(T &a, T &b)        { a = b;       }
template <class T> inline void assignment(T &a, Dual2<T> &b) { a = b.val(); }

// Templated value equality. For scalars, it's the same as regular ==.
// For Dual2's, this only tests the value, not the derivatives. This solves
// a pesky source of confusion about whether operator== of Duals ought to
// return if just their value is equal or if the whole struct (including
// derivs) are equal.
template<class T>
inline constexpr bool equalVal (const T &x, const T &y) {
    return x == y;
}
template<class T>
inline constexpr bool equalVal (const Dual2<T> &x, const Dual2<T> &y) {
    return x.val() == y.val();
}



// f(x) = cos(x), f'(x) = -sin(x)
template<class T>
inline Dual2<T> cos (const Dual2<T> &a)
{
    T sina, cosa;
    OIIO::sincos(a.val(), &sina, &cosa);
    return Dual2<T> (cosa, -sina * a.dx(), -sina * a.dy());
}

inline Dual2<float> fast_cos(const Dual2<float> &a)
{
    float sina, cosa;
    OIIO::fast_sincos (a.val(), &sina, &cosa);
    return Dual2<float> (cosa, -sina * a.dx(), -sina * a.dy());
}

// f(x) = sin(x),  f'(x) = cos(x)
template<class T>
inline Dual2<T> sin (const Dual2<T> &a)
{
    T sina, cosa;
    OIIO::sincos(a.val(), &sina, &cosa);
    return Dual2<T> (sina, cosa * a.dx(), cosa * a.dy());
}

inline Dual2<float> fast_sin(const Dual2<float> &a)
{
    float sina, cosa;
    OIIO::fast_sincos (a.val(), &sina, &cosa);
    return Dual2<float> (sina, cosa * a.dx(), cosa * a.dy());
}

template <class T>
inline void sincos(const Dual2<T> &a, Dual2<T> *sine, Dual2<T> *cosine)
{
	T sina, cosa;
	OIIO::sincos(a.val(), &sina, &cosa);
	*cosine = Dual2<T> (cosa, -sina * a.dx(), -sina * a.dy());
	  *sine = Dual2<T> (sina,  cosa * a.dx(),  cosa * a.dy());
}

inline void fast_sincos(const Dual2<float> &a, Dual2<float> *sine, Dual2<float> *cosine)
{
	float sina, cosa;
	OIIO::fast_sincos(a.val(), &sina, &cosa);
	*cosine = Dual2<float> (cosa, -sina * a.dx(), -sina * a.dy());
	  *sine = Dual2<float> (sina,  cosa * a.dx(),  cosa * a.dy());
}

// f(x) = tan(x), f'(x) = sec^2(x)
template<class T>
inline Dual2<T> tan (const Dual2<T> &a)
{
    T tana  = std::tan (a.val());
    T cosa  = std::cos (a.val());
    T sec2a = T(1)/(cosa*cosa);
    return Dual2<T> (tana, sec2a * a.dx(), sec2a * a.dy());
}

inline Dual2<float> fast_tan(const Dual2<float> &a)
{
    float tana  = OIIO::fast_tan (a.val());
    float cosa  = OIIO::fast_cos (a.val());
    float sec2a = 1 / (cosa * cosa);
    return Dual2<float> (tana, sec2a * a.dx(), sec2a * a.dy());
}

// f(x) = cosh(x), f'(x) = sinh(x)
template<class T>
inline Dual2<T> cosh (const Dual2<T> &a)
{
    T cosha = std::cosh(a.val());
    T sinha = std::sinh(a.val());
    return Dual2<T> (cosha, sinha * a.dx(), sinha * a.dy());
}

inline Dual2<float> fast_cosh(const Dual2<float> &a)
{
    float cosha = OIIO::fast_cosh(a.val());
    float sinha = OIIO::fast_sinh(a.val());
    return Dual2<float> (cosha, sinha * a.dx(), sinha * a.dy());
}


// f(x) = sinh(x), f'(x) = cosh(x)
template<class T>
inline Dual2<T> sinh (const Dual2<T> &a)
{
    T cosha = std::cosh(a.val());
    T sinha = std::sinh(a.val());
    return Dual2<T> (sinha, cosha * a.dx(), cosha * a.dy());
}

inline Dual2<float> fast_sinh(const Dual2<float> &a)
{
    float cosha = OIIO::fast_cosh(a.val());
    float sinha = OIIO::fast_sinh(a.val());
    return Dual2<float> (sinha, cosha * a.dx(), cosha * a.dy());
}

// f(x) = tanh(x), f'(x) = sech^2(x)
template<class T>
inline Dual2<T> tanh (const Dual2<T> &a)
{
    T tanha = std::tanh(a.val());
    T cosha = std::cosh(a.val());
    T sech2a = T(1)/(cosha*cosha);
    return Dual2<T> (tanha, sech2a * a.dx(), sech2a * a.dy());
}

inline Dual2<float> fast_tanh(const Dual2<float> &a)
{
    float tanha = OIIO::fast_tanh(a.val());
    float cosha = OIIO::fast_cosh(a.val());
    float sech2a = 1 / (cosha * cosha);
    return Dual2<float> (tanha, sech2a * a.dx(), sech2a * a.dy());
}

// f(x) = acos(x), f'(x) = -1/(sqrt(1 - x^2))
template<class T>
inline Dual2<T> safe_acos (const Dual2<T> &a)
{
    if (a.val() >= T(1))
        return Dual2<T> (T(0), T(0), T(0));
    if (a.val() <= T(-1))
        return Dual2<T> (T(M_PI), T(0), T(0));
    T arccosa = std::acos (a.val());
    T denom   = -T(1) / std::sqrt (T(1) - a.val()*a.val());
    return Dual2<T> (arccosa, denom * a.dx(), denom * a.dy());
}

inline Dual2<float> fast_acos(const Dual2<float> &a)
{
    float arccosa = OIIO::fast_acos(a.val());
    float denom   = fabsf(a.val()) < 1.0f ? -1.0f / sqrtf(1.0f - a.val() * a.val()) : 0.0f;
    return Dual2<float> (arccosa, denom * a.dx(), denom * a.dy());
}

// f(x) = asin(x), f'(x) = 1/(sqrt(1 - x^2))
template<class T>
inline Dual2<T> safe_asin (const Dual2<T> &a)
{
    if (a.val() >= T(1))
        return Dual2<T> (T(M_PI/2), T(0), T(0));
    if (a.val() <= T(-1))
        return Dual2<T> (T(-M_PI/2), T(0), T(0));

    T arcsina = std::asin (a.val());
    T denom   = T(1) / std::sqrt (T(1) - a.val()*a.val());
    return Dual2<T> (arcsina, denom * a.dx(), denom * a.dy());
}

inline Dual2<float> fast_asin(const Dual2<float> &a)
{
    float arcsina = OIIO::fast_asin(a.val());
    float denom   = fabsf(a.val()) < 1.0f ? 1.0f / sqrtf(1.0f - a.val() * a.val()) : 0.0f;
    return Dual2<float> (arcsina, denom * a.dx(), denom * a.dy());
}


// f(x) = atan(x), f'(x) = 1/(1 + x^2)
template<class T>
inline Dual2<T> atan (const Dual2<T> &a)
{
    T arctana = std::atan (a.val());
    T denom   = T(1) / (T(1) + a.val()*a.val());
    return Dual2<T> (arctana, denom * a.dx(), denom * a.dy());
}

inline Dual2<float> fast_atan(const Dual2<float> &a)
{
    float arctana = OIIO::fast_atan(a.val());
    float denom   = 1.0f / (1.0f + a.val() * a.val());
    return Dual2<float> (arctana, denom * a.dx(), denom * a.dy());

}

// f(x,x) = atan2(y,x); f'(x) =  y x' / (x^2 + y^2),
//                      f'(y) = -x y' / (x^2 + y^2)
// reference:  http://en.wikipedia.org/wiki/Atan2 
// (above link has other formulations)
template<class T>
inline Dual2<T> atan2 (const Dual2<T> &y, const Dual2<T> &x)
{
    T atan2xy = std::atan2 (y.val(), x.val());
    T denom = (x.val() == T(0) && y.val() == T(0)) ? T(0) : T(1) / (x.val()*x.val() + y.val()*y.val());
    return Dual2<T> ( atan2xy, (y.val()*x.dx() - x.val()*y.dx())*denom,
                               (y.val()*x.dy() - x.val()*y.dy())*denom );
}

inline Dual2<float> fast_atan2(const Dual2<float> &y, const Dual2<float> &x)
{
    float atan2xy = OIIO::fast_atan2(y.val(), x.val());
    float denom = (x.val() == 0 && y.val() == 0) ? 0.0f : 1.0f / (x.val() * x.val() + y.val() * y.val());
    return Dual2<float> ( atan2xy, (y.val()*x.dx() - x.val()*y.dx())*denom,
                                   (y.val()*x.dy() - x.val()*y.dy())*denom );
}


// to compute pow(u,v), we need the dual-form representation of
// the pow() operator.  In general, the dual-form of the primitive 
// function 'g' is:
//   g(<u,u'>, <v,v'>) = < g(u,v), dgdu(u,v)u' + dgdv(u,v)v' >
//   (from http://en.wikipedia.org/wiki/Automatic_differentiation)
// so, pow(u,v) = < u^v, vu^(v-1) u' + log(u)u^v v' >
template<class T>
inline Dual2<T> safe_pow (const Dual2<T> &u, const Dual2<T> &v)
{
    // NOTE: this function won't return exactly the same as pow(x,y) because we
    // use the identity u^v=u * u^(v-1) which does not hold in all cases for our
    // "safe" variant (nor does it hold in general in floating point arithmetic).
    T powuvm1 = safe_pow(u.val(), v.val() - T(1));
    T powuv   = powuvm1 * u.val();
    T logu    = u.val() > 0 ? safe_log(u.val()) : T(0);
    return Dual2<T> ( powuv, v.val()*powuvm1 * u.dx() + logu*powuv * v.dx(),
                             v.val()*powuvm1 * u.dy() + logu*powuv * v.dy() );
}

inline Dual2<float> fast_safe_pow(const Dual2<float> &u, const Dual2<float> &v)
{
    // NOTE: same issue as above (fast_safe_pow does even more clamping)
    float powuvm1 = OIIO::fast_safe_pow (u.val(), v.val() - 1.0f);
    float powuv   = powuvm1 * u.val();
    float logu    = u.val() > 0 ? OIIO::fast_log(u.val()) : 0.0f;
    return Dual2<float> ( powuv, v.val()*powuvm1 * u.dx() + logu*powuv * v.dx(),
                                 v.val()*powuvm1 * u.dy() + logu*powuv * v.dy() );

}

// f(x) = log(a), f'(x) = 1/x
// (log base e)
template<class T>
inline Dual2<T> safe_log (const Dual2<T> &a)
{
    T loga = safe_log(a.val());
    T inva = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / a.val();
    return Dual2<T> (loga, inva * a.dx(), inva * a.dy());
}

inline Dual2<float> fast_log(const Dual2<float> &a)
{
    float loga = OIIO::fast_log(a.val());
    float inva = a.val() < std::numeric_limits<float>::min() ? 0.0f : 1.0f / a.val();
    return Dual2<float> (loga, inva * a.dx(), inva * a.dy());
}

// f(x) = log2(x), f'(x) = 1/(x*log2)
// (log base 2)
template<class T>
inline Dual2<T> safe_log2 (const Dual2<T> &a)
{
    T loga = safe_log2(a.val());
    T inva = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / (a.val() * T(M_LN2));
    return Dual2<T> (loga, inva * a.dx(), inva * a.dy());
}

inline Dual2<float> fast_log2(const Dual2<float> &a)
{
    float loga = OIIO::fast_log2(a.val());
    float aln2 = a.val() * float(M_LN2);
    float inva = aln2 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln2;
    return Dual2<float> (loga, inva * a.dx(), inva * a.dy());
}

// f(x) = log10(x), f'(x) = 1/(x*log10)
// (log base 10)
template<class T>
inline Dual2<T> safe_log10 (const Dual2<T> &a)
{
    T loga = safe_log10(a.val());
    T inva = a.val() < std::numeric_limits<T>::min() ? T(0) : T(1) / (a.val() * T(M_LN10));
    return Dual2<T> (loga, inva * a.dx(), inva * a.dy());
}

inline Dual2<float> fast_log10(const Dual2<float> &a)
{
    float loga  = OIIO::fast_log10(a.val());
    float aln10 = a.val() * float(M_LN10);
    float inva  = aln10 < std::numeric_limits<float>::min() ? 0.0f : 1.0f / aln10;
    return Dual2<float> (loga, inva * a.dx(), inva * a.dy());
}

// f(x) = e^x, f'(x) = e^x
template<class T>
inline Dual2<T> exp (const Dual2<T> &a)
{
    T expa = std::exp(a.val());
    return Dual2<T> (expa, expa * a.dx(), expa * a.dy());
}

inline Dual2<float> fast_exp(const Dual2<float> &a)
{
    float expa = OIIO::fast_exp(a.val());
    return Dual2<float> (expa, expa * a.dx(), expa * a.dy());
}

// f(x) = 2^x, f'(x) = (2^x)*log(2)
template<class T>
inline Dual2<T> exp2 (const Dual2<T> &a)
{
    // FIXME: std::exp2 is only available in C++11
    T exp2a = exp2f(float(a.val()));
    return Dual2<T> (exp2a, exp2a*T(M_LN2)*a.dx(), exp2a*T(M_LN2)*a.dy());
}

inline Dual2<float> fast_exp2(const Dual2<float> &a)
{
    float exp2a = OIIO::fast_exp2(float(a.val()));
    return Dual2<float> (exp2a, exp2a*float(M_LN2)*a.dx(), exp2a*float(M_LN2)*a.dy());
}


// f(x) = e^x - 1, f'(x) = e^x
template<class T>
inline Dual2<T> expm1 (const Dual2<T> &a)
{
    // FIXME: std::expm1 is only available in C++11
    T expm1a = expm1f(float(a.val())); // float version!
    T expa   = std::exp  (a.val());
    return Dual2<T> (expm1a, expa * a.dx(), expa * a.dy());
}

inline Dual2<float> fast_expm1(const Dual2<float> &a)
{
    float expm1a = OIIO::fast_expm1(a.val());
    float expa   = OIIO::fast_exp  (a.val());
    return Dual2<float> (expm1a, expa * a.dx(), expa * a.dy());
}

// f(x) = erf(x), f'(x) = (2e^(-x^2))/sqrt(pi)
template<class T>
inline Dual2<T> erf (const Dual2<T> &a)
{
    // FIXME: std::erf is only defined in C++11
    T erfa = erff (float(a.val())); // float version!
    T two_over_sqrt_pi = T(1.128379167095512573896158903);
    T derfadx = std::exp (-a.val() * a.val()) * two_over_sqrt_pi;
    return Dual2<T> (erfa, derfadx * a.dx(), derfadx * a.dy());
}

inline Dual2<float> fast_erf(const Dual2<float> &a)
{
    float erfa = OIIO::fast_erf (float(a.val())); // float version!
    float two_over_sqrt_pi = 1.128379167095512573896158903f;
    float derfadx = OIIO::fast_exp(-a.val() * a.val()) * two_over_sqrt_pi;
    return Dual2<float> (erfa, derfadx * a.dx(), derfadx * a.dy());
}

// f(x) = erfc(x), f'(x) = -(2e^(-x^2))/sqrt(pi)
template<class T>
inline Dual2<T> erfc (const Dual2<T> &a)
{
    // FIXME: std::erfc is only defined in C++11
    T erfca = erfcf (float(a.val())); // float version!
    T two_over_sqrt_pi = -T(1.128379167095512573896158903);
    T derfcadx = std::exp (-a.val() * a.val()) * two_over_sqrt_pi;
    return Dual2<T> (erfca, derfcadx * a.dx(), derfcadx * a.dy());
}

inline Dual2<float> fast_erfc(const Dual2<float> &a)
{
    float erfa = OIIO::fast_erfc (float(a.val())); // float version!
    float two_over_sqrt_pi = -1.128379167095512573896158903f;
    float derfadx = OIIO::fast_exp(-a.val() * a.val()) * two_over_sqrt_pi;
    return Dual2<float> (erfa, derfadx * a.dx(), derfadx * a.dy());
}


// f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
template<class T>
inline Dual2<T> sqrt (const Dual2<T> &a)
{
    if (a.val() <= T(0))
        return Dual2<T> (T(0), T(0), T(0));

    T sqrta      = std::sqrt(a.val());
    T inv_2sqrta = T(1) / (T(2) * sqrta);

    return Dual2<T> (sqrta, inv_2sqrta * a.dx(), inv_2sqrta * a.dy());
}

// f(x) = 1/sqrt(x), f'(x) = -1/(2*x^(3/2))
template<class T>
inline Dual2<T> inversesqrt (const Dual2<T> &a)
{
    // do we want to print an error message?
    if (a.val() <= T(0))
        return Dual2<T> (T(0), T(0), T(0));

    T sqrta          = std::sqrt(a.val());
    T inv_neg2asqrta = -T(1)/(T(2)*a.val()*sqrta);

    return Dual2<T> (T(1)/sqrta, inv_neg2asqrta * a.dx(), inv_neg2asqrta * a.dy());
}

// (fx) = x*(1-a) + y*a, f'(x) = (1-a)x' + (y - x)*a' + a*y'
template<class T>
inline Dual2<T> mix (const Dual2<T> &x, const Dual2<T> &y, const Dual2<T> &a)
{
   T mixval = x.val()*(T(1)-a.val()) + y.val()*a.val();

   return Dual2<T> (mixval, (T(1) - a.val())*x.dx() + (y.val() - x.val())*a.dx() + a.val()*y.dx(),
                            (T(1) - a.val())*x.dy() + (y.val() - x.val())*a.dy() + a.val()*y.dy());
}

template<class T>
inline Dual2<T> dual_min (const Dual2<T> &x, const Dual2<T> &y)
{
   if (x.val() > y.val())
      return y;
   else 
      return x;
}

template<class T>
inline Dual2<T> dual_max (const Dual2<T> &x, const Dual2<T> &y)
{
   if (x.val() > y.val())
      return x;
   else 
      return y;
}


template<class T>
inline Dual2<T> fabs (const Dual2<T> &x)
{
    return x.val() >= T(0) ? x : -x;
}



template<class T>
inline Dual2<T> dual_clamp (const Dual2<T> &x, const Dual2<T> &minv, const Dual2<T> &maxv)
{
   if (x.val() < minv.val()) return minv;
   else if (x.val() > maxv.val()) return maxv;
   else return x;
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
template<class T>
inline Dual2<T> smoothstep (const Dual2<T> &e0, const Dual2<T> &e1, const Dual2<T> &x)
{
   if (x.val() < e0.val()) {
      return Dual2<T> (T(0), T(0), T(0));
   }
   else if (x.val() >= e1.val()) {
      return Dual2<T> (T(1), T(0), T(0));
   }
   Dual2<T> t = (x - e0)/(e1-e0);
   return  (T(3) - T(2)*t)*t*t;
}



// floor(Dual) loses derivatives
template<class T>
inline float floor (const Dual2<T> &x)
{
    return std::floor(x.val());
}


// floor, cast to an int.
template<class T>
inline int
ifloor (const Dual2<T> &x)
{
    return (int)floor(x);
}


OSL_NAMESPACE_EXIT
