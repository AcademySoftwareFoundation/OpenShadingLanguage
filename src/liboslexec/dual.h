/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#ifndef DUAL_H
#define DUAL_H

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


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
    Dual2 (const T &x) : m_val(x), m_dx(T(0)), m_dy(T(0)) { }

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

    /// Return the special dual number (i == 0 is the dx imaginary
    /// number, i == 1 is the dy imaginary number).
    static Dual2<T> d (int i) {
        return i==0 ? Dual2<T> (T(0),T(1),T(0)) : Dual2<T> (T(0),T(0),T(1));
    }

    const Dual2<T> & operator= (const T &x) {
        init (x, T(0), T(0));
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
Dual2<T> operator+ (const Dual2<T> &a, const Dual2<T> &b)
{
    return Dual2<T> (a.val()+b.val(), a.dx()+b.dx(), a.dy()+b.dy());
}


template<class T>
Dual2<T> operator+ (const Dual2<T> &a, const T &b)
{
    return Dual2<T> (a.val()+b, a.dx(), a.dy());
}


template<class T>
Dual2<T> operator+ (const T &a, const Dual2<T> &b)
{
    return Dual2<T> (a+b.val(), b.dx(), b.dy());
}


/// Subtraction of duals.
///
template<class T>
Dual2<T> operator- (const Dual2<T> &a, const Dual2<T> &b)
{
    return Dual2<T> (a.val()-b.val(), a.dx()-b.dx(), a.dy()-b.dy());
}


template<class T>
Dual2<T> operator- (const Dual2<T> &a, const T &b)
{
    return Dual2<T> (a.val()-b, a.dx(), a.dy());
}


template<class T>
Dual2<T> operator- (const T &a, const Dual2<T> &b)
{
    return Dual2<T> (a-b.val(), -b.dx(), -b.dy());
}



/// Negation of duals.
///
template<class T>
Dual2<T> operator- (const Dual2<T> &a)
{
    return Dual2<T> (-a.val(), -a.dx(), -a.dy());
}


/// Multiplication of duals.
///
template<class T>
Dual2<T> operator* (const Dual2<T> &a, const Dual2<T> &b)
{
    // Use the chain rule
    return Dual2<T> (a.val()*b.val(),
                     a.val()*b.dx() + a.dx()*b.val(),
                     a.val()*b.dy() + a.dy()*b.val());
}


/// Multiplication of dual by scalar.
///
template<class T>
Dual2<T> operator* (const Dual2<T> &a, const T &b)
{
    return Dual2<T> (a.val()*b, a.dx()*b, a.dy()*b);
}


/// Multiplication of dual by scalar.
///
template<class T>
Dual2<T> operator* (const T &b, const Dual2<T> &a)
{
    return Dual2<T> (a.val()*b, a.dx()*b, a.dy()*b);
}



/// Division of duals.
///
template<class T>
Dual2<T> operator/ (const Dual2<T> &a, const Dual2<T> &b)
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
Dual2<T> operator/ (const Dual2<T> &a, const T &b)
{
    T binv = 1.0f / b;
    return a * binv;
}


/// Division of scalar by dual.
///
template<class T>
Dual2<T> operator/ (const T &aval, const Dual2<T> &b)
{
    T bvalinv = 1.0f / b.val();
    T aval_bval = aval * bvalinv;
    return Dual2<T> (aval_bval,
                     bvalinv * ( - aval_bval * b.dx()),
                     bvalinv * ( - aval_bval * b.dy()));
}

// f(x) = cos(x), f'(x) = -sin(x)
template<class T>
Dual2<T> cos (const Dual2<T> &a)
{
    T sina, cosa;
    sina = std::sin (a.val());
    cosa = std::cos (a.val());
    return Dual2<T> (cosa, -sina * a.dx(), -sina * a.dy());
}

// f(x) = sin(x),  f'(x) = cos(x)
template<class T>
Dual2<T> sin (const Dual2<T> &a)
{
    T sina, cosa;
    sina = std::sin (a.val());
    cosa = std::cos (a.val());
    return Dual2<T> (sina, cosa * a.dx(), cosa * a.dy());
}

// f(x) = tan(x), f'(x) = sec^2(x)
template<class T>
Dual2<T> tan (const Dual2<T> &a)
{
   T tana, cosa, sec2a;
   tana  = std::tan (a.val());
   cosa  = std::cos (a.val());
   sec2a = T(1)/(cosa*cosa);
   return Dual2<T> (tana, sec2a * a.dx(), sec2a * a.dy());
}

// f(x) = cosh(x), f'(x) = sinh(x)
template<class T>
Dual2<T> cosh (const Dual2<T> &a)
{
   T cosha, sinha;
   cosha = std::cosh(a.val());
   sinha = std::sinh(a.val());
   return Dual2<T> (cosha, sinha * a.dx(), sinha * a.dy());
}

// f(x) = sinh(x), f'(x) = cosh(x)
template<class T>
Dual2<T> sinh (const Dual2<T> &a)
{
   T cosha, sinha;
   cosha = std::cosh(a.val());
   sinha = std::sinh(a.val());
   return Dual2<T> (sinha, cosha * a.dx(), cosha * a.dy());
}

// f(x) = tanh(x), f'(x) = sech^2(x)
template<class T>
Dual2<T> tanh (const Dual2<T> &a)
{
   T cosha, tanha, sech2a;
   tanha = std::tanh(a.val());
   cosha = std::cosh(a.val());
   sech2a = T(1)/(cosha*cosha);
   return Dual2<T> (tanha, sech2a * a.dx(), sech2a * a.dy());
}

// f(x) = acos(x), f'(x) = -1/(sqrt(1 - x^2))
template<class T>
Dual2<T> acos (const Dual2<T> &a)
{
   if (a.val() >= T(1)) 
      return Dual2<T> (T(0), T(0), T(0));
   if (a.val() <= T(-1)) 
      return Dual2<T> (T(M_PI), T(0), T(0));

   T arccosa, denom;
   arccosa = std::acos (a.val());
   denom   = -T(1) / std::sqrt (T(1) - a.val()*a.val());

   return Dual2<T> (arccosa, denom * a.dx(), denom * a.dy());
}

// f(x) = asin(x), f'(x) = 1/(sqrt(1 - x^2))
template<class T>
Dual2<T> asin (const Dual2<T> &a)
{
   if (a.val() >= T(1)) 
      return Dual2<T> (T(M_PI/2), T(0), T(0));
   if (a.val() <= T(-1)) 
      return Dual2<T> (T(-M_PI/2), T(0), T(0));

   T arcsina, denom;
   arcsina = std::asin (a.val());
   denom   = T(1) / std::sqrt (T(1) - a.val()*a.val());

   return Dual2<T> (arcsina, denom * a.dx(), denom * a.dy());
}

// f(x) = atan(x), f'(x) = 1/(1 + x^2)
template<class T>
Dual2<T> atan (const Dual2<T> &a)
{
   T arctana, denom;
   arctana = std::atan (a.val());
   denom   = T(1) / (T(1) + a.val()*a.val());

   return Dual2<T> (arctana, denom * a.dx(), denom * a.dy());
}


// f(x,x) = atan2(y,x); f'(x) =  y x' / (x^2 + y^2),
//                      f'(y) = -x y' / (x^2 + y^2)
// reference:  http://en.wikipedia.org/wiki/Atan2 
// (above link has other formulations)
template<class T>
Dual2<T> atan2 (const Dual2<T> &y, const Dual2<T> &x)
{
   if (y.val() == T(0) && x.val() == T(0))
      return Dual2<T> (T(0), T(0), T(0));

   T atan2xy;
   atan2xy = std::atan2 (y.val(), x.val());
   T denom = T(1) / (x.val()*x.val() + y.val()*y.val());
   return Dual2<T> ( atan2xy, (y.val()*x.dx() - x.val()*y.dx())*denom,
                              (y.val()*x.dy() - x.val()*y.dy())*denom );
}


// to compute pow(u,v), we need the dual-form representation of
// the pow() operator.  In general, the dual-form of the primitive 
// function 'g' is:
//   g(<u,u'>, <v,v'>) = < g(u,v), dgdu(u,v)u' + dgdv(u,v)v' >
//   (from http://en.wikipedia.org/wiki/Automatic_differentiation)
// so, pow(u,v) = < u^v, vu^(v-1) u' + log(u)u^v v' >
template<class T>
Dual2<T> pow (const Dual2<T> &u, const Dual2<T> &v)
{
#define MYTRUNC(x)  ((x < T(0)) ? std::ceil(x) : std::floor(x))
   // assume 0^x is always zero unless x==0
   if (u.val() == T(0))
   {
      if (v.val() == T(0))
         return Dual2<T> ( T(1), T(0), T(0) );
      else
         return Dual2<T> ( T(0), T(0), T(0) );
   }
   // return 0 instead of an imaginary number
   if (u.val() < T(0) && ( MYTRUNC (v.val()) != v.val()) )
      return Dual2<T> ( T(0), T(0), T(0) );
   T powuv   = std::pow (u.val(), v.val());
   T powuvm1 = powuv / u.val(); // u^(v-1)
   T logu    = std::log (u.val());
   return Dual2<T> ( powuv, v.val()*powuvm1 * u.dx() + logu*powuv * v.dx(),
                            v.val()*powuvm1 * u.dy() + logu*powuv * v.dy() );
#undef MYTRUNC
}


// f(x) = log(a), f'(x) = 1/x
// (log base e)
template<class T>
Dual2<T> log (const Dual2<T> &a)
{
   // do we want to print an error message?
   if (a.val() <= T(0))
      return Dual2<T> (T(-std::numeric_limits<T>::max()), T(0), T(0));

   T loga, inv_a;
   loga  = std::log (a.val());
   inv_a = T(1)/a.val();

   return Dual2<T> (loga, inv_a * a.dx(), inv_a * a.dy());
}

// f(x) = log(a)/log(b)  -- leverage Dual2<T>log() and Dua2<T>operator/()
// (log base e)
template<class T>
Dual2<T> log (const Dual2<T> &a, const Dual2<T> &b)
{
   // do we want to print an error message?
   if (a.val() <= T(0) || b.val() <= T(0) || b.val() == T(1))
      if (b.val() == T(1))
         return Dual2<T> (T(std::numeric_limits<T>::max()), T(0), T(0));
      else
         return Dual2<T> (T(-std::numeric_limits<T>::max()), T(0), T(0));

   Dual2<T> loga, logb;
   loga  = log (a);
   logb  = log (b);

   return loga/logb;
}

// f(x) = log2(x), f'(x) = 1/(x*log2)
// (log base 2)
template<class T>
Dual2<T> log2 (const Dual2<T> &a)
{
   // do we want to print an error message?
   if (a.val() <= T(0))
      return Dual2<T> (T(-std::numeric_limits<T>::max()), T(0), T(0));

   T log2, log2a, inv_a_log2;

   log2       = std::log (T(2));
   log2a      = std::log (a.val()) / log2;
   inv_a_log2 = T(1)/(a.val() * log2);

   return Dual2<T> (log2a, inv_a_log2 * a.dx(), inv_a_log2 * a.dy());
}

// f(x) = log10(x), f'(x) = 1/(x*log10)
// (log base 10)
template<class T>
Dual2<T> log10 (const Dual2<T> &a)
{
   // do we want to print an error message?
   if (a.val() <= T(0))
      return Dual2<T> (T(-std::numeric_limits<T>::max()), T(0), T(0));

   T log10, log10a, inv_a_log10;
   log10       = std::log (T(10));
   log10a      = std::log10 (a.val());
   inv_a_log10 = T(1)/(a.val() * log10);

   return Dual2<T> (log10a, inv_a_log10 * a.dx(), inv_a_log10 * a.dy());
}

// f(x) = e^x, f'(x) = e^x
template<class T>
Dual2<T> exp (const Dual2<T> &a)
{
   T expa;
   expa  = std::exp (a.val());

   return Dual2<T> (expa, expa * a.dx(), expa * a.dy());
}

// f(x) = 2^x, f'(x) = (2^x)*log(2)
template<class T>
Dual2<T> exp2 (const Dual2<T> &a)
{
   T exp2a, ln2;
   exp2a  = std::pow (T(2), a.val());
   ln2    = std::log (T(2));

   return Dual2<T> (exp2a, exp2a*ln2*a.dx(), exp2a*ln2*a.dy());
}

// f(x) = e^x - 1, f'(x) = e^x
template<class T>
Dual2<T> expm1 (const Dual2<T> &a)
{
   float expm1a, expa;
   expm1a = expm1f (a.val());
   expa   = std::exp (a.val());

   return Dual2<T> (expm1a, expa * a.dx(), expa * a.dy());
}

// f(x) = erf(x), f'(x) = (2e^(-x^2))/sqrt(pi)
template<class T>
Dual2<T> erf (const Dual2<T> &a)
{
   T erfa, derfadx;
   erfa    = erff (a.val()); // float version!
   derfadx = T(2)*std::exp(-a.val()*a.val())/std::sqrt(T(M_PI));

   return Dual2<T> (erfa, derfadx * a.dx(), derfadx * a.dy());
}

// f(x) = erfc(x), f'(x) = -(2e^(-x^2))/sqrt(pi)
template<class T>
Dual2<T> erfc (const Dual2<T> &a)
{
   T erfca, derfcadx;
   erfca    = erfcf (a.val()); // float version!
   derfcadx = -T(2)*std::exp(-a.val()*a.val())/std::sqrt(T(M_PI));

   return Dual2<T> (erfca, derfcadx * a.dx(), derfcadx * a.dy());
}

// f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
template<class T>
Dual2<T> sqrt (const Dual2<T> &a)
{
   // do we want to print an error message?
   if (a.val() <= T(0))
      return Dual2<T> (T(0), T(0), T(0));

   T sqrta, inv_2sqrta;
   sqrta      = std::sqrt(a.val());
   inv_2sqrta = T(1)/(T(2)*sqrta);

   return Dual2<T> (sqrta, inv_2sqrta * a.dx(), inv_2sqrta * a.dy());
}

// f(x) = 1/sqrt(x), f'(x) = -1/(2*x^(3/2))
template<class T>
Dual2<T> inversesqrt (const Dual2<T> &a)
{
   // do we want to print an error message?
   if (a.val() <= T(0))
      return Dual2<T> (T(0), T(0), T(0));

   T sqrta, inv_neg2asqrta;
   sqrta          = std::sqrt(a.val());
   inv_neg2asqrta = -T(1)/(T(2)*a.val()*sqrta);

   return Dual2<T> (T(1)/sqrta, inv_neg2asqrta * a.dx(), inv_neg2asqrta * a.dy());
}

// (fx) = x*(1-a) + y*a, f'(x) = (1-a)x' + (y - x)*a' + a*y'
template<class T>
Dual2<T> mix (const Dual2<T> &x, const Dual2<T> &y, const Dual2<T> &a)
{
   T mixval = x.val()*(T(1)-a.val()) + y.val()*a.val();

   return Dual2<T> (mixval, (T(1) - a.val())*x.dx() + (y.val() - x.val())*a.dx() + a.val()*y.dx(),
                            (T(1) - a.val())*x.dy() + (y.val() - x.val())*a.dy() + a.val()*y.dy());
}

// f(x) = sqrt(x*x + y*y), f'(x) = (x x' + y y')/sqrt(x*x + y*y)
template<class T>
Dual2<T> dual_hypot (const Dual2<T> &x, const Dual2<T> &y)
{
   if (x.val() == T(0) && y.val() == T(0))
      return Dual2<T> (T(0), T(0), T(0));

   T hypotxy =  std::sqrt(x.val()*x.val() + y.val()*y.val());
   T denom = T(1) / hypotxy;

   return Dual2<T> (hypotxy, (x.val()*x.dx() + y.val()*y.dx()) * denom,
                             (x.val()*x.dy() + y.val()*y.dy()) * denom);
}

// f(x) = sqrt(x*x + y*y + z*z), f'(x) = (x x' + y y' + z z')/sqrt(x*x + y*y + z*z)
template<class T>
Dual2<T> dual_hypot (const Dual2<T> &x, const Dual2<T> &y, const Dual2<T> &z)
{
   if (x.val() == T(0) && y.val() == T(0) && z.val() == T(0))
      return Dual2<T> (T(0), T(0), T(0));

   T hypotxyz =  std::sqrt(x.val()*x.val() + y.val()*y.val() + z.val()*z.val());
   T denom = T(1) / hypotxyz;

   return Dual2<T> (hypotxyz, (x.val()*x.dx() + y.val()*y.dx() + z.val()*z.dx()) * denom,
                              (x.val()*x.dy() + y.val()*y.dy() + z.val()*z.dy()) * denom);
}

template<class T>
Dual2<T> dual_min (const Dual2<T> &x, const Dual2<T> &y)
{
   if (x.val() > y.val())
      return y;
   else 
      return x;
}

template<class T>
Dual2<T> dual_max (const Dual2<T> &x, const Dual2<T> &y)
{
   if (x.val() > y.val())
      return x;
   else 
      return y;
}

template<class T>
Dual2<T> dual_clamp (const Dual2<T> &x, const Dual2<T> &minv, const Dual2<T> &maxv)
{
   if (x.val() < minv.val()) return minv;
   else if (x.val() > maxv.val()) return maxv;
   else return x;
}

// f(t) = (3-2t)t^2,   t = (x-e0)/(e1-e0)
template<class T>
Dual2<T> smoothstep (const Dual2<T> &e0, const Dual2<T> &e1, const Dual2<T> &x)
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

}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* DUAL_H */
