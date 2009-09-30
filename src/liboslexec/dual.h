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

    /// Return the partial derivative with respect to x
    ///
    const T& dx () const { return m_dx; }

    /// Return the partial derivative with respect to y
    ///
    const T& dy () const { return m_dy; }

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
    return Dual2<T> (a.val()*b, a.dx()*b, a.dx()*b);
}


/// Multiplication of dual by scalar.
///
template<class T>
Dual2<T> operator* (const T &b, const Dual2<T> &a)
{
    return Dual2<T> (a.val()*b, a.dx()*b, a.dx()*b);
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


template<class T>
Dual2<T> cos (const Dual2<T> &a)
{
    float sina, cosa;
    sina = std::sin (a.val());
    cosa = std::cos (a.val());
    return Dual2<T> (cosa, -sina * a.dx(), -sina * a.dy());
}


template<class T>
Dual2<T> sin (const Dual2<T> &a)
{
    float sina, cosa;
    sina = std::sin (a.val());
    cosa = std::cos (a.val());
    return Dual2<T> (sina, cosa * a.dx(), cosa * a.dy());
}


}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* DUAL_H */
