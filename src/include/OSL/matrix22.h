/*
Copyright (c) 2012 Sony Pictures Imageworks Inc., et al.
and
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


// Extend Imath matrix classes to include 2x2, as analogous as possible
// to Imath::Matrix33 and Imath::Matrix44.
// 
// The original Imath classes bear the "new BSD" license (same as
// ours above) and this copyright:
// Copyright (c) 2002, Industrial Light & Magic, a division of
// Lucas Digital Ltd. LLC.  All rights reserved.


#pragma once


#include <OSL/oslconfig.h>

#ifndef __CUDA_ARCH__
#include <OpenEXR/ImathMatrix.h>
#else
#include <OSL/ImathMatrix_cuda.h>
#include <limits>
#endif

namespace Imathx {   // "extended" Imath

enum Uninitialized {UNINITIALIZED};


// TODO: It would be preferable to use the Imath versions of these functions in
//       all cases, but these templates should suffice until a more complete
//       device-friendly version of Imath is available.
namespace hostdevice {
template <typename T> inline OSL_HOSTDEVICE T abs (T x);
template <typename T> inline OSL_HOSTDEVICE T smallest ();
#ifndef __CUDA_ARCH__
template <> inline OSL_HOSTDEVICE double abs<double>      (double x) { return Imath::abs (x); }
template <> inline OSL_HOSTDEVICE float  abs<float>       (float x)  { return Imath::abs (x); }
template <> inline OSL_HOSTDEVICE double smallest<double> ()         { return Imath::limits<double>::smallest(); }
template <> inline OSL_HOSTDEVICE float  smallest<float>  ()         { return Imath::limits<float>::smallest();  }
#else
template <> inline OSL_HOSTDEVICE double abs<double>      (double x) { return std::abs (x); }
template <> inline OSL_HOSTDEVICE float  abs<float>       (float x)  { return std::abs (x); }
template <> inline OSL_HOSTDEVICE double smallest<double> ()         { return std::numeric_limits<double>::lowest(); }
template <> inline OSL_HOSTDEVICE float  smallest<float>  ()         { return std::numeric_limits<float>::lowest();  }
#endif
}


template <class T> class Matrix22
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T           x[2][2];

    OSL_HOSTDEVICE
    T *         operator [] (int i);

    OSL_HOSTDEVICE
    const T *   operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    OSL_HOSTDEVICE
    Matrix22 (Uninitialized) {}

    OSL_HOSTDEVICE
    Matrix22 ();
                                // 1 0
                                // 0 1

    OSL_HOSTDEVICE
    Matrix22 (T a);
                                // a a
                                // a a

    OSL_HOSTDEVICE
    Matrix22 (const T a[2][2]);
                                // a[0][0] a[0][1]
                                // a[1][0] a[1][1]

    OSL_HOSTDEVICE
    Matrix22 (T a, T b, T c, T d);
                                // a b
                                // c d


    //--------------------------------
    // Copy constructor and assignment
    //--------------------------------

    OSL_HOSTDEVICE
    Matrix22 (const Matrix22 &v);

    template <class S> OSL_HOSTDEVICE explicit
    Matrix22 (const Matrix22<S> &v);

    OSL_HOSTDEVICE
    const Matrix22 &    operator = (const Matrix22 &v);

    OSL_HOSTDEVICE
    const Matrix22 &    operator = (T a);


    //----------------------
    // Compatibility with Sb
    //----------------------
    
    OSL_HOSTDEVICE
    T *                 getValue ();

    OSL_HOSTDEVICE
    const T *           getValue () const;

    template <class S> OSL_HOSTDEVICE
    void                getValue (Matrix22<S> &v) const;

    template <class S> OSL_HOSTDEVICE
    Matrix22 &          setValue (const Matrix22<S> &v);

    template <class S> OSL_HOSTDEVICE
    Matrix22 &          setTheMatrix (const Matrix22<S> &v);


    //---------
    // Identity
    //---------

    OSL_HOSTDEVICE
    void                makeIdentity();


    //---------
    // Equality
    //---------

    OSL_HOSTDEVICE
    bool                operator == (const Matrix22 &v) const;

    OSL_HOSTDEVICE
    bool                operator != (const Matrix22 &v) const;

    //-----------------------------------------------------------------------
    // Compare two matrices and test if they are "approximately equal":
    //
    // equalWithAbsError (m, e)
    //
    //      Returns true if the coefficients of this and m are the same with
    //      an absolute error of no more than e, i.e., for all i, j
    //
    //      abs (this[i][j] - m[i][j]) <= e
    //
    // equalWithRelError (m, e)
    //
    //      Returns true if the coefficients of this and m are the same with
    //      a relative error of no more than e, i.e., for all i, j
    //
    //      abs (this[i] - v[i][j]) <= e * abs (this[i][j])
    //-----------------------------------------------------------------------

    OSL_HOSTDEVICE
    bool                equalWithAbsError (const Matrix22<T> &v, T e) const;

    OSL_HOSTDEVICE
    bool                equalWithRelError (const Matrix22<T> &v, T e) const;


    //------------------------
    // Component-wise addition
    //------------------------

    OSL_HOSTDEVICE
    const Matrix22 &    operator += (const Matrix22 &v);

    OSL_HOSTDEVICE
    const Matrix22 &    operator += (T a);

    OSL_HOSTDEVICE
    Matrix22            operator + (const Matrix22 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    OSL_HOSTDEVICE
    const Matrix22 &    operator -= (const Matrix22 &v);

    OSL_HOSTDEVICE
    const Matrix22 &    operator -= (T a);

    OSL_HOSTDEVICE
    Matrix22            operator - (const Matrix22 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    OSL_HOSTDEVICE
    Matrix22            operator - () const;

    OSL_HOSTDEVICE
    const Matrix22 &    negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    OSL_HOSTDEVICE
    const Matrix22 &    operator *= (T a);

    OSL_HOSTDEVICE
    Matrix22            operator * (T a) const;


    //-----------------------------------
    // Matrix-times-matrix multiplication
    //-----------------------------------

    OSL_HOSTDEVICE
    const Matrix22 &    operator *= (const Matrix22 &v);

    OSL_HOSTDEVICE
    Matrix22            operator * (const Matrix22 &v) const;


    //-----------------------------------------------------------------
    // Vector-times-matrix multiplication; see also the "operator *"
    // functions defined below.
    //
    // m.multVecMatrix(src,dst) implements a homogeneous transformation
    // by computing Vec3 (src.x, src.y, 1) * m and dividing by the
    // result's third element.
    //
    // m.multDirMatrix(src,dst) multiplies src by the upper left 2x2
    // submatrix, ignoring the rest of matrix m.
    //-----------------------------------------------------------------

    template <class S> OSL_HOSTDEVICE
    void                multMatrix(const Imath::Vec2<S> &src, Imath::Vec2<S> &dst) const;



    //------------------------
    // Component-wise division
    //------------------------

    OSL_HOSTDEVICE
    const Matrix22 &    operator /= (T a);

    OSL_HOSTDEVICE
    Matrix22            operator / (T a) const;


    //------------------
    // Transposed matrix
    //------------------

    OSL_HOSTDEVICE
    const Matrix22 &    transpose ();

    OSL_HOSTDEVICE
    Matrix22            transposed () const;


    //------------------------------------------------------------
    // Inverse matrix: If singExc is false, inverting a singular
    // matrix produces an identity matrix.  If singExc is true,
    // inverting a singular matrix throws a SingMatrixExc.
    //
    // inverse() and invert() invert matrices using determinants.
    // 
    //------------------------------------------------------------

    OSL_HOSTDEVICE
    const Matrix22 &    invert (bool singExc = false);

    OSL_HOSTDEVICE
    Matrix22<T>         inverse (bool singExc = false) const;


    //-----------------------------------------
    // Set matrix to rotation by r (in radians)
    //-----------------------------------------

    template <class S> OSL_HOSTDEVICE
    const Matrix22 &    setRotation (S r);


    //-----------------------------
    // Rotate the given matrix by r
    //-----------------------------

    template <class S> OSL_HOSTDEVICE
    const Matrix22 &    rotate (S r);


    //--------------------------------------------
    // Set matrix to scale by given uniform factor
    //--------------------------------------------

    OSL_HOSTDEVICE
    const Matrix22 &    setScale (T s);


    //------------------------------------
    // Set matrix to scale by given vector
    //------------------------------------

    template <class S> OSL_HOSTDEVICE
    const Matrix22 &    setScale (const Imath::Vec2<S> &s);


    //----------------------
    // Scale the matrix by s
    //----------------------

    template <class S> OSL_HOSTDEVICE
    const Matrix22 &    scale (const Imath::Vec2<S> &s);


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    OSL_HOSTDEVICE
    static T            baseTypeMin()           {return Imath::limits<T>::min();}

    OSL_HOSTDEVICE
    static T            baseTypeMax()           {return Imath::limits<T>::max();}

    OSL_HOSTDEVICE
    static T            baseTypeSmallest()      {return Imath::limits<T>::smallest();}

    OSL_HOSTDEVICE
    static T            baseTypeEpsilon()       {return Imath::limits<T>::epsilon();}

  private:

    template <typename R, typename S>
    struct isSameType
    {
        enum {value = 0};
    };

    template <typename R>
    struct isSameType<R, R>
    {
        enum {value = 1};
    };
};



//--------------
// Stream output
//--------------

template <class T>
std::ostream &  operator << (std::ostream & s, const Matrix22<T> &m); 


//---------------------------------------------
// Vector-times-matrix multiplication operators
//---------------------------------------------

template <class S, class T> OSL_HOSTDEVICE
Imath::Vec2<S> operator * (const Matrix22<T> &m, const Imath::Vec2<S> &v)
{
    Imath::Vec2<S> tmp;
    m.multMatrix (v, tmp);
    return tmp;
}



//-------------------------
// Typedefs for convenience
//-------------------------

typedef Matrix22 <float>  M22f;
typedef Matrix22 <double> M22d;


//---------------------------
// Implementation of Matrix22
//---------------------------

template <class T> OSL_HOSTDEVICE
inline T *
Matrix22<T>::operator [] (int i)
{
    return x[i];
}

template <class T> OSL_HOSTDEVICE
inline const T *
Matrix22<T>::operator [] (int i) const
{
    return x[i];
}

template <class T> OSL_HOSTDEVICE
inline
Matrix22<T>::Matrix22 ()
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
}

template <class T> OSL_HOSTDEVICE
inline
Matrix22<T>::Matrix22 (T a)
{
    x[0][0] = a;
    x[0][1] = a;
    x[1][0] = a;
    x[1][1] = a;
}

template <class T> OSL_HOSTDEVICE
inline
Matrix22<T>::Matrix22 (const T a[2][2]) 
{
    memcpy (x, a, sizeof (x));
}

template <class T> OSL_HOSTDEVICE
inline
Matrix22<T>::Matrix22 (T a, T b, T c, T d)
{
    x[0][0] = a;
    x[0][1] = b;
    x[1][0] = c;
    x[1][1] = d;
}

template <class T> OSL_HOSTDEVICE
inline
Matrix22<T>::Matrix22 (const Matrix22 &v)
{
    memcpy (x, v.x, sizeof (x));
}

template <class T>
template <class S> OSL_HOSTDEVICE
inline
Matrix22<T>::Matrix22 (const Matrix22<S> &v)
{
    x[0][0] = T (v.x[0][0]);
    x[0][1] = T (v.x[0][1]);
    x[1][0] = T (v.x[1][0]);
    x[1][1] = T (v.x[1][1]);
}

template <class T> OSL_HOSTDEVICE
inline const Matrix22<T> &
Matrix22<T>::operator = (const Matrix22 &v)
{
    memcpy (x, v.x, sizeof (x));
    return *this;
}

template <class T> OSL_HOSTDEVICE
inline const Matrix22<T> &
Matrix22<T>::operator = (T a)
{
    x[0][0] = a;
    x[0][1] = a;
    x[1][0] = a;
    x[1][1] = a;
    return *this;
}

template <class T> OSL_HOSTDEVICE
inline T *
Matrix22<T>::getValue ()
{
    return (T *) &x[0][0];
}

template <class T> OSL_HOSTDEVICE
inline const T *
Matrix22<T>::getValue () const
{
    return (const T *) &x[0][0];
}

template <class T>
template <class S> OSL_HOSTDEVICE
inline void
Matrix22<T>::getValue (Matrix22<S> &v) const
{
    if (isSameType<S,T>::value)
    {
        memcpy (v.x, x, sizeof (x));
    }
    else
    {
        v.x[0][0] = x[0][0];
        v.x[0][1] = x[0][1];
        v.x[1][0] = x[1][0];
        v.x[1][1] = x[1][1];
    }
}

template <class T>
template <class S> OSL_HOSTDEVICE
inline Matrix22<T> &
Matrix22<T>::setValue (const Matrix22<S> &v)
{
    if (isSameType<S,T>::value)
    {
        memcpy (x, v.x, sizeof (x));
    }
    else
    {
        x[0][0] = v.x[0][0];
        x[0][1] = v.x[0][1];
        x[1][0] = v.x[1][0];
        x[1][1] = v.x[1][1];
    }

    return *this;
}

template <class T>
template <class S> OSL_HOSTDEVICE
inline Matrix22<T> &
Matrix22<T>::setTheMatrix (const Matrix22<S> &v)
{
    if (isSameType<S,T>::value)
    {
        memcpy (x, v.x, sizeof (x));
    }
    else
    {
        x[0][0] = v.x[0][0];
        x[0][1] = v.x[0][1];
        x[1][0] = v.x[1][0];
        x[1][1] = v.x[1][1];
    }

    return *this;
}

template <class T> OSL_HOSTDEVICE
inline void
Matrix22<T>::makeIdentity()
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
}

template <class T> OSL_HOSTDEVICE
bool
Matrix22<T>::operator == (const Matrix22 &v) const
{
    return x[0][0] == v.x[0][0] &&
           x[0][1] == v.x[0][1] &&
           x[1][0] == v.x[1][0] &&
           x[1][1] == v.x[1][1];
}

template <class T> OSL_HOSTDEVICE
bool
Matrix22<T>::operator != (const Matrix22 &v) const
{
    return x[0][0] != v.x[0][0] ||
           x[0][1] != v.x[0][1] ||
           x[1][0] != v.x[1][0] ||
           x[1][1] != v.x[1][1];
}

template <class T> OSL_HOSTDEVICE
bool
Matrix22<T>::equalWithAbsError (const Matrix22<T> &m, T e) const
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            if (!Imath::equalWithAbsError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T> OSL_HOSTDEVICE
bool
Matrix22<T>::equalWithRelError (const Matrix22<T> &m, T e) const
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            if (!Imath::equalWithRelError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::operator += (const Matrix22<T> &v)
{
    x[0][0] += v.x[0][0];
    x[0][1] += v.x[0][1];
    x[1][0] += v.x[1][0];
    x[1][1] += v.x[1][1];

    return *this;
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::operator += (T a)
{
    x[0][0] += a;
    x[0][1] += a;
    x[1][0] += a;
    x[1][1] += a;
  
    return *this;
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::operator + (const Matrix22<T> &v) const
{
    return Matrix22 (x[0][0] + v.x[0][0],
                     x[0][1] + v.x[0][1],
                     x[1][0] + v.x[1][0],
                     x[1][1] + v.x[1][1]);
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::operator -= (const Matrix22<T> &v)
{
    x[0][0] -= v.x[0][0];
    x[0][1] -= v.x[0][1];
    x[1][0] -= v.x[1][0];
    x[1][1] -= v.x[1][1];
  
    return *this;
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::operator -= (T a)
{
    x[0][0] -= a;
    x[0][1] -= a;
    x[1][0] -= a;
    x[1][1] -= a;
  
    return *this;
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::operator - (const Matrix22<T> &v) const
{
    return Matrix22 (x[0][0] - v.x[0][0],
                     x[0][1] - v.x[0][1],
                     x[1][0] - v.x[1][0],
                     x[1][1] - v.x[1][1]);
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::operator - () const
{
    return Matrix22 (-x[0][0],
                     -x[0][1],
                     -x[1][0],
                     -x[1][1]);
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::negate ()
{
    x[0][0] = -x[0][0];
    x[0][1] = -x[0][1];
    x[1][0] = -x[1][0];
    x[1][1] = -x[1][1];

    return *this;
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::operator *= (T a)
{
    x[0][0] *= a;
    x[0][1] *= a;
    x[1][0] *= a;
    x[1][1] *= a;
  
    return *this;
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::operator * (T a) const
{
    return Matrix22 (x[0][0] * a,
                     x[0][1] * a,
                     x[1][0] * a,
                     x[1][1] * a);
}

template <class T> OSL_HOSTDEVICE
inline Matrix22<T>
operator * (T a, const Matrix22<T> &v)
{
    return v * a;
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::operator *= (const Matrix22<T> &v)
{
    Matrix22 tmp (T (0));

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                tmp.x[i][j] += x[i][k] * v.x[k][j];

    *this = tmp;
    return *this;
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::operator * (const Matrix22<T> &v) const
{
    Matrix22 tmp (T (0));

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                tmp.x[i][j] += x[i][k] * v.x[k][j];

    return tmp;
}

template <class T>
template <class S> OSL_HOSTDEVICE
void
Matrix22<T>::multMatrix(const Imath::Vec2<S> &src, Imath::Vec2<S> &dst) const
{
    S a, b;

    a = src[0] * x[0][0] + src[1] * x[1][0];
    b = src[0] * x[0][1] + src[1] * x[1][1];

    dst.x = a;
    dst.y = b;
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::operator /= (T a)
{
    x[0][0] /= a;
    x[0][1] /= a;
    x[1][0] /= a;
    x[1][1] /= a;
  
    return *this;
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::operator / (T a) const
{
    return Matrix22 (x[0][0] / a,
                     x[0][1] / a,
                     x[1][0] / a,
                     x[1][1] / a);
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::transpose ()
{
    Matrix22 tmp (x[0][0],
                  x[1][0],
                  x[0][1],
                  x[1][1]);
    *this = tmp;
    return *this;
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::transposed () const
{
    return Matrix22 (x[0][0],
                     x[1][0],
                     x[0][1],
                     x[1][1]);
}


template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::invert (bool singExc)
{
    *this = inverse (singExc);
    return *this;
}

template <class T> OSL_HOSTDEVICE
Matrix22<T>
Matrix22<T>::inverse (bool singExc) const
{
    Matrix22 s ( x[1][1],  -x[0][1],
                -x[1][0],   x[0][0]);
    T r = x[0][0] * x[1][1] - x[1][0] * x[0][1];  // determinant

    if (hostdevice::abs (r) >= 1)
    {
        s /= r;
    }
    else
    {
        T mr = hostdevice::abs (r) / hostdevice::smallest<T>();

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                if (mr > hostdevice::abs (s[i][j]))
                {
                    s[i][j] /= r;
                }
                else
                {
#ifndef __CUDA_ARCH__
                    if (singExc)
                        throw Imath::SingMatrixExc ("Cannot invert "
                                             "singular matrix.");
#endif
                    return Matrix22();
                }
            }
        }
    }

    return s;
}

template <class T>
template <class S> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::setRotation (S r)
{
    S cos_r, sin_r;

    cos_r = Imath::Math<T>::cos (r);
    sin_r = Imath::Math<T>::sin (r);

    x[0][0] =  cos_r;
    x[0][1] =  sin_r;

    x[1][0] =  -sin_r;
    x[1][1] =  cos_r;

    return *this;
}

template <class T>
template <class S> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::rotate (S r)
{
    *this *= Matrix22<T>().setRotation (r);
    return *this;
}

template <class T> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::setScale (T s)
{
    memset (x, 0, sizeof (x));
    x[0][0] = s;
    x[1][1] = s;

    return *this;
}

template <class T>
template <class S> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::setScale (const Imath::Vec2<S> &s)
{
    memset (x, 0, sizeof (x));
    x[0][0] = s[0];
    x[1][1] = s[1];

    return *this;
}

template <class T>
template <class S> OSL_HOSTDEVICE
const Matrix22<T> &
Matrix22<T>::scale (const Imath::Vec2<S> &s)
{
    x[0][0] *= s[0];
    x[0][1] *= s[0];

    x[1][0] *= s[1];
    x[1][1] *= s[1];

    return *this;
}


//--------------------------------
// Implementation of stream output
//--------------------------------

template <class T>
std::ostream &
operator << (std::ostream &s, const Matrix22<T> &m)
{
    std::ios_base::fmtflags oldFlags = s.flags();
    int width;

    if (s.flags() & std::ios_base::fixed)
    {
        s.setf (std::ios_base::showpoint);
        width = s.precision() + 5;
    }
    else
    {
        s.setf (std::ios_base::scientific);
        s.setf (std::ios_base::showpoint);
        width = s.precision() + 8;
    }

    s << "(" << std::setw (width) << m[0][0] <<
         " " << std::setw (width) << m[0][1] << "\n" <<

         " " << std::setw (width) << m[1][0] <<
         " " << std::setw (width) << m[1][1] << ")\n";

    s.flags (oldFlags);
    return s;
}

} // namespace Imathx
