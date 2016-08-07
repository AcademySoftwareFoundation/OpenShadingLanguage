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

#include "llvm_ops_math.h"

namespace Imath {

template <class T> class Vec3
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T			x, y, z;

    T &			operator [] (int i);
    const T &		operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    Vec3 ();			   // no initialization
    explicit Vec3 (T a);           // (a a a)
    Vec3 (T a, T b, T c);	   // (a b c)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    Vec3 (const Vec3 &v);
    template <class S> Vec3 (const Vec3<S> &v);

    const Vec3 &	operator = (const Vec3 &v);

    //----------------------
    // Compatibility with Sb
    //----------------------

    template <class S>
    void		setValue (S a, S b, S c);

    template <class S>
    void		setValue (const Vec3<S> &v);

    template <class S>
    void		getValue (S &a, S &b, S &c) const;

    template <class S>
    void		getValue (Vec3<S> &v) const;

    T *			getValue();
    const T *		getValue() const;


    //---------
    // Equality
    //---------

    template <class S>
    bool		operator == (const Vec3<S> &v) const;

    template <class S>
    bool		operator != (const Vec3<S> &v) const;

    //------------
    // Dot product
    //------------

    T			dot (const Vec3 &v) const;
    T			operator ^ (const Vec3 &v) const;


    //---------------------------
    // Right-handed cross product
    //---------------------------

    Vec3		cross (const Vec3 &v) const;
    const Vec3 &	operator %= (const Vec3 &v);
    Vec3		operator % (const Vec3 &v) const;


    //------------------------
    // Component-wise addition
    //------------------------

    const Vec3 &	operator += (const Vec3 &v);
    Vec3		operator + (const Vec3 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Vec3 &	operator -= (const Vec3 &v);
    Vec3		operator - (const Vec3 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Vec3		operator - () const;
    const Vec3 &	negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Vec3 &	operator *= (const Vec3 &v);
    const Vec3 &	operator *= (T a);
    Vec3		operator * (const Vec3 &v) const;
    Vec3		operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Vec3 &	operator /= (const Vec3 &v);
    const Vec3 &	operator /= (T a);
    Vec3		operator / (const Vec3 &v) const;
    Vec3		operator / (T a) const;


    //----------------------------------------------------------------
    // Length and normalization:  If v.length() is 0.0, v.normalize()
    // and v.normalized() produce a null vector; v.normalizeExc() and
    // v.normalizedExc() throw a NullVecExc.
    // v.normalizeNonNull() and v.normalizedNonNull() are slightly
    // faster than the other normalization routines, but if v.length()
    // is 0.0, the result is undefined.
    //----------------------------------------------------------------

    T			length () const;
    T			length2 () const;

    const Vec3 &	normalize ();           // modifies *this
    const Vec3 &	normalizeExc ();
    const Vec3 &	normalizeNonNull ();

    Vec3<T>		normalized () const;	// does not modify *this
    Vec3<T>		normalizedExc () const;
    Vec3<T>		normalizedNonNull () const;


    //--------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Vec3
    //--------------------------------------------------------

    static unsigned int	dimensions() {return 3;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T		baseTypeMin()		{return limits<T>::min();}
    static T		baseTypeMax()		{return limits<T>::max();}
    static T		baseTypeSmallest()	{return limits<T>::smallest();}
    static T		baseTypeEpsilon()	{return limits<T>::epsilon();}


    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be either a Vec2<T>, a Vec3<T>, or a Vec4<T> you can 
    // refer to T as V::BaseType
    //--------------------------------------------------------------

    typedef T		BaseType;

  private:

    T			lengthTiny () const;
};

//----------------------------------------------------
// Reverse multiplication: S * Vec2<T> and S * Vec3<T>
//----------------------------------------------------

template <class T> Vec3<T>	operator * (T a, const Vec3<T> &v);

//-----------------------
// Implementation of Vec3
//-----------------------

template <class T>
inline T &
Vec3<T>::operator [] (int i)
{
    return (&x)[i];
}

template <class T>
inline const T &
Vec3<T>::operator [] (int i) const
{
    return (&x)[i];
}

template <class T>
inline
Vec3<T>::Vec3 ()
{
    // empty
}

template <class T>
inline
Vec3<T>::Vec3 (T a)
{
    x = y = z = a;
}

template <class T>
inline
Vec3<T>::Vec3 (T a, T b, T c)
{
    x = a;
    y = b;
    z = c;
}

template <class T>
inline
Vec3<T>::Vec3 (const Vec3 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
}

template <class T>
template <class S>
inline
Vec3<T>::Vec3 (const Vec3<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator = (const Vec3 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

template <class T>
template <class S>
inline void
Vec3<T>::setValue (S a, S b, S c)
{
    x = T (a);
    y = T (b);
    z = T (c);
}

template <class T>
template <class S>
inline void
Vec3<T>::setValue (const Vec3<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
}

template <class T>
template <class S>
inline void
Vec3<T>::getValue (S &a, S &b, S &c) const
{
    a = S (x);
    b = S (y);
    c = S (z);
}

template <class T>
template <class S>
inline void
Vec3<T>::getValue (Vec3<S> &v) const
{
    v.x = S (x);
    v.y = S (y);
    v.z = S (z);
}

template <class T>
inline T *
Vec3<T>::getValue()
{
    return (T *) &x;
}

template <class T>
inline const T *
Vec3<T>::getValue() const
{
    return (const T *) &x;
}

template <class T>
template <class S>
inline bool
Vec3<T>::operator == (const Vec3<S> &v) const
{
    return x == v.x && y == v.y && z == v.z;
}

template <class T>
template <class S>
inline bool
Vec3<T>::operator != (const Vec3<S> &v) const
{
    return x != v.x || y != v.y || z != v.z;
}

template <class T>
inline T
Vec3<T>::dot (const Vec3 &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

template <class T>
inline T
Vec3<T>::operator ^ (const Vec3 &v) const
{
    return dot (v);
}

template <class T>
inline Vec3<T>
Vec3<T>::cross (const Vec3 &v) const
{
    return Vec3 (y * v.z - z * v.y,
		 z * v.x - x * v.z,
		 x * v.y - y * v.x);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator %= (const Vec3 &v)
{
    T a = y * v.z - z * v.y;
    T b = z * v.x - x * v.z;
    T c = x * v.y - y * v.x;
    x = a;
    y = b;
    z = c;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator % (const Vec3 &v) const
{
    return Vec3 (y * v.z - z * v.y,
		 z * v.x - x * v.z,
		 x * v.y - y * v.x);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator += (const Vec3 &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator + (const Vec3 &v) const
{
    return Vec3 (x + v.x, y + v.y, z + v.z);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator -= (const Vec3 &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator - (const Vec3 &v) const
{
    return Vec3 (x - v.x, y - v.y, z - v.z);
}

template <class T>
inline Vec3<T>
Vec3<T>::operator - () const
{
    return Vec3 (-x, -y, -z);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::negate ()
{
    x = -x;
    y = -y;
    z = -z;
    return *this;
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator *= (const Vec3 &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator *= (T a)
{
    x *= a;
    y *= a;
    z *= a;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator * (const Vec3 &v) const
{
    return Vec3 (x * v.x, y * v.y, z * v.z);
}

template <class T>
inline Vec3<T>
Vec3<T>::operator * (T a) const
{
    return Vec3 (x * a, y * a, z * a);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator /= (const Vec3 &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator /= (T a)
{
    x /= a;
    y /= a;
    z /= a;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator / (const Vec3 &v) const
{
    return Vec3 (x / v.x, y / v.y, z / v.z);
}

template <class T>
inline Vec3<T>
Vec3<T>::operator / (T a) const
{
    return Vec3 (x / a, y / a, z / a);
}

template <class T>
T
Vec3<T>::lengthTiny () const
{
    T absX = (x >= T (0))? x: -x;
    T absY = (y >= T (0))? y: -y;
    T absZ = (z >= T (0))? z: -z;
    
    T max = absX;

    if (max < absY)
	max = absY;

    if (max < absZ)
	max = absZ;

    if (max == T (0))
		return T (0);

    //
    // Do not replace the divisions by max with multiplications by 1/max.
    // Computing 1/max can overflow but the divisions below will always
    // produce results less than or equal to 1.
    //

    absX /= max;
    absY /= max;
    absZ /= max;

    return max * Math<T>::sqrt (absX * absX + absY * absY + absZ * absZ);
}

template <class T>
inline T
Vec3<T>::length () const
{
    T length2 = dot (*this);

    if (length2 < T (2) * limits<T>::smallest())
		return lengthTiny();

    return Math<T>::sqrt (length2);
}

template <class T>
inline T
Vec3<T>::length2 () const
{
    return dot (*this);
}

template <class T>
const Vec3<T> &
Vec3<T>::normalize ()
{
    T l = length();

    if (l != T (0))
    {
        //
        // Do not replace the divisions by l with multiplications by 1/l.
        // Computing 1/l can overflow but the divisions below will always
        // produce results less than or equal to 1.
        //

	x /= l;
	y /= l;
	z /= l;
    }

    return *this;
}

template <class T>
const Vec3<T> &
Vec3<T>::normalizeExc ()
{
    T l = length();

    if (l == T (0))
		return *this;

    x /= l;
    y /= l;
    z /= l;
    return *this;
}

template <class T>
inline
const Vec3<T> &
Vec3<T>::normalizeNonNull ()
{
    T l = length();
    x /= l;
    y /= l;
    z /= l;
    return *this;
}

template <class T>
Vec3<T>
Vec3<T>::normalized () const
{
    T l = length();

    if (l == T (0))
	return Vec3 (T (0));

    return Vec3 (x / l, y / l, z / l);
}

template <class T>
Vec3<T>
Vec3<T>::normalizedExc () const
{
    T l = length();

    if (l == T (0))
		return Vec3(T(0));

    return Vec3 (x / l, y / l, z / l);
}

template <class T>
inline
Vec3<T>
Vec3<T>::normalizedNonNull () const
{
    T l = length();
    return Vec3 (x / l, y / l, z / l);
}


template <class T> class Matrix44
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T           x[4][4];

    T *         operator [] (int i);
    const T *   operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    Matrix44 ();
                                // 1 0 0 0
                                // 0 1 0 0
                                // 0 0 1 0
                                // 0 0 0 1

    Matrix44 (T a);
                                // a a a a
                                // a a a a
                                // a a a a
                                // a a a a

    Matrix44 (const T a[4][4]) ;
                                // a[0][0] a[0][1] a[0][2] a[0][3]
                                // a[1][0] a[1][1] a[1][2] a[1][3]
                                // a[2][0] a[2][1] a[2][2] a[2][3]
                                // a[3][0] a[3][1] a[3][2] a[3][3]

    Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h,
              T i, T j, T k, T l, T m, T n, T o, T p);

                                // a b c d
                                // e f g h
                                // i j k l
                                // m n o p


    //--------------------------------
    // Copy constructor and assignment
    //--------------------------------

    Matrix44 (const Matrix44 &v);
    template <class S> explicit Matrix44 (const Matrix44<S> &v);

    const Matrix44 &    operator = (const Matrix44 &v);
    const Matrix44 &    operator = (T a);


    //----------------------
    // Compatibility with Sb
    //----------------------
    
    T *                 getValue ();
    const T *           getValue () const;

    template <class S>
    void                getValue (Matrix44<S> &v) const;
    template <class S>
    Matrix44 &          setValue (const Matrix44<S> &v);

    template <class S>
    Matrix44 &          setTheMatrix (const Matrix44<S> &v);

    //---------
    // Identity
    //---------

    void                makeIdentity();


    //---------
    // Equality
    //---------

    bool                operator == (const Matrix44 &v) const;
    bool                operator != (const Matrix44 &v) const;


    //------------------------
    // Component-wise addition
    //------------------------

    const Matrix44 &    operator += (const Matrix44 &v);
    const Matrix44 &    operator += (T a);
    Matrix44            operator + (const Matrix44 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Matrix44 &    operator -= (const Matrix44 &v);
    const Matrix44 &    operator -= (T a);
    Matrix44            operator - (const Matrix44 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Matrix44            operator - () const;
    const Matrix44 &    negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Matrix44 &    operator *= (T a);
    Matrix44            operator * (T a) const;


    //-----------------------------------
    // Matrix-times-matrix multiplication
    //-----------------------------------

    const Matrix44 &    operator *= (const Matrix44 &v);
    Matrix44            operator * (const Matrix44 &v) const;

    static void         multiply (const Matrix44 &a,    // assumes that
                                  const Matrix44 &b,    // &a != &c and
                                  Matrix44 &c);         // &b != &c.


    //-----------------------------------------------------------------
    // Vector-times-matrix multiplication; see also the "operator *"
    // functions defined below.
    //
    // m.multVecMatrix(src,dst) implements a homogeneous transformation
    // by computing Vec4 (src.x, src.y, src.z, 1) * m and dividing by
    // the result's third element.
    //
    // m.multDirMatrix(src,dst) multiplies src by the upper left 3x3
    // submatrix, ignoring the rest of matrix m.
    //-----------------------------------------------------------------

    template <class S>
    void                multVecMatrix(const Vec3<S> &src, Vec3<S> &dst) const;

    template <class S>
    void                multDirMatrix(const Vec3<S> &src, Vec3<S> &dst) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Matrix44 &    operator /= (T a);
    Matrix44            operator / (T a) const;


    //------------------
    // Transposed matrix
    //------------------

    const Matrix44 &    transpose ();
    Matrix44            transposed () const;


    //------------------------------------------------------------
    // Inverse matrix: If singExc is false, inverting a singular
    // matrix produces an identity matrix.  If singExc is true,
    // inverting a singular matrix throws a SingMatrixExc.
    //
    // inverse() and invert() invert matrices using determinants;
    // gjInverse() and gjInvert() use the Gauss-Jordan method.
    //
    // inverse() and invert() are significantly faster than
    // gjInverse() and gjInvert(), but the results may be slightly
    // less accurate.
    // 
    //------------------------------------------------------------

    const Matrix44 &    invert (bool singExc = false);

    Matrix44<T>         inverse (bool singExc = false) const;

    const Matrix44 &    gjInvert (bool singExc = false);

    Matrix44<T>         gjInverse (bool singExc = false) const;

    //---------------------------------------------------
    // Build a minor using the specified rows and columns
    //---------------------------------------------------

    T                   fastMinor (const int r0, const int r1, const int r2,
                                   const int c0, const int c1, const int c2) const;

    //------------
    // Determinant
    //------------

    T                   determinant() const;

    //--------------------------------------------------------
    // Set matrix to rotation by XYZ euler angles (in radians)
    //--------------------------------------------------------

    template <class S>
    const Matrix44 &    setEulerAngles (const Vec3<S>& r);


    //--------------------------------------------------------
    // Set matrix to rotation around given axis by given angle
    //--------------------------------------------------------

    template <class S>
    const Matrix44 &    setAxisAngle (const Vec3<S>& ax, S ang);


    //-------------------------------------------
    // Rotate the matrix by XYZ euler angles in r
    //-------------------------------------------

    template <class S>
    const Matrix44 &    rotate (const Vec3<S> &r);


    //--------------------------------------------
    // Set matrix to scale by given uniform factor
    //--------------------------------------------

    const Matrix44 &    setScale (T s);


    //------------------------------------
    // Set matrix to scale by given vector
    //------------------------------------

    template <class S>
    const Matrix44 &    setScale (const Vec3<S> &s);


    //----------------------
    // Scale the matrix by s
    //----------------------

    template <class S>
    const Matrix44 &    scale (const Vec3<S> &s);


    //------------------------------------------
    // Set matrix to translation by given vector
    //------------------------------------------

    template <class S>
    const Matrix44 &    setTranslation (const Vec3<S> &t);


    //-----------------------------
    // Return translation component
    //-----------------------------

    const Vec3<T>       translation () const;


    //--------------------------
    // Translate the matrix by t
    //--------------------------

    template <class S>
    const Matrix44 &    translate (const Vec3<S> &t);


    //-------------------------------------------------------------
    // Set matrix to shear by given vector h.  The resulting matrix
    //    will shear x for each y coord. by a factor of h[0] ;
    //    will shear x for each z coord. by a factor of h[1] ;
    //    will shear y for each z coord. by a factor of h[2] .
    //-------------------------------------------------------------

    template <class S>
    const Matrix44 &    setShear (const Vec3<S> &h);


    //--------------------------------------------------------
    // Shear the matrix by given vector.  The composed matrix 
    // will be <shear> * <this>, where the shear matrix ...
    //    will shear x for each y coord. by a factor of h[0] ;
    //    will shear x for each z coord. by a factor of h[1] ;
    //    will shear y for each z coord. by a factor of h[2] .
    //--------------------------------------------------------

    template <class S>
    const Matrix44 &    shear (const Vec3<S> &h);

    //--------------------------------------------------------
    // Number of the row and column dimensions, since
    // Matrix44 is a square matrix.
    //--------------------------------------------------------

    static unsigned int	dimensions() {return 4;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T            baseTypeMin()           {return limits<T>::min();}
    static T            baseTypeMax()           {return limits<T>::max();}
    static T            baseTypeSmallest()      {return limits<T>::smallest();}
    static T            baseTypeEpsilon()       {return limits<T>::epsilon();}

    typedef T		BaseType;

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


//---------------------------------------------
// Vector-times-matrix multiplication operators
//---------------------------------------------

template <class S, class T>
const Vec3<S> &            operator *= (Vec3<S> &v, const Matrix44<T> &m);

template <class S, class T>
Vec3<S>                    operator * (const Vec3<S> &v, const Matrix44<T> &m);


//---------------------------
// Implementation of Matrix44
//---------------------------

template <class T>
inline T *
Matrix44<T>::operator [] (int i)
{
    return x[i];
}

template <class T>
inline const T *
Matrix44<T>::operator [] (int i) const
{
    return x[i];
}

template <class T>
inline
Matrix44<T>::Matrix44 ()
{
	x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;
}

template <class T>
inline
Matrix44<T>::Matrix44 (T a)
{
    x[0][0] = a;
    x[0][1] = a;
    x[0][2] = a;
    x[0][3] = a;
    x[1][0] = a;
    x[1][1] = a;
    x[1][2] = a;
    x[1][3] = a;
    x[2][0] = a;
    x[2][1] = a;
    x[2][2] = a;
    x[2][3] = a;
    x[3][0] = a;
    x[3][1] = a;
    x[3][2] = a;
    x[3][3] = a;
}

template <class T>
inline
Matrix44<T>::Matrix44 (const T a[4][4]) 
{
	x[0][0] = a[0][0];
    x[0][1] = a[0][1];
    x[0][2] = a[0][2];
    x[0][3] = a[0][3];
    x[1][0] = a[1][0];
    x[1][1] = a[1][1];
    x[1][2] = a[1][2];
    x[1][3] = a[1][3];
    x[2][0] = a[2][0];
    x[2][1] = a[2][1];
    x[2][2] = a[2][2];
    x[2][3] = a[2][3];
    x[3][0] = a[3][0];
    x[3][1] = a[3][1];
    x[3][2] = a[3][2];
    x[3][3] = a[3][3];
}

template <class T>
inline
Matrix44<T>::Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h,
                       T i, T j, T k, T l, T m, T n, T o, T p)
{
    x[0][0] = a;
    x[0][1] = b;
    x[0][2] = c;
    x[0][3] = d;
    x[1][0] = e;
    x[1][1] = f;
    x[1][2] = g;
    x[1][3] = h;
    x[2][0] = i;
    x[2][1] = j;
    x[2][2] = k;
    x[2][3] = l;
    x[3][0] = m;
    x[3][1] = n;
    x[3][2] = o;
    x[3][3] = p;
}

template <class T>
inline
Matrix44<T>::Matrix44 (const Matrix44 &v)
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[0][3] = v.x[0][3];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[1][3] = v.x[1][3];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    x[2][3] = v.x[2][3];
    x[3][0] = v.x[3][0];
    x[3][1] = v.x[3][1];
    x[3][2] = v.x[3][2];
    x[3][3] = v.x[3][3];
}

template <class T>
template <class S>
inline
Matrix44<T>::Matrix44 (const Matrix44<S> &v)
{
    x[0][0] = T (v.x[0][0]);
    x[0][1] = T (v.x[0][1]);
    x[0][2] = T (v.x[0][2]);
    x[0][3] = T (v.x[0][3]);
    x[1][0] = T (v.x[1][0]);
    x[1][1] = T (v.x[1][1]);
    x[1][2] = T (v.x[1][2]);
    x[1][3] = T (v.x[1][3]);
    x[2][0] = T (v.x[2][0]);
    x[2][1] = T (v.x[2][1]);
    x[2][2] = T (v.x[2][2]);
    x[2][3] = T (v.x[2][3]);
    x[3][0] = T (v.x[3][0]);
    x[3][1] = T (v.x[3][1]);
    x[3][2] = T (v.x[3][2]);
    x[3][3] = T (v.x[3][3]);
}

template <class T>
inline const Matrix44<T> &
Matrix44<T>::operator = (const Matrix44 &v)
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[0][3] = v.x[0][3];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[1][3] = v.x[1][3];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    x[2][3] = v.x[2][3];
    x[3][0] = v.x[3][0];
    x[3][1] = v.x[3][1];
    x[3][2] = v.x[3][2];
    x[3][3] = v.x[3][3];
    return *this;
}

template <class T>
inline const Matrix44<T> &
Matrix44<T>::operator = (T a)
{
    x[0][0] = a;
    x[0][1] = a;
    x[0][2] = a;
    x[0][3] = a;
    x[1][0] = a;
    x[1][1] = a;
    x[1][2] = a;
    x[1][3] = a;
    x[2][0] = a;
    x[2][1] = a;
    x[2][2] = a;
    x[2][3] = a;
    x[3][0] = a;
    x[3][1] = a;
    x[3][2] = a;
    x[3][3] = a;
    return *this;
}

template <class T>
inline T *
Matrix44<T>::getValue ()
{
    return (T *) &x[0][0];
}

template <class T>
inline const T *
Matrix44<T>::getValue () const
{
    return (const T *) &x[0][0];
}

template <class T>
template <class S>
inline void
Matrix44<T>::getValue (Matrix44<S> &v) const
{
    v.x[0][0] = x[0][0];
    v.x[0][1] = x[0][1];
    v.x[0][2] = x[0][2];
    v.x[0][3] = x[0][3];
    v.x[1][0] = x[1][0];
    v.x[1][1] = x[1][1];
    v.x[1][2] = x[1][2];
    v.x[1][3] = x[1][3];
    v.x[2][0] = x[2][0];
    v.x[2][1] = x[2][1];
    v.x[2][2] = x[2][2];
    v.x[2][3] = x[2][3];
    v.x[3][0] = x[3][0];
    v.x[3][1] = x[3][1];
    v.x[3][2] = x[3][2];
    v.x[3][3] = x[3][3];
}

template <class T>
template <class S>
inline Matrix44<T> &
Matrix44<T>::setValue (const Matrix44<S> &v)
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[0][3] = v.x[0][3];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[1][3] = v.x[1][3];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    x[2][3] = v.x[2][3];
    x[3][0] = v.x[3][0];
    x[3][1] = v.x[3][1];
    x[3][2] = v.x[3][2];
    x[3][3] = v.x[3][3];

    return *this;
}

template <class T>
template <class S>
inline Matrix44<T> &
Matrix44<T>::setTheMatrix (const Matrix44<S> &v)
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[0][3] = v.x[0][3];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[1][3] = v.x[1][3];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    x[2][3] = v.x[2][3];
    x[3][0] = v.x[3][0];
    x[3][1] = v.x[3][1];
    x[3][2] = v.x[3][2];
    x[3][3] = v.x[3][3];

    return *this;
}

template <class T>
inline void
Matrix44<T>::makeIdentity()
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;
}

template <class T>
bool
Matrix44<T>::operator == (const Matrix44 &v) const
{
    return x[0][0] == v.x[0][0] &&
           x[0][1] == v.x[0][1] &&
           x[0][2] == v.x[0][2] &&
           x[0][3] == v.x[0][3] &&
           x[1][0] == v.x[1][0] &&
           x[1][1] == v.x[1][1] &&
           x[1][2] == v.x[1][2] &&
           x[1][3] == v.x[1][3] &&
           x[2][0] == v.x[2][0] &&
           x[2][1] == v.x[2][1] &&
           x[2][2] == v.x[2][2] &&
           x[2][3] == v.x[2][3] &&
           x[3][0] == v.x[3][0] &&
           x[3][1] == v.x[3][1] &&
           x[3][2] == v.x[3][2] &&
           x[3][3] == v.x[3][3];
}

template <class T>
bool
Matrix44<T>::operator != (const Matrix44 &v) const
{
    return x[0][0] != v.x[0][0] ||
           x[0][1] != v.x[0][1] ||
           x[0][2] != v.x[0][2] ||
           x[0][3] != v.x[0][3] ||
           x[1][0] != v.x[1][0] ||
           x[1][1] != v.x[1][1] ||
           x[1][2] != v.x[1][2] ||
           x[1][3] != v.x[1][3] ||
           x[2][0] != v.x[2][0] ||
           x[2][1] != v.x[2][1] ||
           x[2][2] != v.x[2][2] ||
           x[2][3] != v.x[2][3] ||
           x[3][0] != v.x[3][0] ||
           x[3][1] != v.x[3][1] ||
           x[3][2] != v.x[3][2] ||
           x[3][3] != v.x[3][3];
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator += (const Matrix44<T> &v)
{
    x[0][0] += v.x[0][0];
    x[0][1] += v.x[0][1];
    x[0][2] += v.x[0][2];
    x[0][3] += v.x[0][3];
    x[1][0] += v.x[1][0];
    x[1][1] += v.x[1][1];
    x[1][2] += v.x[1][2];
    x[1][3] += v.x[1][3];
    x[2][0] += v.x[2][0];
    x[2][1] += v.x[2][1];
    x[2][2] += v.x[2][2];
    x[2][3] += v.x[2][3];
    x[3][0] += v.x[3][0];
    x[3][1] += v.x[3][1];
    x[3][2] += v.x[3][2];
    x[3][3] += v.x[3][3];

    return *this;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator += (T a)
{
    x[0][0] += a;
    x[0][1] += a;
    x[0][2] += a;
    x[0][3] += a;
    x[1][0] += a;
    x[1][1] += a;
    x[1][2] += a;
    x[1][3] += a;
    x[2][0] += a;
    x[2][1] += a;
    x[2][2] += a;
    x[2][3] += a;
    x[3][0] += a;
    x[3][1] += a;
    x[3][2] += a;
    x[3][3] += a;

    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::operator + (const Matrix44<T> &v) const
{
    return Matrix44 (x[0][0] + v.x[0][0],
                     x[0][1] + v.x[0][1],
                     x[0][2] + v.x[0][2],
                     x[0][3] + v.x[0][3],
                     x[1][0] + v.x[1][0],
                     x[1][1] + v.x[1][1],
                     x[1][2] + v.x[1][2],
                     x[1][3] + v.x[1][3],
                     x[2][0] + v.x[2][0],
                     x[2][1] + v.x[2][1],
                     x[2][2] + v.x[2][2],
                     x[2][3] + v.x[2][3],
                     x[3][0] + v.x[3][0],
                     x[3][1] + v.x[3][1],
                     x[3][2] + v.x[3][2],
                     x[3][3] + v.x[3][3]);
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator -= (const Matrix44<T> &v)
{
    x[0][0] -= v.x[0][0];
    x[0][1] -= v.x[0][1];
    x[0][2] -= v.x[0][2];
    x[0][3] -= v.x[0][3];
    x[1][0] -= v.x[1][0];
    x[1][1] -= v.x[1][1];
    x[1][2] -= v.x[1][2];
    x[1][3] -= v.x[1][3];
    x[2][0] -= v.x[2][0];
    x[2][1] -= v.x[2][1];
    x[2][2] -= v.x[2][2];
    x[2][3] -= v.x[2][3];
    x[3][0] -= v.x[3][0];
    x[3][1] -= v.x[3][1];
    x[3][2] -= v.x[3][2];
    x[3][3] -= v.x[3][3];

    return *this;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator -= (T a)
{
    x[0][0] -= a;
    x[0][1] -= a;
    x[0][2] -= a;
    x[0][3] -= a;
    x[1][0] -= a;
    x[1][1] -= a;
    x[1][2] -= a;
    x[1][3] -= a;
    x[2][0] -= a;
    x[2][1] -= a;
    x[2][2] -= a;
    x[2][3] -= a;
    x[3][0] -= a;
    x[3][1] -= a;
    x[3][2] -= a;
    x[3][3] -= a;

    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::operator - (const Matrix44<T> &v) const
{
    return Matrix44 (x[0][0] - v.x[0][0],
                     x[0][1] - v.x[0][1],
                     x[0][2] - v.x[0][2],
                     x[0][3] - v.x[0][3],
                     x[1][0] - v.x[1][0],
                     x[1][1] - v.x[1][1],
                     x[1][2] - v.x[1][2],
                     x[1][3] - v.x[1][3],
                     x[2][0] - v.x[2][0],
                     x[2][1] - v.x[2][1],
                     x[2][2] - v.x[2][2],
                     x[2][3] - v.x[2][3],
                     x[3][0] - v.x[3][0],
                     x[3][1] - v.x[3][1],
                     x[3][2] - v.x[3][2],
                     x[3][3] - v.x[3][3]);
}

template <class T>
Matrix44<T>
Matrix44<T>::operator - () const
{
    return Matrix44 (-x[0][0],
                     -x[0][1],
                     -x[0][2],
                     -x[0][3],
                     -x[1][0],
                     -x[1][1],
                     -x[1][2],
                     -x[1][3],
                     -x[2][0],
                     -x[2][1],
                     -x[2][2],
                     -x[2][3],
                     -x[3][0],
                     -x[3][1],
                     -x[3][2],
                     -x[3][3]);
}

template <class T>
const Matrix44<T> &
Matrix44<T>::negate ()
{
    x[0][0] = -x[0][0];
    x[0][1] = -x[0][1];
    x[0][2] = -x[0][2];
    x[0][3] = -x[0][3];
    x[1][0] = -x[1][0];
    x[1][1] = -x[1][1];
    x[1][2] = -x[1][2];
    x[1][3] = -x[1][3];
    x[2][0] = -x[2][0];
    x[2][1] = -x[2][1];
    x[2][2] = -x[2][2];
    x[2][3] = -x[2][3];
    x[3][0] = -x[3][0];
    x[3][1] = -x[3][1];
    x[3][2] = -x[3][2];
    x[3][3] = -x[3][3];

    return *this;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator *= (T a)
{
    x[0][0] *= a;
    x[0][1] *= a;
    x[0][2] *= a;
    x[0][3] *= a;
    x[1][0] *= a;
    x[1][1] *= a;
    x[1][2] *= a;
    x[1][3] *= a;
    x[2][0] *= a;
    x[2][1] *= a;
    x[2][2] *= a;
    x[2][3] *= a;
    x[3][0] *= a;
    x[3][1] *= a;
    x[3][2] *= a;
    x[3][3] *= a;

    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::operator * (T a) const
{
    return Matrix44 (x[0][0] * a,
                     x[0][1] * a,
                     x[0][2] * a,
                     x[0][3] * a,
                     x[1][0] * a,
                     x[1][1] * a,
                     x[1][2] * a,
                     x[1][3] * a,
                     x[2][0] * a,
                     x[2][1] * a,
                     x[2][2] * a,
                     x[2][3] * a,
                     x[3][0] * a,
                     x[3][1] * a,
                     x[3][2] * a,
                     x[3][3] * a);
}

template <class T>
inline Matrix44<T>
operator * (T a, const Matrix44<T> &v)
{
    return v * a;
}

template <class T>
inline const Matrix44<T> &
Matrix44<T>::operator *= (const Matrix44<T> &v)
{
    Matrix44 tmp (T (0));

    multiply (*this, v, tmp);
    *this = tmp;
    return *this;
}

template <class T>
inline Matrix44<T>
Matrix44<T>::operator * (const Matrix44<T> &v) const
{
    Matrix44 tmp (T (0));

    multiply (*this, v, tmp);
    return tmp;
}

template <class T>
void
Matrix44<T>::multiply (const Matrix44<T> &a,
                       const Matrix44<T> &b,
                       Matrix44<T> &c)
{
    register const T * ap = &a.x[0][0];
    register const T * bp = &b.x[0][0];
    register       T * cp = &c.x[0][0];

    register T a0, a1, a2, a3;

    a0 = ap[0];
    a1 = ap[1];
    a2 = ap[2];
    a3 = ap[3];

    cp[0]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[1]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[2]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[3]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[4];
    a1 = ap[5];
    a2 = ap[6];
    a3 = ap[7];

    cp[4]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[5]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[6]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[7]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[8];
    a1 = ap[9];
    a2 = ap[10];
    a3 = ap[11];

    cp[8]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[9]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[10] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[11] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[12];
    a1 = ap[13];
    a2 = ap[14];
    a3 = ap[15];

    cp[12] = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[13] = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[14] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[15] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];
}

template <class T> template <class S>
void
Matrix44<T>::multVecMatrix(const Vec3<S> &src, Vec3<S> &dst) const
{
    S a, b, c, w;

    a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0] + x[3][0];
    b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1] + x[3][1];
    c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2] + x[3][2];
    w = src[0] * x[0][3] + src[1] * x[1][3] + src[2] * x[2][3] + x[3][3];

    dst.x = a / w;
    dst.y = b / w;
    dst.z = c / w;
}

template <class T> template <class S>
void
Matrix44<T>::multDirMatrix(const Vec3<S> &src, Vec3<S> &dst) const
{
    S a, b, c;

    a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0];
    b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1];
    c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2];

    dst.x = a;
    dst.y = b;
    dst.z = c;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator /= (T a)
{
    x[0][0] /= a;
    x[0][1] /= a;
    x[0][2] /= a;
    x[0][3] /= a;
    x[1][0] /= a;
    x[1][1] /= a;
    x[1][2] /= a;
    x[1][3] /= a;
    x[2][0] /= a;
    x[2][1] /= a;
    x[2][2] /= a;
    x[2][3] /= a;
    x[3][0] /= a;
    x[3][1] /= a;
    x[3][2] /= a;
    x[3][3] /= a;

    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::operator / (T a) const
{
    return Matrix44 (x[0][0] / a,
                     x[0][1] / a,
                     x[0][2] / a,
                     x[0][3] / a,
                     x[1][0] / a,
                     x[1][1] / a,
                     x[1][2] / a,
                     x[1][3] / a,
                     x[2][0] / a,
                     x[2][1] / a,
                     x[2][2] / a,
                     x[2][3] / a,
                     x[3][0] / a,
                     x[3][1] / a,
                     x[3][2] / a,
                     x[3][3] / a);
}

template <class T>
const Matrix44<T> &
Matrix44<T>::transpose ()
{
    Matrix44 tmp (x[0][0],
                  x[1][0],
                  x[2][0],
                  x[3][0],
                  x[0][1],
                  x[1][1],
                  x[2][1],
                  x[3][1],
                  x[0][2],
                  x[1][2],
                  x[2][2],
                  x[3][2],
                  x[0][3],
                  x[1][3],
                  x[2][3],
                  x[3][3]);
    *this = tmp;
    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::transposed () const
{
    return Matrix44 (x[0][0],
                     x[1][0],
                     x[2][0],
                     x[3][0],
                     x[0][1],
                     x[1][1],
                     x[2][1],
                     x[3][1],
                     x[0][2],
                     x[1][2],
                     x[2][2],
                     x[3][2],
                     x[0][3],
                     x[1][3],
                     x[2][3],
                     x[3][3]);
}

template <class T>
const Matrix44<T> &
Matrix44<T>::gjInvert (bool singExc)
{
    *this = gjInverse (singExc);
    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::gjInverse (bool singExc) const
{
    int i, j, k;
    Matrix44 s;
    Matrix44 t (*this);

    // Forward elimination

    for (i = 0; i < 3 ; i++)
    {
        int pivot = i;

        T pivotsize = t[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (j = i + 1; j < 4; j++)
        {
            T tmp = t[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            return Matrix44();
        }

        if (pivot != i)
        {
            for (j = 0; j < 4; j++)
            {
                T tmp;

                tmp = t[i][j];
                t[i][j] = t[pivot][j];
                t[pivot][j] = tmp;

                tmp = s[i][j];
                s[i][j] = s[pivot][j];
                s[pivot][j] = tmp;
            }
        }

        for (j = i + 1; j < 4; j++)
        {
            T f = t[j][i] / t[i][i];

            for (k = 0; k < 4; k++)
            {
                t[j][k] -= f * t[i][k];
                s[j][k] -= f * s[i][k];
            }
        }
    }

    // Backward substitution

    for (i = 3; i >= 0; --i)
    {
        T f;

        if ((f = t[i][i]) == 0)
        {
            return Matrix44();
        }

        for (j = 0; j < 4; j++)
        {
            t[i][j] /= f;
            s[i][j] /= f;
        }

        for (j = 0; j < i; j++)
        {
            f = t[j][i];

            for (k = 0; k < 4; k++)
            {
                t[j][k] -= f * t[i][k];
                s[j][k] -= f * s[i][k];
            }
        }
    }

    return s;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::invert (bool singExc)
{
    *this = inverse (singExc);
    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::inverse (bool singExc) const
{
    if (x[0][3] != 0 || x[1][3] != 0 || x[2][3] != 0 || x[3][3] != 1)
        return gjInverse(singExc);

    Matrix44 s (x[1][1] * x[2][2] - x[2][1] * x[1][2],
                x[2][1] * x[0][2] - x[0][1] * x[2][2],
                x[0][1] * x[1][2] - x[1][1] * x[0][2],
                0,

                x[2][0] * x[1][2] - x[1][0] * x[2][2],
                x[0][0] * x[2][2] - x[2][0] * x[0][2],
                x[1][0] * x[0][2] - x[0][0] * x[1][2],
                0,

                x[1][0] * x[2][1] - x[2][0] * x[1][1],
                x[2][0] * x[0][1] - x[0][0] * x[2][1],
                x[0][0] * x[1][1] - x[1][0] * x[0][1],
                0,

                0,
                0,
                0,
                1);

    T r = x[0][0] * s[0][0] + x[0][1] * s[1][0] + x[0][2] * s[2][0];

    if (Math<T>::fabs (r) >= 1)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                s[i][j] /= r;
            }
        }
    }
    else
    {
        T mr = Math<T>::fabs (r) / limits<T>::smallest();

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (mr > Math<T>::fabs (s[i][j]))
                {
                    s[i][j] /= r;
                }
                else
                {
                    return Matrix44();
                }
            }
        }
    }

    s[3][0] = -x[3][0] * s[0][0] - x[3][1] * s[1][0] - x[3][2] * s[2][0];
    s[3][1] = -x[3][0] * s[0][1] - x[3][1] * s[1][1] - x[3][2] * s[2][1];
    s[3][2] = -x[3][0] * s[0][2] - x[3][1] * s[1][2] - x[3][2] * s[2][2];

    return s;
}

template <class T>
inline T
Matrix44<T>::fastMinor( const int r0, const int r1, const int r2,
                        const int c0, const int c1, const int c2) const
{
    return x[r0][c0] * (x[r1][c1]*x[r2][c2] - x[r1][c2]*x[r2][c1])
         + x[r0][c1] * (x[r1][c2]*x[r2][c0] - x[r1][c0]*x[r2][c2])
         + x[r0][c2] * (x[r1][c0]*x[r2][c1] - x[r1][c1]*x[r2][c0]);
}

template <class T>
inline T
Matrix44<T>::determinant () const
{
    T sum = (T)0;

    if (x[0][3] != 0.) sum -= x[0][3] * fastMinor(1,2,3,0,1,2);
    if (x[1][3] != 0.) sum += x[1][3] * fastMinor(0,2,3,0,1,2);
    if (x[2][3] != 0.) sum -= x[2][3] * fastMinor(0,1,3,0,1,2);
    if (x[3][3] != 0.) sum += x[3][3] * fastMinor(0,1,2,0,1,2);

    return sum;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setEulerAngles (const Vec3<S>& r)
{
    S cos_rz, sin_rz, cos_ry, sin_ry, cos_rx, sin_rx;
    
    cos_rz = Math<T>::cos (r[2]);
    cos_ry = Math<T>::cos (r[1]);
    cos_rx = Math<T>::cos (r[0]);
    
    sin_rz = Math<T>::sin (r[2]);
    sin_ry = Math<T>::sin (r[1]);
    sin_rx = Math<T>::sin (r[0]);
    
    x[0][0] =  cos_rz * cos_ry;
    x[0][1] =  sin_rz * cos_ry;
    x[0][2] = -sin_ry;
    x[0][3] =  0;
    
    x[1][0] = -sin_rz * cos_rx + cos_rz * sin_ry * sin_rx;
    x[1][1] =  cos_rz * cos_rx + sin_rz * sin_ry * sin_rx;
    x[1][2] =  cos_ry * sin_rx;
    x[1][3] =  0;
    
    x[2][0] =  sin_rz * sin_rx + cos_rz * sin_ry * cos_rx;
    x[2][1] = -cos_rz * sin_rx + sin_rz * sin_ry * cos_rx;
    x[2][2] =  cos_ry * cos_rx;
    x[2][3] =  0;

    x[3][0] =  0;
    x[3][1] =  0;
    x[3][2] =  0;
    x[3][3] =  1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setAxisAngle (const Vec3<S>& axis, S angle)
{
    Vec3<S> unit (axis.normalized());
    S sine   = Math<T>::sin (angle);
    S cosine = Math<T>::cos (angle);

    x[0][0] = unit[0] * unit[0] * (1 - cosine) + cosine;
    x[0][1] = unit[0] * unit[1] * (1 - cosine) + unit[2] * sine;
    x[0][2] = unit[0] * unit[2] * (1 - cosine) - unit[1] * sine;
    x[0][3] = 0;

    x[1][0] = unit[0] * unit[1] * (1 - cosine) - unit[2] * sine;
    x[1][1] = unit[1] * unit[1] * (1 - cosine) + cosine;
    x[1][2] = unit[1] * unit[2] * (1 - cosine) + unit[0] * sine;
    x[1][3] = 0;

    x[2][0] = unit[0] * unit[2] * (1 - cosine) + unit[1] * sine;
    x[2][1] = unit[1] * unit[2] * (1 - cosine) - unit[0] * sine;
    x[2][2] = unit[2] * unit[2] * (1 - cosine) + cosine;
    x[2][3] = 0;

    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::rotate (const Vec3<S> &r)
{
    S cos_rz, sin_rz, cos_ry, sin_ry, cos_rx, sin_rx;
    S m00, m01, m02;
    S m10, m11, m12;
    S m20, m21, m22;

    cos_rz = Math<S>::cos (r[2]);
    cos_ry = Math<S>::cos (r[1]);
    cos_rx = Math<S>::cos (r[0]);
    
    sin_rz = Math<S>::sin (r[2]);
    sin_ry = Math<S>::sin (r[1]);
    sin_rx = Math<S>::sin (r[0]);

    m00 =  cos_rz *  cos_ry;
    m01 =  sin_rz *  cos_ry;
    m02 = -sin_ry;
    m10 = -sin_rz *  cos_rx + cos_rz * sin_ry * sin_rx;
    m11 =  cos_rz *  cos_rx + sin_rz * sin_ry * sin_rx;
    m12 =  cos_ry *  sin_rx;
    m20 = -sin_rz * -sin_rx + cos_rz * sin_ry * cos_rx;
    m21 =  cos_rz * -sin_rx + sin_rz * sin_ry * cos_rx;
    m22 =  cos_ry *  cos_rx;

    Matrix44<T> P (*this);

    x[0][0] = P[0][0] * m00 + P[1][0] * m01 + P[2][0] * m02;
    x[0][1] = P[0][1] * m00 + P[1][1] * m01 + P[2][1] * m02;
    x[0][2] = P[0][2] * m00 + P[1][2] * m01 + P[2][2] * m02;
    x[0][3] = P[0][3] * m00 + P[1][3] * m01 + P[2][3] * m02;

    x[1][0] = P[0][0] * m10 + P[1][0] * m11 + P[2][0] * m12;
    x[1][1] = P[0][1] * m10 + P[1][1] * m11 + P[2][1] * m12;
    x[1][2] = P[0][2] * m10 + P[1][2] * m11 + P[2][2] * m12;
    x[1][3] = P[0][3] * m10 + P[1][3] * m11 + P[2][3] * m12;

    x[2][0] = P[0][0] * m20 + P[1][0] * m21 + P[2][0] * m22;
    x[2][1] = P[0][1] * m20 + P[1][1] * m21 + P[2][1] * m22;
    x[2][2] = P[0][2] * m20 + P[1][2] * m21 + P[2][2] * m22;
    x[2][3] = P[0][3] * m20 + P[1][3] * m21 + P[2][3] * m22;

    return *this;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::setScale (T s)
{
    x[0][0] = s;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = s;
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = s;
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setScale (const Vec3<S> &s)
{
    x[0][0] = s[0];
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = s[1];
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = s[2];
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::scale (const Vec3<S> &s)
{
    x[0][0] *= s[0];
    x[0][1] *= s[0];
    x[0][2] *= s[0];
    x[0][3] *= s[0];

    x[1][0] *= s[1];
    x[1][1] *= s[1];
    x[1][2] *= s[1];
    x[1][3] *= s[1];

    x[2][0] *= s[2];
    x[2][1] *= s[2];
    x[2][2] *= s[2];
    x[2][3] *= s[2];

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setTranslation (const Vec3<S> &t)
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;

    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;

    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    x[2][3] = 0;

    x[3][0] = t[0];
    x[3][1] = t[1];
    x[3][2] = t[2];
    x[3][3] = 1;

    return *this;
}

template <class T>
inline const Vec3<T>
Matrix44<T>::translation () const
{
    return Vec3<T> (x[3][0], x[3][1], x[3][2]);
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::translate (const Vec3<S> &t)
{
    x[3][0] += t[0] * x[0][0] + t[1] * x[1][0] + t[2] * x[2][0];
    x[3][1] += t[0] * x[0][1] + t[1] * x[1][1] + t[2] * x[2][1];
    x[3][2] += t[0] * x[0][2] + t[1] * x[1][2] + t[2] * x[2][2];
    x[3][3] += t[0] * x[0][3] + t[1] * x[1][3] + t[2] * x[2][3];

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setShear (const Vec3<S> &h)
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;

    x[1][0] = h[0];
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;

    x[2][0] = h[1];
    x[2][1] = h[2];
    x[2][2] = 1;
    x[2][3] = 0;

    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::shear (const Vec3<S> &h)
{
    //
    // In this case, we don't need a temp. copy of the matrix 
    // because we never use a value on the RHS after we've 
    // changed it on the LHS.
    // 

    for (int i=0; i < 4; i++)
    {
        x[2][i] += h[1] * x[0][i] + h[2] * x[1][i];
        x[1][i] += h[0] * x[0][i];
    }

    return *this;
}


//---------------------------------------------------------------
// Implementation of vector-times-matrix multiplication operators
//---------------------------------------------------------------

template <class S, class T>
inline const Vec3<S> &
operator *= (Vec3<S> &v, const Matrix44<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0] + m[3][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1] + m[3][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2] + m[3][2]);
    S w = S(v.x * m[0][3] + v.y * m[1][3] + v.z * m[2][3] + m[3][3]);

    v.x = x / w;
    v.y = y / w;
    v.z = z / w;

    return v;
}

template <class S, class T>
inline Vec3<S>
operator * (const Vec3<S> &v, const Matrix44<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0] + m[3][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1] + m[3][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2] + m[3][2]);
    S w = S(v.x * m[0][3] + v.y * m[1][3] + v.z * m[2][3] + m[3][3]);

    return Vec3<S> (x / w, y / w, z / w);
}

}

typedef Imath::Vec3<float>     Vec3;
typedef Imath::Matrix44<float> Matrix44;

