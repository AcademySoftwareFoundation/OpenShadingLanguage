///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004-2012, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////



#ifndef INCLUDED_IMATHVEC_H
#define INCLUDED_IMATHVEC_H

//----------------------------------------------------
//
// 2D, 3D and 4D point/vector class templates
//
//----------------------------------------------------

#include <OpenEXR/ImathExc.h>
#include "ImathLimits.h"
#include <OpenEXR/ImathMath.h>
#include <OpenEXR/ImathNamespace.h>

#include <iostream>
#include <limits>

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// suppress exception specification warnings
#pragma warning(push)
#pragma warning(disable:4290)
#endif

#ifndef IMATH_HOSTDEVICE
    #ifdef __CUDACC__
        #define IMATH_HOSTDEVICE __host__ __device__
    #else
        #define IMATH_HOSTDEVICE
    #endif
#endif


IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

template <class T> class Vec2;
template <class T> class Vec3;
template <class T> class Vec4;

enum InfException {INF_EXCEPTION};

namespace {
// Define a host/device-compatible sqrt template that matches Math<T>::sqrt.
template <typename T> IMATH_HOSTDEVICE inline T hostdevice_sqrt (T x);
template <> IMATH_HOSTDEVICE inline double hostdevice_sqrt<double> (double x) { return sqrt (x); }
template <> IMATH_HOSTDEVICE inline float  hostdevice_sqrt<float> (float x)  { return sqrtf (x); }
};


template <class T> class Vec2
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T x, y;

    IMATH_HOSTDEVICE       T & operator [] (int i);
    IMATH_HOSTDEVICE const T & operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    IMATH_HOSTDEVICE          Vec2 ();             // no initialization
    IMATH_HOSTDEVICE explicit Vec2 (T a);          // (a a)
    IMATH_HOSTDEVICE          Vec2 (T a, T b);     // (a b)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    IMATH_HOSTDEVICE Vec2 (const Vec2 &v);
    template <class S> IMATH_HOSTDEVICE Vec2 (const Vec2<S> &v);

    IMATH_HOSTDEVICE const Vec2 & operator = (const Vec2 &v);

    //------------
    // Destructor
    //------------
	
    ~Vec2 () = default;

    //----------------------
    // Compatibility with Sb
    //----------------------

    template <class S> IMATH_HOSTDEVICE
    void        setValue (S a, S b);

    template <class S> IMATH_HOSTDEVICE
    void        setValue (const Vec2<S> &v);

    template <class S> IMATH_HOSTDEVICE
    void        getValue (S &a, S &b) const;

    template <class S> IMATH_HOSTDEVICE
    void        getValue (Vec2<S> &v) const;

    IMATH_HOSTDEVICE       T * getValue ();
    IMATH_HOSTDEVICE const T * getValue () const;


    //---------
    // Equality
    //---------

    template <class S> IMATH_HOSTDEVICE
    bool        operator == (const Vec2<S> &v) const;

    template <class S> IMATH_HOSTDEVICE
    bool        operator != (const Vec2<S> &v) const;


    //-----------------------------------------------------------------------
    // Compare two vectors and test if they are "approximately equal":
    //
    // equalWithAbsError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      an absolute error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e
    //
    // equalWithRelError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      a relative error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e * abs (this[i])
    //-----------------------------------------------------------------------

    IMATH_HOSTDEVICE bool equalWithAbsError (const Vec2<T> &v, T e) const;
    IMATH_HOSTDEVICE bool equalWithRelError (const Vec2<T> &v, T e) const;

    //------------
    // Dot product
    //------------

    IMATH_HOSTDEVICE T dot (const Vec2 &v) const;
    IMATH_HOSTDEVICE T operator ^ (const Vec2 &v) const;


    //------------------------------------------------
    // Right-handed cross product, i.e. z component of
    // Vec3 (this->x, this->y, 0) % Vec3 (v.x, v.y, 0)
    //------------------------------------------------

    IMATH_HOSTDEVICE T cross (const Vec2 &v) const;
    IMATH_HOSTDEVICE T operator % (const Vec2 &v) const;


    //------------------------
    // Component-wise addition
    //------------------------

    IMATH_HOSTDEVICE const Vec2 & operator += (const Vec2 &v);
    IMATH_HOSTDEVICE Vec2 operator + (const Vec2 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    IMATH_HOSTDEVICE const Vec2 & operator -= (const Vec2 &v);
    IMATH_HOSTDEVICE Vec2 operator - (const Vec2 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    IMATH_HOSTDEVICE Vec2 operator - () const;
    IMATH_HOSTDEVICE const Vec2 & negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    IMATH_HOSTDEVICE const Vec2 & operator *= (const Vec2 &v);
    IMATH_HOSTDEVICE const Vec2 & operator *= (T a);
    IMATH_HOSTDEVICE Vec2     operator * (const Vec2 &v) const;
    IMATH_HOSTDEVICE Vec2     operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    IMATH_HOSTDEVICE const Vec2 & operator /= (const Vec2 &v);
    IMATH_HOSTDEVICE const Vec2 & operator /= (T a);
    IMATH_HOSTDEVICE Vec2     operator / (const Vec2 &v) const;
    IMATH_HOSTDEVICE Vec2     operator / (T a) const;


    //----------------------------------------------------------------
    // Length and normalization:  If v.length() is 0.0, v.normalize()
    // and v.normalized() produce a null vector; v.normalizeExc() and
    // v.normalizedExc() throw a NullVecExc.
    // v.normalizeNonNull() and v.normalizedNonNull() are slightly
    // faster than the other normalization routines, but if v.length()
    // is 0.0, the result is undefined.
    //----------------------------------------------------------------

    IMATH_HOSTDEVICE T length () const;
    IMATH_HOSTDEVICE T length2 () const;

    IMATH_HOSTDEVICE const Vec2 & normalize ();           // modifies *this
    IMATH_HOSTDEVICE const Vec2 & normalizeExc ();
    IMATH_HOSTDEVICE const Vec2 & normalizeNonNull ();

    IMATH_HOSTDEVICE Vec2<T> normalized () const;        // does not modify *this
    IMATH_HOSTDEVICE Vec2<T> normalizedExc () const;
    IMATH_HOSTDEVICE Vec2<T> normalizedNonNull () const;


    //--------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Vec2
    //--------------------------------------------------------

    IMATH_HOSTDEVICE static unsigned int  dimensions() {return 2;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    IMATH_HOSTDEVICE static T baseTypeMin()       {return limits<T>::min();}
    IMATH_HOSTDEVICE static T baseTypeMax()       {return limits<T>::max();}
    IMATH_HOSTDEVICE static T baseTypeSmallest()  {return limits<T>::smallest();}
    IMATH_HOSTDEVICE static T baseTypeEpsilon()   {return limits<T>::epsilon();}


    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be either a Vec2<T>, a Vec3<T>, or a Vec4<T> you can
    // refer to T as V::BaseType
    //--------------------------------------------------------------

    typedef T       BaseType;

  private:

    IMATH_HOSTDEVICE T lengthTiny () const;
};


template <class T> class Vec3
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T x, y, z;

    IMATH_HOSTDEVICE       T & operator [] (int i);
    IMATH_HOSTDEVICE const T & operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    IMATH_HOSTDEVICE Vec3 ();            // no initialization
    IMATH_HOSTDEVICE explicit Vec3 (T a);           // (a a a)
    IMATH_HOSTDEVICE Vec3 (T a, T b, T c);       // (a b c)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    IMATH_HOSTDEVICE Vec3 (const Vec3 &v);
    template <class S> IMATH_HOSTDEVICE Vec3 (const Vec3<S> &v);

    IMATH_HOSTDEVICE const Vec3 & operator = (const Vec3 &v);

    //-----------
    // Destructor
    //-----------
	
    ~Vec3 () = default;

    //---------------------------------------------------------
    // Vec4 to Vec3 conversion, divides x, y and z by w:
    //
    // The one-argument conversion function divides by w even
    // if w is zero.  The result depends on how the environment
    // handles floating-point exceptions.
    //
    // The two-argument version thows an InfPointExc exception
    // if w is zero or if division by w would overflow.
    //---------------------------------------------------------

    template <class S> IMATH_HOSTDEVICE explicit Vec3 (const Vec4<S> &v);
    template <class S> IMATH_HOSTDEVICE explicit Vec3 (const Vec4<S> &v, InfException);


    //----------------------
    // Compatibility with Sb
    //----------------------

    template <class S> IMATH_HOSTDEVICE
    void      setValue (S a, S b, S c);

    template <class S> IMATH_HOSTDEVICE
    void      setValue (const Vec3<S> &v);

    template <class S> IMATH_HOSTDEVICE
    void      getValue (S &a, S &b, S &c) const;

    template <class S> IMATH_HOSTDEVICE
    void      getValue (Vec3<S> &v) const;

    IMATH_HOSTDEVICE T *       getValue();
    IMATH_HOSTDEVICE const T * getValue() const;


    //---------
    // Equality
    //---------

    template <class S> IMATH_HOSTDEVICE
    bool operator == (const Vec3<S> &v) const;

    template <class S> IMATH_HOSTDEVICE
    bool operator != (const Vec3<S> &v) const;

    //-----------------------------------------------------------------------
    // Compare two vectors and test if they are "approximately equal":
    //
    // equalWithAbsError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      an absolute error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e
    //
    // equalWithRelError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      a relative error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e * abs (this[i])
    //-----------------------------------------------------------------------

    IMATH_HOSTDEVICE bool equalWithAbsError (const Vec3<T> &v, T e) const;
    IMATH_HOSTDEVICE bool equalWithRelError (const Vec3<T> &v, T e) const;

    //------------
    // Dot product
    //------------

    IMATH_HOSTDEVICE T dot (const Vec3 &v) const;
    IMATH_HOSTDEVICE T operator ^ (const Vec3 &v) const;


    //---------------------------
    // Right-handed cross product
    //---------------------------

    IMATH_HOSTDEVICE Vec3     cross (const Vec3 &v) const;
    IMATH_HOSTDEVICE const Vec3 & operator %= (const Vec3 &v);
    IMATH_HOSTDEVICE Vec3     operator % (const Vec3 &v) const;


    //------------------------
    // Component-wise addition
    //------------------------

    IMATH_HOSTDEVICE const Vec3 & operator += (const Vec3 &v);
    IMATH_HOSTDEVICE Vec3     operator + (const Vec3 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    IMATH_HOSTDEVICE const Vec3 & operator -= (const Vec3 &v);
    IMATH_HOSTDEVICE Vec3     operator - (const Vec3 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    IMATH_HOSTDEVICE Vec3     operator - () const;
    IMATH_HOSTDEVICE const Vec3 & negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    IMATH_HOSTDEVICE const Vec3 & operator *= (const Vec3 &v);
    IMATH_HOSTDEVICE const Vec3 & operator *= (T a);
    IMATH_HOSTDEVICE Vec3     operator * (const Vec3 &v) const;
    IMATH_HOSTDEVICE Vec3     operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    IMATH_HOSTDEVICE const Vec3 & operator /= (const Vec3 &v);
    IMATH_HOSTDEVICE const Vec3 & operator /= (T a);
    IMATH_HOSTDEVICE Vec3     operator / (const Vec3 &v) const;
    IMATH_HOSTDEVICE Vec3     operator / (T a) const;


    //----------------------------------------------------------------
    // Length and normalization:  If v.length() is 0.0, v.normalize()
    // and v.normalized() produce a null vector; v.normalizeExc() and
    // v.normalizedExc() throw a NullVecExc.
    // v.normalizeNonNull() and v.normalizedNonNull() are slightly
    // faster than the other normalization routines, but if v.length()
    // is 0.0, the result is undefined.
    //----------------------------------------------------------------

    IMATH_HOSTDEVICE T length () const;
    IMATH_HOSTDEVICE T length2 () const;

    IMATH_HOSTDEVICE const Vec3 & normalize ();           // modifies *this
    IMATH_HOSTDEVICE const Vec3 & normalizeExc ();
    IMATH_HOSTDEVICE const Vec3 & normalizeNonNull ();

    IMATH_HOSTDEVICE Vec3<T> normalized () const; // does not modify *this
    IMATH_HOSTDEVICE Vec3<T> normalizedExc () const;
    IMATH_HOSTDEVICE Vec3<T> normalizedNonNull () const;


    //--------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Vec3
    //--------------------------------------------------------

    IMATH_HOSTDEVICE static unsigned int  dimensions() {return 3;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    IMATH_HOSTDEVICE static T baseTypeMin()       {return limits<T>::min();}
    IMATH_HOSTDEVICE static T baseTypeMax()       {return limits<T>::max();}
    IMATH_HOSTDEVICE static T baseTypeSmallest()  {return limits<T>::smallest();}
    IMATH_HOSTDEVICE static T baseTypeEpsilon()   {return limits<T>::epsilon();}


    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be either a Vec2<T>, a Vec3<T>, or a Vec4<T> you can
    // refer to T as V::BaseType
    //--------------------------------------------------------------

    typedef T       BaseType;

  private:

    IMATH_HOSTDEVICE T lengthTiny () const;
};



template <class T> class Vec4
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T               x, y, z, w;

    IMATH_HOSTDEVICE       T & operator [] (int i);
    IMATH_HOSTDEVICE const T & operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    IMATH_HOSTDEVICE          Vec4 ();                       // no initialization
    IMATH_HOSTDEVICE explicit Vec4 (T a);                    // (a a a a)
    IMATH_HOSTDEVICE          Vec4 (T a, T b, T c, T d);     // (a b c d)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    IMATH_HOSTDEVICE Vec4 (const Vec4 &v);
    template <class S> IMATH_HOSTDEVICE Vec4 (const Vec4<S> &v);

    IMATH_HOSTDEVICE const Vec4 & operator = (const Vec4 &v);

    //-----------
    // Destructor
    //-----------
	
    ~Vec4 () = default;

    //-------------------------------------
    // Vec3 to Vec4 conversion, sets w to 1
    //-------------------------------------

    template <class S> IMATH_HOSTDEVICE explicit Vec4 (const Vec3<S> &v);


    //---------
    // Equality
    //---------

    template <class S>
    IMATH_HOSTDEVICE bool operator == (const Vec4<S> &v) const;

    template <class S>
    IMATH_HOSTDEVICE bool operator != (const Vec4<S> &v) const;


    //-----------------------------------------------------------------------
    // Compare two vectors and test if they are "approximately equal":
    //
    // equalWithAbsError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      an absolute error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e
    //
    // equalWithRelError (v, e)
    //
    //      Returns true if the coefficients of this and v are the same with
    //      a relative error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e * abs (this[i])
    //-----------------------------------------------------------------------

    IMATH_HOSTDEVICE bool equalWithAbsError (const Vec4<T> &v, T e) const;
    IMATH_HOSTDEVICE bool equalWithRelError (const Vec4<T> &v, T e) const;


    //------------
    // Dot product
    //------------

    IMATH_HOSTDEVICE T dot (const Vec4 &v) const;
    IMATH_HOSTDEVICE T operator ^ (const Vec4 &v) const;


    //-----------------------------------
    // Cross product is not defined in 4D
    //-----------------------------------

    //------------------------
    // Component-wise addition
    //------------------------

    IMATH_HOSTDEVICE const Vec4 &    operator += (const Vec4 &v);
    IMATH_HOSTDEVICE Vec4            operator + (const Vec4 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    IMATH_HOSTDEVICE const Vec4 &    operator -= (const Vec4 &v);
    IMATH_HOSTDEVICE Vec4            operator - (const Vec4 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    IMATH_HOSTDEVICE Vec4            operator - () const;
    IMATH_HOSTDEVICE const Vec4 &    negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    IMATH_HOSTDEVICE const Vec4 &    operator *= (const Vec4 &v);
    IMATH_HOSTDEVICE const Vec4 &    operator *= (T a);
    IMATH_HOSTDEVICE Vec4            operator * (const Vec4 &v) const;
    IMATH_HOSTDEVICE Vec4            operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    IMATH_HOSTDEVICE const Vec4 &    operator /= (const Vec4 &v);
    IMATH_HOSTDEVICE const Vec4 &    operator /= (T a);
    IMATH_HOSTDEVICE Vec4            operator / (const Vec4 &v) const;
    IMATH_HOSTDEVICE Vec4            operator / (T a) const;


    //----------------------------------------------------------------
    // Length and normalization:  If v.length() is 0.0, v.normalize()
    // and v.normalized() produce a null vector; v.normalizeExc() and
    // v.normalizedExc() throw a NullVecExc.
    // v.normalizeNonNull() and v.normalizedNonNull() are slightly
    // faster than the other normalization routines, but if v.length()
    // is 0.0, the result is undefined.
    //----------------------------------------------------------------

    IMATH_HOSTDEVICE T               length () const;
    IMATH_HOSTDEVICE T               length2 () const;

    IMATH_HOSTDEVICE const Vec4 &    normalize ();           // modifies *this
    IMATH_HOSTDEVICE const Vec4 &    normalizeExc ();
    IMATH_HOSTDEVICE const Vec4 &    normalizeNonNull ();

    IMATH_HOSTDEVICE Vec4<T>         normalized () const;    // does not modify *this
    IMATH_HOSTDEVICE Vec4<T>         normalizedExc () const;
    IMATH_HOSTDEVICE Vec4<T>         normalizedNonNull () const;

    //--------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Vec4
    //--------------------------------------------------------

    IMATH_HOSTDEVICE static unsigned int dimensions() {return 4;}

    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    IMATH_HOSTDEVICE static T        baseTypeMin()       {return limits<T>::min();}
    IMATH_HOSTDEVICE static T        baseTypeMax()       {return limits<T>::max();}
    IMATH_HOSTDEVICE static T        baseTypeSmallest()  {return limits<T>::smallest();}
    IMATH_HOSTDEVICE static T        baseTypeEpsilon()   {return limits<T>::epsilon();}
    

    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be either a Vec2<T>, a Vec3<T>, or a Vec4<T> you can
    // refer to T as V::BaseType
    //--------------------------------------------------------------

    typedef T       BaseType;

  private:

    IMATH_HOSTDEVICE T lengthTiny () const;
};


//--------------
// Stream output
//--------------

template <class T>
std::ostream &  operator << (std::ostream &s, const Vec2<T> &v);

template <class T>
std::ostream &  operator << (std::ostream &s, const Vec3<T> &v);

template <class T>
std::ostream &  operator << (std::ostream &s, const Vec4<T> &v);

//----------------------------------------------------
// Reverse multiplication: S * Vec2<T> and S * Vec3<T>
//----------------------------------------------------

template <class T> IMATH_HOSTDEVICE Vec2<T>  operator * (T a, const Vec2<T> &v);
template <class T> IMATH_HOSTDEVICE Vec3<T>  operator * (T a, const Vec3<T> &v);
template <class T> IMATH_HOSTDEVICE Vec4<T>  operator * (T a, const Vec4<T> &v);


//-------------------------
// Typedefs for convenience
//-------------------------

typedef Vec2 <short>  V2s;
typedef Vec2 <int>    V2i;
typedef Vec2 <float>  V2f;
typedef Vec2 <double> V2d;
typedef Vec3 <short>  V3s;
typedef Vec3 <int>    V3i;
typedef Vec3 <float>  V3f;
typedef Vec3 <double> V3d;
typedef Vec4 <short>  V4s;
typedef Vec4 <int>    V4i;
typedef Vec4 <float>  V4f;
typedef Vec4 <double> V4d;


//-------------------------------------------
// Specializations for VecN<short>, VecN<int>
//-------------------------------------------

// Vec2<short>

template <> IMATH_HOSTDEVICE short
Vec2<short>::length () const;

template <> IMATH_HOSTDEVICE const Vec2<short> &
Vec2<short>::normalize ();

template <> IMATH_HOSTDEVICE const Vec2<short> &
Vec2<short>::normalizeExc ();

template <> IMATH_HOSTDEVICE const Vec2<short> &
Vec2<short>::normalizeNonNull ();

template <> IMATH_HOSTDEVICE Vec2<short>
Vec2<short>::normalized () const;

template <> IMATH_HOSTDEVICE Vec2<short>
Vec2<short>::normalizedExc () const;

template <> IMATH_HOSTDEVICE Vec2<short>
Vec2<short>::normalizedNonNull () const;


// Vec2<int>

template <> IMATH_HOSTDEVICE int
Vec2<int>::length () const;

template <> IMATH_HOSTDEVICE const Vec2<int> &
Vec2<int>::normalize ();

template <> IMATH_HOSTDEVICE const Vec2<int> &
Vec2<int>::normalizeExc ();

template <> IMATH_HOSTDEVICE const Vec2<int> &
Vec2<int>::normalizeNonNull ();

template <> IMATH_HOSTDEVICE Vec2<int>
Vec2<int>::normalized () const;

template <> IMATH_HOSTDEVICE Vec2<int>
Vec2<int>::normalizedExc () const;

template <> IMATH_HOSTDEVICE Vec2<int>
Vec2<int>::normalizedNonNull () const;


// Vec3<short>

template <> IMATH_HOSTDEVICE short
Vec3<short>::length () const;

template <> IMATH_HOSTDEVICE const Vec3<short> &
Vec3<short>::normalize ();

template <> IMATH_HOSTDEVICE const Vec3<short> &
Vec3<short>::normalizeExc ();

template <> IMATH_HOSTDEVICE const Vec3<short> &
Vec3<short>::normalizeNonNull ();

template <> IMATH_HOSTDEVICE Vec3<short>
Vec3<short>::normalized () const;

template <> IMATH_HOSTDEVICE Vec3<short>
Vec3<short>::normalizedExc () const;

template <> IMATH_HOSTDEVICE Vec3<short>
Vec3<short>::normalizedNonNull () const;


// Vec3<int>

template <> IMATH_HOSTDEVICE int
Vec3<int>::length () const;

template <> IMATH_HOSTDEVICE const Vec3<int> &
Vec3<int>::normalize ();

template <> IMATH_HOSTDEVICE const Vec3<int> &
Vec3<int>::normalizeExc ();

template <> IMATH_HOSTDEVICE const Vec3<int> &
Vec3<int>::normalizeNonNull ();

template <> IMATH_HOSTDEVICE Vec3<int>
Vec3<int>::normalized () const;

template <> IMATH_HOSTDEVICE Vec3<int>
Vec3<int>::normalizedExc () const;

template <> IMATH_HOSTDEVICE Vec3<int>
Vec3<int>::normalizedNonNull () const;


// Vec4<short>

template <> short IMATH_HOSTDEVICE
Vec4<short>::length () const;

template <> IMATH_HOSTDEVICE const Vec4<short> &
Vec4<short>::normalize ();

template <> IMATH_HOSTDEVICE const Vec4<short> &
Vec4<short>::normalizeExc ();

template <> IMATH_HOSTDEVICE const Vec4<short> &
Vec4<short>::normalizeNonNull ();

template <> IMATH_HOSTDEVICE Vec4<short>
Vec4<short>::normalized () const;

template <> IMATH_HOSTDEVICE Vec4<short>
Vec4<short>::normalizedExc () const;

template <> IMATH_HOSTDEVICE Vec4<short>
Vec4<short>::normalizedNonNull () const;


// Vec4<int>

template <> IMATH_HOSTDEVICE int
Vec4<int>::length () const;

template <> IMATH_HOSTDEVICE const Vec4<int> &
Vec4<int>::normalize ();

template <> IMATH_HOSTDEVICE const Vec4<int> &
Vec4<int>::normalizeExc ();

template <> IMATH_HOSTDEVICE const Vec4<int> &
Vec4<int>::normalizeNonNull ();

template <> IMATH_HOSTDEVICE Vec4<int>
Vec4<int>::normalized () const;

template <> IMATH_HOSTDEVICE Vec4<int>
Vec4<int>::normalizedExc () const;

template <> IMATH_HOSTDEVICE Vec4<int>
Vec4<int>::normalizedNonNull () const;


//------------------------
// Implementation of Vec2:
//------------------------

template <class T> IMATH_HOSTDEVICE
inline T &
Vec2<T>::operator [] (int i)
{
    return (&x)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T> IMATH_HOSTDEVICE
inline const T &
Vec2<T>::operator [] (int i) const
{
    return (&x)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T> IMATH_HOSTDEVICE
inline
Vec2<T>::Vec2 ()
{
    // empty
}

template <class T> IMATH_HOSTDEVICE
inline
Vec2<T>::Vec2 (T a)
{
    x = y = a;
}

template <class T> IMATH_HOSTDEVICE
inline
Vec2<T>::Vec2 (T a, T b)
{
    x = a;
    y = b;
}

template <class T> IMATH_HOSTDEVICE
inline
Vec2<T>::Vec2 (const Vec2 &v)
{
    x = v.x;
    y = v.y;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline
Vec2<T>::Vec2 (const Vec2<S> &v)
{
    x = T (v.x);
    y = T (v.y);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::operator = (const Vec2 &v)
{
    x = v.x;
    y = v.y;
    return *this;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec2<T>::setValue (S a, S b)
{
    x = T (a);
    y = T (b);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec2<T>::setValue (const Vec2<S> &v)
{
    x = T (v.x);
    y = T (v.y);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec2<T>::getValue (S &a, S &b) const
{
    a = S (x);
    b = S (y);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec2<T>::getValue (Vec2<S> &v) const
{
    v.x = S (x);
    v.y = S (y);
}

template <class T> IMATH_HOSTDEVICE
inline T *
Vec2<T>::getValue()
{
    return (T *) &x;
}

template <class T> IMATH_HOSTDEVICE
inline const T *
Vec2<T>::getValue() const
{
    return (const T *) &x;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline bool
Vec2<T>::operator == (const Vec2<S> &v) const
{
    return x == v.x && y == v.y;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline bool
Vec2<T>::operator != (const Vec2<S> &v) const
{
    return x != v.x || y != v.y;
}

template <class T> IMATH_HOSTDEVICE
bool
Vec2<T>::equalWithAbsError (const Vec2<T> &v, T e) const
{
    for (int i = 0; i < 2; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this)[i], v[i], e))
            return false;

    return true;
}

template <class T> IMATH_HOSTDEVICE
bool
Vec2<T>::equalWithRelError (const Vec2<T> &v, T e) const
{
    for (int i = 0; i < 2; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this)[i], v[i], e))
            return false;

    return true;
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec2<T>::dot (const Vec2 &v) const
{
    return x * v.x + y * v.y;
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec2<T>::operator ^ (const Vec2 &v) const
{
    return dot (v);
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec2<T>::cross (const Vec2 &v) const
{
    return x * v.y - y * v.x;

}

template <class T> IMATH_HOSTDEVICE
inline T
Vec2<T>::operator % (const Vec2 &v) const
{
    return x * v.y - y * v.x;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::operator += (const Vec2 &v)
{
    x += v.x;
    y += v.y;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
Vec2<T>::operator + (const Vec2 &v) const
{
    return Vec2 (x + v.x, y + v.y);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::operator -= (const Vec2 &v)
{
    x -= v.x;
    y -= v.y;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
Vec2<T>::operator - (const Vec2 &v) const
{
    return Vec2 (x - v.x, y - v.y);
}

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
Vec2<T>::operator - () const
{
    return Vec2 (-x, -y);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::negate ()
{
    x = -x;
    y = -y;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::operator *= (const Vec2 &v)
{
    x *= v.x;
    y *= v.y;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::operator *= (T a)
{
    x *= a;
    y *= a;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
Vec2<T>::operator * (const Vec2 &v) const
{
    return Vec2 (x * v.x, y * v.y);
}

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
Vec2<T>::operator * (T a) const
{
    return Vec2 (x * a, y * a);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::operator /= (const Vec2 &v)
{
    x /= v.x;
    y /= v.y;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec2<T> &
Vec2<T>::operator /= (T a)
{
    x /= a;
    y /= a;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
Vec2<T>::operator / (const Vec2 &v) const
{
    return Vec2 (x / v.x, y / v.y);
}

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
Vec2<T>::operator / (T a) const
{
    return Vec2 (x / a, y / a);
}

template <class T> IMATH_HOSTDEVICE
T
Vec2<T>::lengthTiny () const
{
    T absX = (x >= T (0))? x: -x;
    T absY = (y >= T (0))? y: -y;

    T max = absX;

    if (max < absY)
        max = absY;

    if (max == T (0))
        return T (0);

    //
    // Do not replace the divisions by max with multiplications by 1/max.
    // Computing 1/max can overflow but the divisions below will always
    // produce results less than or equal to 1.
    //

    absX /= max;
    absY /= max;

    return max * hostdevice_sqrt<T> (absX * absX + absY * absY);
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec2<T>::length () const
{
    T length2 = dot (*this);

    if (length2 < T (2) * std::numeric_limits<T>::min())
        return lengthTiny();

    return hostdevice_sqrt<T> (length2);
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec2<T>::length2 () const
{
    return dot (*this);
}

template <class T> IMATH_HOSTDEVICE
const Vec2<T> &
Vec2<T>::normalize ()
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
    }

    return *this;
}

template <class T> IMATH_HOSTDEVICE
const Vec2<T> &
Vec2<T>::normalizeExc ()
{
    T l = length();

    if (l == T (0))
        throw NullVecExc ("Cannot normalize null vector.");

    x /= l;
    y /= l;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline
const Vec2<T> &
Vec2<T>::normalizeNonNull ()
{
    T l = length();
    x /= l;
    y /= l;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
Vec2<T>
Vec2<T>::normalized () const
{
    T l = length();

    if (l == T (0))
        return Vec2 (T (0));

    return Vec2 (x / l, y / l);
}

template <class T> IMATH_HOSTDEVICE
Vec2<T>
Vec2<T>::normalizedExc () const
{
    T l = length();

    if (l == T (0))
        throw NullVecExc ("Cannot normalize null vector.");

    return Vec2 (x / l, y / l);
}

template <class T> IMATH_HOSTDEVICE
inline
Vec2<T>
Vec2<T>::normalizedNonNull () const
{
    T l = length();
    return Vec2 (x / l, y / l);
}


//-----------------------
// Implementation of Vec3
//-----------------------

template <class T> IMATH_HOSTDEVICE
inline T &
Vec3<T>::operator [] (int i)
{
    return (&x)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T> IMATH_HOSTDEVICE
inline const T &
Vec3<T>::operator [] (int i) const
{
    return (&x)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T> IMATH_HOSTDEVICE
inline
Vec3<T>::Vec3 ()
{
    // empty
}

template <class T> IMATH_HOSTDEVICE
inline
Vec3<T>::Vec3 (T a)
{
    x = y = z = a;
}

template <class T> IMATH_HOSTDEVICE
inline
Vec3<T>::Vec3 (T a, T b, T c)
{
    x = a;
    y = b;
    z = c;
}

template <class T> IMATH_HOSTDEVICE
inline
Vec3<T>::Vec3 (const Vec3 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline
Vec3<T>::Vec3 (const Vec3<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::operator = (const Vec3 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline
Vec3<T>::Vec3 (const Vec4<S> &v)
{
    x = T (v.x / v.w);
    y = T (v.y / v.w);
    z = T (v.z / v.w);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
Vec3<T>::Vec3 (const Vec4<S> &v, InfException)
{
    T vx = T (v.x);
    T vy = T (v.y);
    T vz = T (v.z);
    T vw = T (v.w);

    T absW = (vw >= T (0))? vw: -vw;

    if (absW < 1)
    {
        T m = baseTypeMax() * absW;

        if (vx <= -m || vx >= m || vy <= -m || vy >= m || vz <= -m || vz >= m)
            throw InfPointExc ("Cannot normalize point at infinity.");
    }

    x = vx / vw;
    y = vy / vw;
    z = vz / vw;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec3<T>::setValue (S a, S b, S c)
{
    x = T (a);
    y = T (b);
    z = T (c);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec3<T>::setValue (const Vec3<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec3<T>::getValue (S &a, S &b, S &c) const
{
    a = S (x);
    b = S (y);
    c = S (z);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline void
Vec3<T>::getValue (Vec3<S> &v) const
{
    v.x = S (x);
    v.y = S (y);
    v.z = S (z);
}

template <class T> IMATH_HOSTDEVICE
inline T *
Vec3<T>::getValue()
{
    return (T *) &x;
}

template <class T> IMATH_HOSTDEVICE
inline const T *
Vec3<T>::getValue() const
{
    return (const T *) &x;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline bool
Vec3<T>::operator == (const Vec3<S> &v) const
{
    return x == v.x && y == v.y && z == v.z;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline bool
Vec3<T>::operator != (const Vec3<S> &v) const
{
    return x != v.x || y != v.y || z != v.z;
}

template <class T> IMATH_HOSTDEVICE
bool
Vec3<T>::equalWithAbsError (const Vec3<T> &v, T e) const
{
    for (int i = 0; i < 3; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this)[i], v[i], e))
            return false;

    return true;
}

template <class T> IMATH_HOSTDEVICE
bool
Vec3<T>::equalWithRelError (const Vec3<T> &v, T e) const
{
    for (int i = 0; i < 3; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this)[i], v[i], e))
            return false;

    return true;
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec3<T>::dot (const Vec3 &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec3<T>::operator ^ (const Vec3 &v) const
{
    return dot (v);
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::cross (const Vec3 &v) const
{
    return Vec3 (y * v.z - z * v.y,
		 z * v.x - x * v.z,
		 x * v.y - y * v.x);
}

template <class T> IMATH_HOSTDEVICE
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

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator % (const Vec3 &v) const
{
    return Vec3 (y * v.z - z * v.y,
		 z * v.x - x * v.z,
		 x * v.y - y * v.x);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::operator += (const Vec3 &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator + (const Vec3 &v) const
{
    return Vec3 (x + v.x, y + v.y, z + v.z);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::operator -= (const Vec3 &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator - (const Vec3 &v) const
{
    return Vec3 (x - v.x, y - v.y, z - v.z);
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator - () const
{
    return Vec3 (-x, -y, -z);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::negate ()
{
    x = -x;
    y = -y;
    z = -z;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::operator *= (const Vec3 &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::operator *= (T a)
{
    x *= a;
    y *= a;
    z *= a;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator * (const Vec3 &v) const
{
    return Vec3 (x * v.x, y * v.y, z * v.z);
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator * (T a) const
{
    return Vec3 (x * a, y * a, z * a);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::operator /= (const Vec3 &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec3<T> &
Vec3<T>::operator /= (T a)
{
    x /= a;
    y /= a;
    z /= a;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator / (const Vec3 &v) const
{
    return Vec3 (x / v.x, y / v.y, z / v.z);
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
Vec3<T>::operator / (T a) const
{
    return Vec3 (x / a, y / a, z / a);
}

template <class T> IMATH_HOSTDEVICE
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

    return max *hostdevice_sqrt<T> (absX * absX + absY * absY + absZ * absZ);
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec3<T>::length () const
{
    T length2 = dot (*this);

    if (length2 < T (2) * std::numeric_limits<T>::min())
        return lengthTiny();

    return hostdevice_sqrt<T> (length2);
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec3<T>::length2 () const
{
    return dot (*this);
}

template <class T> IMATH_HOSTDEVICE
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

template <class T> IMATH_HOSTDEVICE
const Vec3<T> &
Vec3<T>::normalizeExc ()
{
    T l = length();

    if (l == T (0))
        throw NullVecExc ("Cannot normalize null vector.");

    x /= l;
    y /= l;
    z /= l;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
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

template <class T> IMATH_HOSTDEVICE
Vec3<T>
Vec3<T>::normalized () const
{
    T l = length();

    if (l == T (0))
        return Vec3 (T (0));

    return Vec3 (x / l, y / l, z / l);
}

template <class T> IMATH_HOSTDEVICE
Vec3<T>
Vec3<T>::normalizedExc () const
{
    T l = length();

    if (l == T (0))
        throw NullVecExc ("Cannot normalize null vector.");

    return Vec3 (x / l, y / l, z / l);
}

template <class T> IMATH_HOSTDEVICE
inline
Vec3<T>
Vec3<T>::normalizedNonNull () const
{
    T l = length();
    return Vec3 (x / l, y / l, z / l);
}


//-----------------------
// Implementation of Vec4
//-----------------------

template <class T> IMATH_HOSTDEVICE
inline T &
Vec4<T>::operator [] (int i)
{
    return (&x)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T> IMATH_HOSTDEVICE
inline const T &
Vec4<T>::operator [] (int i) const
{
    return (&x)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T> IMATH_HOSTDEVICE
inline
Vec4<T>::Vec4 ()
{
    // empty
}

template <class T> IMATH_HOSTDEVICE
inline
Vec4<T>::Vec4 (T a)
{
    x = y = z = w = a;
}

template <class T> IMATH_HOSTDEVICE
inline
Vec4<T>::Vec4 (T a, T b, T c, T d)
{
    x = a;
    y = b;
    z = c;
    w = d;
}

template <class T> IMATH_HOSTDEVICE
inline
Vec4<T>::Vec4 (const Vec4 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline
Vec4<T>::Vec4 (const Vec4<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
    w = T (v.w);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::operator = (const Vec4 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    w = v.w;
    return *this;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline
Vec4<T>::Vec4 (const Vec3<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
    w = T (1);
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline bool
Vec4<T>::operator == (const Vec4<S> &v) const
{
    return x == v.x && y == v.y && z == v.z && w == v.w;
}

template <class T>
template <class S> IMATH_HOSTDEVICE
inline bool
Vec4<T>::operator != (const Vec4<S> &v) const
{
    return x != v.x || y != v.y || z != v.z || w != v.w;
}

template <class T> IMATH_HOSTDEVICE
bool
Vec4<T>::equalWithAbsError (const Vec4<T> &v, T e) const
{
    for (int i = 0; i < 4; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this)[i], v[i], e))
            return false;

    return true;
}

template <class T> IMATH_HOSTDEVICE
bool
Vec4<T>::equalWithRelError (const Vec4<T> &v, T e) const
{
    for (int i = 0; i < 4; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this)[i], v[i], e))
            return false;

    return true;
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec4<T>::dot (const Vec4 &v) const
{
    return x * v.x + y * v.y + z * v.z + w * v.w;
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec4<T>::operator ^ (const Vec4 &v) const
{
    return dot (v);
}


template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::operator += (const Vec4 &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    w += v.w;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
Vec4<T>::operator + (const Vec4 &v) const
{
    return Vec4 (x + v.x, y + v.y, z + v.z, w + v.w);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::operator -= (const Vec4 &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    w -= v.w;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
Vec4<T>::operator - (const Vec4 &v) const
{
    return Vec4 (x - v.x, y - v.y, z - v.z, w - v.w);
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
Vec4<T>::operator - () const
{
    return Vec4 (-x, -y, -z, -w);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::negate ()
{
    x = -x;
    y = -y;
    z = -z;
    w = -w;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::operator *= (const Vec4 &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    w *= v.w;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::operator *= (T a)
{
    x *= a;
    y *= a;
    z *= a;
    w *= a;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
Vec4<T>::operator * (const Vec4 &v) const
{
    return Vec4 (x * v.x, y * v.y, z * v.z, w * v.w);
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
Vec4<T>::operator * (T a) const
{
    return Vec4 (x * a, y * a, z * a, w * a);
}

template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::operator /= (const Vec4 &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    w /= v.w;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline const Vec4<T> &
Vec4<T>::operator /= (T a)
{
    x /= a;
    y /= a;
    z /= a;
    w /= a;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
Vec4<T>::operator / (const Vec4 &v) const
{
    return Vec4 (x / v.x, y / v.y, z / v.z, w / v.w);
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
Vec4<T>::operator / (T a) const
{
    return Vec4 (x / a, y / a, z / a, w / a);
}

template <class T> IMATH_HOSTDEVICE
T
Vec4<T>::lengthTiny () const
{
    T absX = (x >= T (0))? x: -x;
    T absY = (y >= T (0))? y: -y;
    T absZ = (z >= T (0))? z: -z;
    T absW = (w >= T (0))? w: -w;

    T max = absX;

    if (max < absY)
        max = absY;

    if (max < absZ)
        max = absZ;

    if (max < absW)
        max = absW;

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
    absW /= max;

    return max *
       hostdevice_sqrt<T> (absX * absX + absY * absY + absZ * absZ + absW * absW);
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec4<T>::length () const
{
    T length2 = dot (*this);

    if (length2 < T (2) * std::numeric_limits<T>::min())
        return lengthTiny();

    return hostdevice_sqrt<T> (length2);
}

template <class T> IMATH_HOSTDEVICE
inline T
Vec4<T>::length2 () const
{
    return dot (*this);
}

template <class T> IMATH_HOSTDEVICE
const Vec4<T> &
Vec4<T>::normalize ()
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
        w /= l;
    }

    return *this;
}

template <class T> IMATH_HOSTDEVICE
const Vec4<T> &
Vec4<T>::normalizeExc ()
{
    T l = length();

    if (l == T (0))
        throw NullVecExc ("Cannot normalize null vector.");

    x /= l;
    y /= l;
    z /= l;
    w /= l;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
inline
const Vec4<T> &
Vec4<T>::normalizeNonNull ()
{
    T l = length();
    x /= l;
    y /= l;
    z /= l;
    w /= l;
    return *this;
}

template <class T> IMATH_HOSTDEVICE
Vec4<T>
Vec4<T>::normalized () const
{
    T l = length();

    if (l == T (0))
        return Vec4 (T (0));

    return Vec4 (x / l, y / l, z / l, w / l);
}

template <class T> IMATH_HOSTDEVICE
Vec4<T>
Vec4<T>::normalizedExc () const
{
    T l = length();

    if (l == T (0))
        throw NullVecExc ("Cannot normalize null vector.");

    return Vec4 (x / l, y / l, z / l, w / l);
}

template <class T> IMATH_HOSTDEVICE
inline
Vec4<T>
Vec4<T>::normalizedNonNull () const
{
    T l = length();
    return Vec4 (x / l, y / l, z / l, w / l);
}

//-----------------------------
// Stream output implementation
//-----------------------------

template <class T>
std::ostream &
operator << (std::ostream &s, const Vec2<T> &v)
{
    return s << '(' << v.x << ' ' << v.y << ')';
}

template <class T>
std::ostream &
operator << (std::ostream &s, const Vec3<T> &v)
{
    return s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
}

template <class T>
std::ostream &
operator << (std::ostream &s, const Vec4<T> &v)
{
    return s << '(' << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w << ')';
}


//-----------------------------------------
// Implementation of reverse multiplication
//-----------------------------------------

template <class T> IMATH_HOSTDEVICE
inline Vec2<T>
operator * (T a, const Vec2<T> &v)
{
    return Vec2<T> (a * v.x, a * v.y);
}

template <class T> IMATH_HOSTDEVICE
inline Vec3<T>
operator * (T a, const Vec3<T> &v)
{
    return Vec3<T> (a * v.x, a * v.y, a * v.z);
}

template <class T> IMATH_HOSTDEVICE
inline Vec4<T>
operator * (T a, const Vec4<T> &v)
{
    return Vec4<T> (a * v.x, a * v.y, a * v.z, a * v.w);
}


#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
#pragma warning(pop)
#endif

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHVEC_H
