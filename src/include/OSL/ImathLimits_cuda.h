///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMATHLIMITS_H
#define INCLUDED_IMATHLIMITS_H

//----------------------------------------------------------------
//
//	Limitations of the basic C++ numerical data types
//
//----------------------------------------------------------------

#include <OpenEXR/ImathNamespace.h>
#include <float.h>
#include <limits.h>

//------------------------------------------
// In Windows, min and max are macros.  Yay.
//------------------------------------------

#if defined _WIN32 || defined _WIN64
    #ifdef min
        #undef min
    #endif
    #ifdef max
        #undef max
    #endif
#endif

#ifndef IMATH_HOSTDEVICE
  #error "This should be included with the proper IMATH_HOSTDEVICE define"
#endif

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER


//-----------------------------------------------------------------
//
// Template class limits<T> returns information about the limits
// of numerical data type T:
//
//	min()		largest possible negative value of type T
//
//	max()		largest possible positive value of type T
//
//	smallest()	smallest possible positive value of type T
//			(for float and double: smallest normalized
//			positive value)
//
//	epsilon()	smallest possible e of type T, for which
//			1 + e != 1
//
//	isIntegral()	returns true if T is an integral type
//
//	isSigned()	returns true if T is signed
//
// Class limits<T> is useful to implement template classes or
// functions which depend on the limits of a numerical type
// which is not known in advance; for example:
//
//	template <class T> max (T x[], int n)
//	{
//	    T m = limits<T>::min();
//
//	    for (int i = 0; i < n; i++)
//		if (m < x[i])
//		    m = x[i];
//
//	    return m;
//	}
//
// Class limits<T> has been implemented for the following types:
//
//	char, signed char, unsigned char
//	short, unsigned short
//	int, unsigned int
//	long, unsigned long
//	float
//	double
//	long double
//
// Class limits<T> has only static member functions, all of which
// are implemented as inlines.  No objects of type limits<T> are
// ever created.
//
//-----------------------------------------------------------------


template <class T> struct limits
{
    IMATH_HOSTDEVICE constexpr static T	min();
    IMATH_HOSTDEVICE constexpr static T	max();
    IMATH_HOSTDEVICE constexpr static T	smallest();
    IMATH_HOSTDEVICE constexpr static T	epsilon();
    IMATH_HOSTDEVICE constexpr static bool	isIntegral();
    IMATH_HOSTDEVICE constexpr static bool	isSigned();
};


//---------------
// Implementation
//---------------

template <>
struct limits <char>
{
    IMATH_HOSTDEVICE constexpr static char min()		{return CHAR_MIN;}
    IMATH_HOSTDEVICE constexpr static char max()		{return CHAR_MAX;}
    IMATH_HOSTDEVICE constexpr static char smallest()	{return 1;}
    IMATH_HOSTDEVICE constexpr static char epsilon()	{return 1;}
    IMATH_HOSTDEVICE constexpr static bool isIntegral()	{return true;}
    IMATH_HOSTDEVICE constexpr static bool isSigned()	{return (char) ~0 < 0;}
};

template <>
struct limits <signed char>
{
    IMATH_HOSTDEVICE constexpr static signed char	min()		 {return SCHAR_MIN;}
    IMATH_HOSTDEVICE constexpr static signed char	max()		 {return SCHAR_MAX;}
    IMATH_HOSTDEVICE constexpr static signed char	smallest()	 {return 1;}
    IMATH_HOSTDEVICE constexpr static signed char	epsilon()	 {return 1;}
    IMATH_HOSTDEVICE constexpr static bool			isIntegral() {return true;}
    IMATH_HOSTDEVICE constexpr static bool			isSigned()	 {return true;}
};

template <>
struct limits <unsigned char>
{
    IMATH_HOSTDEVICE constexpr static unsigned char	min()		 {return 0;}
    IMATH_HOSTDEVICE constexpr static unsigned char	max()		 {return UCHAR_MAX;}
    IMATH_HOSTDEVICE constexpr static unsigned char	smallest()	 {return 1;}
    IMATH_HOSTDEVICE constexpr static unsigned char	epsilon()	 {return 1;}
    IMATH_HOSTDEVICE constexpr static bool			isIntegral() {return true;}
    IMATH_HOSTDEVICE constexpr static bool			isSigned()	 {return false;}
};

template <>
struct limits <short>
{
    IMATH_HOSTDEVICE constexpr static short min()		 {return SHRT_MIN;}
    IMATH_HOSTDEVICE constexpr static short max()		 {return SHRT_MAX;}
    IMATH_HOSTDEVICE constexpr static short smallest()	 {return 1;}
    IMATH_HOSTDEVICE constexpr static short epsilon()	 {return 1;}
    IMATH_HOSTDEVICE constexpr static bool  isIntegral() {return true;}
    IMATH_HOSTDEVICE constexpr static bool  isSigned()	 {return true;}
};

template <>
struct limits <unsigned short>
{
    IMATH_HOSTDEVICE constexpr static unsigned short  min()        {return 0;}
    IMATH_HOSTDEVICE constexpr static unsigned short  max()        {return USHRT_MAX;}
    IMATH_HOSTDEVICE constexpr static unsigned short  smallest()   {return 1;}
    IMATH_HOSTDEVICE constexpr static unsigned short  epsilon()    {return 1;}
    IMATH_HOSTDEVICE constexpr static bool            isIntegral() {return true;}
    IMATH_HOSTDEVICE constexpr static bool            isSigned()   {return false;}
};

template <>
struct limits <int>
{
    IMATH_HOSTDEVICE constexpr static int  min()		{return INT_MIN;}
    IMATH_HOSTDEVICE constexpr static int  max()		{return INT_MAX;}
    IMATH_HOSTDEVICE constexpr static int  smallest()	{return 1;}
    IMATH_HOSTDEVICE constexpr static int  epsilon()	{return 1;}
    IMATH_HOSTDEVICE constexpr static bool isIntegral() {return true;}
    IMATH_HOSTDEVICE constexpr static bool isSigned()	{return true;}
};

template <>
struct limits <unsigned int>
{
    IMATH_HOSTDEVICE constexpr static unsigned int  min()        {return 0;}
    IMATH_HOSTDEVICE constexpr static unsigned int  max()        {return UINT_MAX;}
    IMATH_HOSTDEVICE constexpr static unsigned int  smallest()   {return 1;}
    IMATH_HOSTDEVICE constexpr static unsigned int  epsilon()    {return 1;}
    IMATH_HOSTDEVICE constexpr static bool          isIntegral() {return true;}
    IMATH_HOSTDEVICE constexpr static bool          isSigned()   {return false;}
};

template <>
struct limits <long>
{
    IMATH_HOSTDEVICE constexpr static long  min()        {return LONG_MIN;}
    IMATH_HOSTDEVICE constexpr static long  max()        {return LONG_MAX;}
    IMATH_HOSTDEVICE constexpr static long  smallest()   {return 1;}
    IMATH_HOSTDEVICE constexpr static long  epsilon()    {return 1;}
    IMATH_HOSTDEVICE constexpr static bool isIntegral()  {return true;}
    IMATH_HOSTDEVICE constexpr static bool isSigned()    {return true;}
};

template <>
struct limits <unsigned long>
{
    IMATH_HOSTDEVICE constexpr static unsigned long min()        {return 0;}
    IMATH_HOSTDEVICE constexpr static unsigned long max()        {return ULONG_MAX;}
    IMATH_HOSTDEVICE constexpr static unsigned long smallest()   {return 1;}
    IMATH_HOSTDEVICE constexpr static unsigned long epsilon()    {return 1;}
    IMATH_HOSTDEVICE constexpr static bool          isIntegral() {return true;}
    IMATH_HOSTDEVICE constexpr static bool          isSigned()   {return false;}
};

template <>
struct limits <float>
{
    IMATH_HOSTDEVICE constexpr static float min()        {return -FLT_MAX;}
    IMATH_HOSTDEVICE constexpr static float max()        {return FLT_MAX;}
    IMATH_HOSTDEVICE constexpr static float smallest()   {return FLT_MIN;}
    IMATH_HOSTDEVICE constexpr static float epsilon()    {return FLT_EPSILON;}
    IMATH_HOSTDEVICE constexpr static bool  isIntegral() {return false;}
    IMATH_HOSTDEVICE constexpr static bool  isSigned()   {return true;}
};

template <>
struct limits <double>
{
    IMATH_HOSTDEVICE constexpr static double min()        {return -DBL_MAX;}
    IMATH_HOSTDEVICE constexpr static double max()        {return DBL_MAX;}
    IMATH_HOSTDEVICE constexpr static double smallest()   {return DBL_MIN;}
    IMATH_HOSTDEVICE constexpr static double epsilon()    {return DBL_EPSILON;}
    IMATH_HOSTDEVICE constexpr static bool   isIntegral() {return false;}
    IMATH_HOSTDEVICE constexpr static bool   isSigned()   {return true;}
};

template <>
struct limits <long double>
{
    IMATH_HOSTDEVICE constexpr static long double min()        {return -LDBL_MAX;}
    IMATH_HOSTDEVICE constexpr static long double max()        {return LDBL_MAX;}
    IMATH_HOSTDEVICE constexpr static long double smallest()   {return LDBL_MIN;}
    IMATH_HOSTDEVICE constexpr static long double epsilon()    {return LDBL_EPSILON;}
    IMATH_HOSTDEVICE constexpr static bool        isIntegral() {return false;}
    IMATH_HOSTDEVICE constexpr static bool        isSigned()   {return true;}
};


IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHLIMITS_H
