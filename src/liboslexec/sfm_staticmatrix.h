/*
Copyright (c) 2017 Intel Inc., et al.
All Rights Reserved.

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

#include <iostream>

#include <OpenImageIO/fmath.h>

#include "oslexec_pvt.h"
#include "OSL/dual_vec.h"

using namespace std;

OSL_NAMESPACE_ENTER

namespace pvt {

// non SIMD version, should be scalar code meant to be used
// inside SIMD loops
// SIMD FRIENDLY MATH
namespace sfm {
	template <int MT, int DivisorT>
	struct Multiplier {
		static float multiply(float value) { return value*(static_cast<float>(MT)/static_cast<float>(DivisorT)) ; }
	
	};
	
	template <int DivisorT>
	struct Multiplier<0, DivisorT> {
		static float multiply(float value) { return 0.0f; }
	
	};
	
	template <int DivisorT>
	struct Multiplier<DivisorT, DivisorT> {
		static float multiply(float value) { return value; }
	
	};
	
	template <int ValueT>
	struct Negative{
		static constexpr int value=-ValueT;
	};
	
	
	
	template <int MT, int DivisorT>
	struct ProxyElement {
	
		OSL_INLINE float operator * (float value) const
		{
			return value*(static_cast<float>(MT)/static_cast<float>(DivisorT));
		}
	
		OSL_INLINE float to_float() const { return static_cast<float>(MT)/static_cast<float>(DivisorT); }
	};
	
	template <int DivisorT>
	struct ProxyElement<0, DivisorT> {
		OSL_INLINE ProxyElement<0, DivisorT> operator * (float) const
		{
			return ProxyElement<0, DivisorT>();
		}
	
		OSL_INLINE float operator + (float value) const
		{
			return value;
		}
	
		template<int OtherMt, int OtherDivisorT>
		OSL_INLINE ProxyElement<OtherMt, OtherDivisorT> operator +
		(ProxyElement<OtherMt, OtherDivisorT>) const
		{
			return ProxyElement<OtherMt, OtherDivisorT>();
		}
	
		OSL_INLINE float to_float() const { return 0.0f; }
	
	};
	
	template <int MT>
	struct ProxyElement<MT, 1> {
	
		OSL_INLINE float operator * (float value) const
		{
			return static_cast<float>(MT)*value;
		}
	
	
		template<int OtherDivisorT>
		OSL_INLINE float operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
		{
			return static_cast<float>(MT + 1);
		}
	
		template<int OtherMT>
		OSL_INLINE ProxyElement<OtherMT+MT, 1> operator + (ProxyElement<OtherMT, 1>) const
		{
			return ProxyElement<OtherMT+MT, 1>();
		}
	
		template<int OtherDivisorT>
		OSL_INLINE ProxyElement<MT, 1> operator + (ProxyElement<0, OtherDivisorT>) const
		{
			return ProxyElement<MT, 1>();
		}
	
		OSL_INLINE float to_float() const { return static_cast<float>(MT); }
	
	};
	
	
	template <>
	struct ProxyElement<0, 1> {
		OSL_INLINE ProxyElement<0, 1> operator * (float) const
		{
			return ProxyElement<0, 1>();
		}
	
		OSL_INLINE float operator + (float value) const
		{
			return value;
		}
	
		template<int OtherMt, int OtherDivisorT>
		OSL_INLINE ProxyElement<OtherMt, OtherDivisorT> operator +
		(ProxyElement<OtherMt, OtherDivisorT>) const
		{
			return ProxyElement<OtherMt, OtherDivisorT>();
		}
	
		OSL_INLINE float to_float() const { return 0.0f; }
	
	};
	
	template <int DivisorT>
	struct ProxyElement<DivisorT, DivisorT> {
		OSL_INLINE float operator * (float value) const
		{
			return value;
		}
	
		OSL_INLINE float operator + (float value) const
		{
			return 1.0f + value;//
		}
	
		template<int OtherDivisorT>
		OSL_INLINE ProxyElement<2, 1> operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
		{
			return ProxyElement<2, 1>();
		}
	
		template<int OtherDivisorT>
		OSL_INLINE ProxyElement<0, DivisorT> operator + (ProxyElement<0, OtherDivisorT>) const
		{
			return ProxyElement<0, DivisorT>();
		}
	
		OSL_INLINE float to_float() const { return 1.0f; }
	
	};
	//
	template <>
	struct ProxyElement<1, 1> {
	
		OSL_INLINE float
		operator * (float value) const
		{
			return value;
		}
	
		OSL_INLINE float
		operator + (float value) const
		{
			return 1.0f + value;
		}
	
		template<int OtherDivisorT>
		OSL_INLINE ProxyElement<2, 1>
		operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
		{
			return ProxyElement<2, 1>();
		}
	
		template<int OtherDivisorT>
		OSL_INLINE
		ProxyElement<0, 1> operator + (ProxyElement<0, OtherDivisorT>) const
		{
			return ProxyElement<0, 1>();
		}
	
		OSL_INLINE float
		to_float() const { return 1.0f; }
	
	};
	
	
	template <>
	struct ProxyElement<-1, 1> {
	
		OSL_INLINE float
		operator * (float value) const
		{
			return -value;
		}
	
		OSL_INLINE float
		operator + (float value) const
		{
			return -1.0f + value;
		}//
	
		template<int OtherDivisorT>
		OSL_INLINE ProxyElement<0, 1>
		operator + (ProxyElement<OtherDivisorT, OtherDivisorT>) const
		{
			return ProxyElement<0, 1>();
		}
	
		template<int OtherDivisorT>
		OSL_INLINE
		ProxyElement<-1, 1> operator + (ProxyElement<0, OtherDivisorT>) const
		{
			return ProxyElement<-1, 1>();
		}
	
		OSL_INLINE float
		to_float() const { return -1.0f; }
	
	};
	
	template<typename ProxyElementX_T, typename ProxyElementY_T, typename ProxyElementZ_T>
	struct ProxyVec3
	{
		ProxyVec3() = delete;
		OSL_INLINE ProxyVec3(const ProxyVec3 &other)
		: x(other.x)
		, y(other.y)
		, z(other.z)
		{}
		OSL_INLINE ProxyVec3(
			const ProxyElementX_T &x_,
			const ProxyElementY_T &y_,
			const ProxyElementZ_T &z_)
		: x(x_)
		, y(y_)
		, z(z_)
		{}
	
		ProxyElementX_T x;
		ProxyElementY_T y;
		ProxyElementZ_T z;
	};
	
	template<typename ProxyElementX_T, typename ProxyElementY_T, typename ProxyElementZ_T>
	OSL_INLINE ProxyVec3<ProxyElementX_T, ProxyElementY_T, ProxyElementZ_T>
	makeProxyVec3(ProxyElementX_T x, ProxyElementY_T y, ProxyElementZ_T z)
	{
		return ProxyVec3<ProxyElementX_T, ProxyElementY_T, ProxyElementZ_T>(x,y,z);
	}
	
	OSL_INLINE
	Vec3
	makeProxyVec3(float x, float y, float z)
	{
		return Vec3(x,y,z);
	}
	
	
	template <int MT, int DivisorT>
	OSL_INLINE auto
	operator + (float a, ProxyElement<MT, DivisorT> b)
	-> decltype(b + a)
	{
		return b + a;
	}
	
	template <int MT, int DivisorT>
	OSL_INLINE auto
	operator * (float a, ProxyElement<MT, DivisorT> b)
	-> decltype(b * a)
	{
		return b * a;
	}
	
	
	
	OSL_INLINE
	float unproxy_element(float value)
	{
		return value;
	}
	
	
	
	OSL_INLINE
	Vec3 unproxy_element(const Vec3 &value)
	{
		return value;
	}
	
	template<typename ProxyElementX_T, typename ProxyElementY_T, typename ProxyElementZ_T>
	OSL_INLINE Vec3
	unproxy_element(const ProxyVec3<ProxyElementX_T, ProxyElementY_T, ProxyElementZ_T> &value)
	{
		return Vec3(unproxy_element(value.x), unproxy_element(value.y), unproxy_element(value.z));
	}
	
	
	template <int MT, int DivisorT>
	OSL_INLINE
	float unproxy_element(ProxyElement<MT, DivisorT> value)
	{
		return value.to_float();
	}
	
	template<typename ProxyElementX_T, typename ProxyElementY_T, typename ProxyElementZ_T>
	Vec3 unproxy_vec3(const ProxyVec3<ProxyElementX_T, ProxyElementY_T, ProxyElementZ_T> &a)
	{
		return Vec3(unproxy_element(a.x), unproxy_element(a.y), unproxy_element(a.z));
	}
	
	Vec3 unproxy_vec3(const Vec3 &a)
	{
		return a;
	}
	
	
	
	//Specialize operators for Vec3 to interact with ProxyElements:
	
	template <int MT, int DivisorT>
	OSL_INLINE auto
	operator* (ProxyElement<MT, DivisorT> a, const Vec3 &b)
	-> decltype(makeProxyVec3 (a*b.x, a*b.y, a*b.z))
	{
		return makeProxyVec3 (a*b.x, a*b.y, a*b.z);
	}
	
	
	template<int MT, int DivisorT>
	OSL_INLINE auto
	operator+ (ProxyElement<MT, DivisorT> a, const Vec3 &b)
	-> decltype(makeProxyVec3 (a + b.x, a + b.y, a + b.z))
	{
		return makeProxyVec3 (a + b.x, a + b.y, a + b.z);
	}
	
	template<typename XT, typename YT, typename ZT>
	OSL_INLINE auto
	operator+ (ProxyVec3<XT, YT, ZT> a, const Vec3 &b)
	-> decltype(makeProxyVec3 (a.x + b.x, a.y + b.y, a.z + b.z))
	{
		return makeProxyVec3 (a.x + b.x, a.y + b.y, a.z + b.z);
	}
	
	
	template<typename XT, typename YT, typename ZT,
			 typename X2T, typename Y2T, typename Z2T>
	OSL_INLINE auto
	operator+ (ProxyVec3<XT, YT, ZT> a, ProxyVec3<X2T, Y2T, Z2T> b)
	-> decltype(makeProxyVec3 (a.x + b.x, a.y + b.y, a.z + b.z))
	{
		return makeProxyVec3 (a.x + b.x, a.y + b.y, a.z + b.z);
	}
	
	template<typename XT, typename YT, typename ZT>
	OSL_INLINE auto
	operator* (ProxyVec3<XT, YT, ZT> a, float b)
	-> decltype(makeProxyVec3 (a.x*b, a.y*b, a.z*b))
	{
		return makeProxyVec3 (a.x*b, a.y*b, a.z*b);
	}
	
	template <int MT, int DivisorT>
	OSL_INLINE auto
	operator + (const Vec3 &a, ProxyElement<MT, DivisorT> b)
	-> decltype(b + a)
	{
		return b + a;
	}
	
	
	template<typename XT, typename YT, typename ZT>
	OSL_INLINE auto
	operator + (const Vec3 &a, ProxyVec3<XT, YT, ZT> b)
	-> decltype(b + a)
	{
		return b + a;
	}
	
	template <int MT, int DivisorT>
	OSL_INLINE auto
	operator * (const Vec3 &a, const ProxyElement<MT, DivisorT> &b)
	-> decltype (b * a)
	{
		return b * a;
	}
	
	
	//Dual2
	
	// Specialize operators for Dual2 to interact with ProxyElements
	template <class T, int MT, int DivisorT>
	OSL_INLINE Dual2<T>
	operator* (ProxyElement<MT, DivisorT> b, const Dual2<T> &a)
	{
		return Dual2<T> (unproxy_element(a.val()*b),
						 unproxy_element(a.dx()*b),
						 unproxy_element(a.dy()*b));
	}
	
	
	template<class T, int MT, int DivisorT>
	OSL_INLINE Dual2<T>
	operator+ (const Dual2<T> &a, ProxyElement<MT, DivisorT> b)
	{
		return Dual2<T> (a.val()+b, a.dx(), a.dy());
	}
	
	
	template<typename XT, typename YT, typename ZT>
	OSL_INLINE Dual2<Vec3>
	operator* (ProxyVec3<XT, YT, ZT> a, const Dual2<float> & b)
	{
		return Dual2<Vec3>(unproxy_vec3(a*b.val()), unproxy_vec3(a*b.dx()), unproxy_vec3(a*b.dy()));
	}
	
	template<typename XT, typename YT, typename ZT>
	OSL_INLINE Dual2<Vec3>
	operator*(const Dual2<float> & b, ProxyVec3<XT, YT, ZT> a)
	{
		return Dual2<Vec3>(unproxy_vec3(a*b.val()), unproxy_vec3(a*b.dx()), unproxy_vec3(a*b.dy()));
	}
	
	
	template<typename XT, typename YT, typename ZT>
	OSL_INLINE Dual2<Vec3>
	operator+(const Dual2<Vec3> & b, ProxyVec3<XT, YT, ZT> a)
	{
		return Dual2<Vec3>(unproxy_vec3(b.val() + a), b.dx(), b.dy());
	}
	
	OSL_INLINE
	Dual2<float> unproxy_element(const Dual2<float> &value)
	{
		return value;
	}
	
	OSL_INLINE
	Dual2<Vec3> unproxy_element(const Dual2<Vec3> &value)
	{
		return value;
	}
	
	template <int M00, int M01, int M02, int M03,
			  int M10, int M11, int M12, int M13,
			  int M20, int M21, int M22, int M23,
			  int M30, int M31, int M32, int M33, int DivisorT>
	struct StaticMatrix44
	{
		ProxyElement<M00,DivisorT> m00;
		ProxyElement<M01,DivisorT> m01;
		ProxyElement<M02,DivisorT> m02;
		ProxyElement<M03,DivisorT> m03;
	
		ProxyElement<M10,DivisorT> m10;
		ProxyElement<M11,DivisorT> m11;
		ProxyElement<M12,DivisorT> m12;
		ProxyElement<M13,DivisorT> m13;
	
		ProxyElement<M20,DivisorT> m20;
		ProxyElement<M21,DivisorT> m21;
		ProxyElement<M22,DivisorT> m22;
		ProxyElement<M23,DivisorT> m23;
	
		ProxyElement<M30,DivisorT> m30;
		ProxyElement<M31,DivisorT> m31;
		ProxyElement<M32,DivisorT> m32;
		ProxyElement<M33,DivisorT> m33;
	};
	
	// For debugging purposes, you can swap in a DynamicMatrix44 for a StaticMatrix44
	#if __OSL_DEBUG_STATIC_MATRIX44
	struct DynamicMatrix44
	{
		float m00;
		float m01;
		float m02;
		float m03;
	
		float m10;
		float  m11;
		float  m12;
		float  m13;
	
		float  m20;
		float m21;
		float m22;
		float m23;
	
		float m30;
		float m31;
		float m32;
		float m33;
	};
	#endif


} // namespace sfm
} // namespace pvt
OSL_NAMESPACE_EXIT
