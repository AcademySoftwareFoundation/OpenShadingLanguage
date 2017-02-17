/*
Copyright (c) 2009-2013 Sony Pictures Imageworks Inc., et al.
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

#include <type_traits>

#include "dual_vec.h"

OSL_NAMESPACE_ENTER

// TODO: add conditional compilation to change this
static constexpr int SimdLaneCount = 8;


/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void * TransformationPtr;


template <typename BuiltinT, int WidthT>
struct WideBuiltin
{	
	static constexpr int width = WidthT; 
	
	BuiltinT data[WidthT];
	
	void set(int index, BuiltinT value) 
	{
		data[index] = value;
	}

	BuiltinT get(int index) const 
	{
		return data[index];
	}	
};


namespace internal {

template<int... IntegerListT>
struct int_sequence
{
};

template<int StartAtT, int EndBeforeT, typename IntSequenceT>
struct int_sequence_generator;

template<int StartAtT, int EndBeforeT, int... IntegerListT>
struct int_sequence_generator<StartAtT, EndBeforeT, int_sequence<IntegerListT...>>
{
    typedef typename int_sequence_generator<StartAtT+1, EndBeforeT, int_sequence<IntegerListT..., StartAtT>>::type type;
};

template<int EndBeforeT, int... IntegerListT>
struct int_sequence_generator<EndBeforeT, EndBeforeT, int_sequence<IntegerListT...>>
{
    typedef int_sequence<IntegerListT...> type;
};

template<int EndBeforeT, int StartAtT=0>
using make_int_sequence = typename int_sequence_generator<StartAtT, EndBeforeT, int_sequence<> >::type;

// We need the SFINAE type to be different for
// enable_ifType from disable_ifType so that we can apply both to
// the same template signature to avoid
// "error: invalid redeclaration of member function template"
// NOTE: std::enable_if_t is a c++14 library feature, our baseline
// and we wish to remain compatible with c++11 header libraries
template <bool TestT, typename TypeT = std::true_type>
using enable_if_type = typename std::enable_if<TestT, TypeT>::type;

} // namespace internal


template <typename DataT, int WidthT = SimdLaneCount>
struct Wide; // undefined

// Specializations
template <int WidthT>
struct Wide<float, WidthT> : public WideBuiltin<float, WidthT> {};

template <int WidthT>
struct Wide<int, WidthT> : public WideBuiltin<int, WidthT> {};

template <int WidthT>
struct Wide<TransformationPtr, WidthT> : public WideBuiltin<TransformationPtr, WidthT> {};

template <int WidthT>
struct Wide<Vec3, WidthT>
{	
	static constexpr int width = WidthT; 
	float x[WidthT];
	float y[WidthT];
	float z[WidthT];
	
	void set(int index, const Vec3 & value) 
	{
		x[index] = value.x;
		y[index] = value.y;
		z[index] = value.z;
	}
	
protected:
	template<int HeadIndexT>
	void set(internal::int_sequence<HeadIndexT>, const Vec3 & value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... Vec3ListT>
	void set(internal::int_sequence<HeadIndexT, TailIndexListT...>, Vec3 headValue, Vec3ListT... tailValues)
	{
		set(HeadIndexT, headValue);
		set(internal::int_sequence<TailIndexListT...>(), tailValues...);
		return;
	}
public:
	
	Wide() = default;
	Wide(const Wide &other) = delete;

	template<typename... Vec3ListT, typename = internal::enable_if_type<(sizeof...(Vec3ListT) == WidthT)> >
	Wide(const Vec3ListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(Vec3ListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	

	Vec3 get(int index) const 
	{
		return Vec3(x[index], y[index], z[index]);
	}		
};


template <int WidthT>
struct Wide<Matrix44, WidthT>
{	
	static constexpr int width = WidthT; 
	Wide<float, WidthT> x[4][4];
	
	void set(int index, const Matrix44 & value) 
	{
		x[0][0].set(index, value.x[0][0]);
		x[0][1].set(index, value.x[0][1]);
		x[0][2].set(index, value.x[0][2]);
		x[0][3].set(index, value.x[0][3]);
		x[1][0].set(index, value.x[1][0]);
		x[1][1].set(index, value.x[1][1]);
		x[1][2].set(index, value.x[1][2]);
		x[1][3].set(index, value.x[1][3]);
		x[2][0].set(index, value.x[2][0]);
		x[2][1].set(index, value.x[2][1]);
		x[2][2].set(index, value.x[2][2]);
		x[2][3].set(index, value.x[2][3]);
		x[3][0].set(index, value.x[3][0]);
		x[3][1].set(index, value.x[3][1]);
		x[3][2].set(index, value.x[3][2]);
		x[3][3].set(index, value.x[3][3]);
}

	Matrix44 get(int index) const 
	{
		return Matrix44(
			x[0][0].get(index), x[0][1].get(index), x[0][2].get(index), x[0][3].get(index),
			x[1][0].get(index), x[1][1].get(index), x[1][2].get(index), x[1][3].get(index),
			x[2][0].get(index), x[2][1].get(index), x[2][2].get(index), x[2][3].get(index),
			x[3][0].get(index), x[3][1].get(index), x[3][2].get(index), x[3][3].get(index));
	}		
};


template <int WidthT>
struct Wide<Dual2<float>, WidthT>
{	
	typedef Dual2<float> value_type;
	static constexpr int width = WidthT; 
	float x[WidthT];
	float dx[WidthT];
	float dy[WidthT];
	
	void set(int index, const value_type & value) 
	{
		x[index] = value.val();
		dx[index] = value.dx();
		dy[index] = value.dy();
	}
	
protected:
	template<int HeadIndexT>
	void set(internal::int_sequence<HeadIndexT>, const value_type &value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
	void set(internal::int_sequence<HeadIndexT, TailIndexListT...>, value_type headValue, ValueListT... tailValues)
	{
		set(HeadIndexT, headValue);
		set(internal::int_sequence<TailIndexListT...>(), tailValues...);
		return;
	}
public:
	
	Wide() = default;
	Wide(const Wide &other) = delete;

	template<typename... ValueListT, typename = internal::enable_if_type<(sizeof...(ValueListT) == WidthT)> >
	Wide(const ValueListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	

	value_type get(int index) const 
	{
		return value_type(x[index], dx[index], dy[index]);
	}		
};

template <int WidthT>
struct Wide<Dual2<Vec3>, WidthT>
{	
	typedef Dual2<Vec3> value_type;
	static constexpr int width = WidthT; 
	Wide<Vec3> x;
	Wide<Vec3> dx;
	Wide<Vec3> dy;
	
	void set(int index, const value_type & value) 
	{
		x.set(index, value.val());
		dx.set(index, value.dx());
		dy.set(index, value.dy());
	}
	
protected:
	template<int HeadIndexT>
	void set(internal::int_sequence<HeadIndexT>, const value_type &value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
	void set(internal::int_sequence<HeadIndexT, TailIndexListT...>, value_type headValue, ValueListT... tailValues)
	{
		set(HeadIndexT, headValue);
		set(internal::int_sequence<TailIndexListT...>(), tailValues...);
		return;
	}
public:
	
	Wide() = default;
	Wide(const Wide &other) = delete;

	template<typename... ValueListT, typename = internal::enable_if_type<(sizeof...(ValueListT) == WidthT)> >
	Wide(const ValueListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	

	value_type get(int index) const 
	{
		return value_type(x.get(index), dx.get(index), dy.get(index));
	}		
};


template <typename DataT, int WidthT>
struct WideUniformProxy
{
	WideUniformProxy(Wide<DataT, WidthT> & ref_wide_data)
	: m_ref_wide_data(ref_wide_data)
	{}

	// Sets all data lanes of wide to the value
	const DataT & operator = (const DataT & value)  
	{
		for(int i = 0; i < WidthT; ++i) {
			m_ref_wide_data.set(i, value);
		}
		return value;
	}
	
private:
	Wide<DataT, WidthT> & m_ref_wide_data;
};


template <typename DataT, int WidthT>
struct WideProxy
{
	WideProxy(Wide<DataT, WidthT> & ref_wide_data, const int index)
	: m_ref_wide_data(ref_wide_data)
	, m_index(index)
	{}

	
	operator DataT const () const 
	{
		return m_ref_wide_data.get(m_index);
	}

	const DataT & operator = (const DataT & value)  
	{
		m_ref_wide_data.set(m_index, value);
		return value;
	}
	
	WideUniformProxy<DataT, WidthT> uniform() { return WideUniformProxy<DataT, WidthT>(m_ref_wide_data); }
	
private:
	Wide<DataT, WidthT> & m_ref_wide_data;
	const int m_index;
};

template <typename DataT, int WidthT>
struct ConstWideProxy
{
	ConstWideProxy(const Wide<DataT, WidthT> & ref_wide_data, const int index)
	: m_ref_wide_data(ref_wide_data)
	, m_index(index)
	{}

	
	operator DataT const () const 
	{
		return m_ref_wide_data.get(m_index);
	}

private:
	const Wide<DataT, WidthT> & m_ref_wide_data;
	const int m_index;
};


template <typename DataT, int WidthT>
struct WideAccessor
{
	WideAccessor(const void *ptr_wide_data)
	: m_ref_wide_data(*reinterpret_cast<const Wide<DataT, WidthT> *>(ptr_wide_data))
	{}
	
	WideAccessor(const Wide<DataT, WidthT> & ref_wide_data)
	: m_ref_wide_data(ref_wide_data)
	{}
	
	typedef ConstWideProxy<DataT, WidthT> Proxy;
	
	Proxy const operator[](int index) const
	{
		return Proxy(m_ref_wide_data, index);
	}
		
private:
	const Wide<DataT, WidthT> & m_ref_wide_data;	
};


inline void robust_multVecMatrix(const Wide<Matrix44>& wx, const Wide< Imath::Vec3<float> >& wsrc, Wide< Imath::Vec3<float> >& wdst)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		//OSL_INTEL_PRAGMA("ivdep")
		//OSL_INTEL_PRAGMA("novector")
		//OSL_INTEL_PRAGMA("nounroll")
		//OSL_INTEL_PRAGMA("novector")
		OSL_INTEL_PRAGMA("simd vectorlength(Wide<Matrix44>::width)")
		for(int index=0; index < Wide<Matrix44>::width; ++index)
		{
		   const Matrix44 x = wx.get(index);
		   Imath::Vec3<float> src = wsrc.get(index);
		   
		   //std::cout << "----src>" << src << std::endl;
		   
		   Imath::Vec3<float> dst;	   
	
		   
		   //robust_multVecMatrix(x, src, dst);
#if 1
#if 0
		   float a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0] + x[3][0];
		    float b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1] + x[3][1];
		    float c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2] + x[3][2];
		    float w = src[0] * x[0][3] + src[1] * x[1][3] + src[2] * x[2][3] + x[3][3];
#else
			   float a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0] + x[3][0];
			    float b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1] + x[3][1];
			    float c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2] + x[3][2];
			    float w = src.x * x[0][3] + src.y * x[1][3] + src.z * x[2][3] + x[3][3];
		    
#endif

		    if (__builtin_expect(w != 0, 1)) {
		       dst.x = a / w;
		       dst.y = b / w;
		       dst.z = c / w;
		    } else {
		       dst.x = 0;
		       dst.y = 0;
		       dst.z = 0;
		    }		   
#endif
		    
		   //std::cout << "----dst>" << dst << std::endl;
		   
		   wdst.set(index, dst);
		   
		   //Imath::Vec3<float> verify = wdst.get(index);
		   //std::cout << "---->" << verify << "<-----" << std::endl;
		}
	}
}

inline void
avoidAliasingMultDirMatrix (const Matrix44 &M, const Vec3 &src, Vec3 &dst)
{
	float a = src.x * M[0][0] + src.y * M[1][0] + src.z * M[2][0];
	float b = src.x * M[0][1] + src.y * M[1][1] + src.z * M[2][1];
	float c = src.x * M[0][2] + src.y * M[1][2] + src.z * M[2][2];

	dst.x = a;
	dst.y = b;
	dst.z = c;
    
}

#if 0 // In development, not done 
/// Multiply a matrix times a direction with derivatives to obtain
/// a transformed direction with derivatives.
inline void
multDirMatrix (const Matrix44 &M, const Wide<Dual2<Vec3>> &win, Wide<Dual2<Vec3>> &wout)
{   
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		//OSL_INTEL_PRAGMA("ivdep")
		//OSL_INTEL_PRAGMA("novector")
		//OSL_INTEL_PRAGMA("nounroll")
		//OSL_INTEL_PRAGMA("novector")
		//OSL_INTEL_PRAGMA("simd vectorlength(Wide<Matrix44>::width)")
		for(int index=0; index < Wide<Matrix44>::width; ++index)
		{
		   const Matrix44 M = win.get(index);
		   Dual2<Imath::Vec3<float>> src = wsrc.get(index);
		   
		   Dual2<Imath::Vec3<float>> dst;	   
	
		   avoidAliasingMultDirMatrix(M, src.val(), dst.val());
		   avoidAliasingMultDirMatrix(M, src.dx(), dst.dx());
		   avoidAliasingMultDirMatrix(M, src.dy(), dst.dy());
		   
		   wout.set(index, dst);
		}
	}    
}
#endif


/// Multiply a matrix times a vector with derivatives to obtain
/// a transformed vector with derivatives.
inline void
robust_multVecMatrix (const Wide<Matrix44> &WM, const Wide<Dual2<Vec3>> &win, Wide<Dual2<Vec3>> &wout)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		//OSL_INTEL_PRAGMA("ivdep")
		//OSL_INTEL_PRAGMA("novector")
		//OSL_INTEL_PRAGMA("nounroll")
		//OSL_INTEL_PRAGMA("novector")
		//OSL_INTEL_PRAGMA("simd vectorlength(Wide<Matrix44>::width)")
		for(int index=0; index < Wide<Matrix44>::width; ++index)
		{
			const Matrix44 M = WM.get(index);			
			const Dual2<Vec3> in = win.get(index);
	
			// Rearrange into a Vec3<Dual2<float> >
			Imath::Vec3<Dual2<float> > din, dout;
			for (int i = 0;  i < 3;  ++i)
				din[i].set (in.val()[i], in.dx()[i], in.dy()[i]);
		
#if 0
			Dual2<float> a = din[0] * M[0][0] + din[1] * M[1][0] + din[2] * M[2][0] + M[3][0];
			Dual2<float> b = din[0] * M[0][1] + din[1] * M[1][1] + din[2] * M[2][1] + M[3][1];
			Dual2<float> c = din[0] * M[0][2] + din[1] * M[1][2] + din[2] * M[2][2] + M[3][2];
			Dual2<float> w = din[0] * M[0][3] + din[1] * M[1][3] + din[2] * M[2][3] + M[3][3];
#else
			Dual2<float> a = din.x * M[0][0] + din.y * M[1][0] + din.z * M[2][0] + M[3][0];
			Dual2<float> b = din.x * M[0][1] + din.y * M[1][1] + din.z * M[2][1] + M[3][1];
			Dual2<float> c = din.x * M[0][2] + din.y * M[1][2] + din.z * M[2][2] + M[3][2];
			Dual2<float> w = din.x * M[0][3] + din.y * M[1][3] + din.z * M[2][3] + M[3][3];
#endif
			
		
			if (w.val() != 0) {
			   dout.x = a / w;
			   dout.y = b / w;
			   dout.z = c / w;
			} else {
			   dout.x = 0;
			   dout.y = 0;
			   dout.z = 0;
			}
		
			Dual2<Vec3> out;
			// Rearrange back into Dual2<Vec3>
#if 0
			out.set (Vec3 (dout[0].val(), dout[1].val(), dout[2].val()),
					 Vec3 (dout[0].dx(),  dout[1].dx(),  dout[2].dx()),
					 Vec3 (dout[0].dy(),  dout[1].dy(),  dout[2].dy()));
#else
			out.set (Vec3 (dout.x.val(), dout.y.val(), dout.z.val()),
					 Vec3 (dout.x.dx(),  dout.y.dx(),  dout.z.dx()),
					 Vec3 (dout.x.dy(),  dout.y.dy(),  dout.z.dy()));
#endif
			
			wout.set(index, out);
		   
		   //Imath::Vec3<float> verify = wdst.get(index);
		   //std::cout << "---->" << verify << "<-----" << std::endl;
		}
	}
    
}


OSL_NAMESPACE_EXIT
