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
static constexpr int SimdLaneCount = 16;


/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void * TransformationPtr;


template <int WidthT>
class WideMask
{
    typedef unsigned int value_type;
    static_assert(sizeof(value_type)*8 > WidthT, "unsupported WidthT");

    value_type m_value;
public:

    OSL_INLINE WideMask()
    {}

    explicit OSL_INLINE WideMask(bool all_on_or_off)
    : m_value((all_on_or_off) ? (0xFFFFFFFF >> (32-WidthT)) : 0)
    {}
    
    OSL_INLINE WideMask(value_type value_)
        : m_value(value_)
    {}

    OSL_INLINE WideMask(int value_)
        : m_value(static_cast<value_type>(value_))
    {}

    OSL_INLINE WideMask(const WideMask &other)
        : m_value(other.m_value)
    {}

    OSL_INLINE value_type value() const
    { return m_value; }

    // Testers
    OSL_INLINE bool operator[](int lane) const
    {
        // From testing code generation this is the preferred form
        return (m_value & (1<<lane))==(1<<lane);
    }

    OSL_INLINE bool is_on(int lane) const
    {
        // From testing code generation this is the preferred form
        return (m_value & (1<<lane))==(1<<lane);
    }

    OSL_INLINE bool is_off(int lane) const
    {
        // From testing code generation this is the preferred form
        return (m_value & (1<<lane))==0;
    }

    OSL_INLINE bool all_on() const
    {
        // TODO:  is this more expensive than == ?
        return (m_value >= (0xFFFFFFFF >> (32-WidthT));
    }

    OSL_INLINE bool all_off() const
    {
        return (m_value == static_cast<value_type>(0));
    }

    OSL_INLINE bool any_on() const
    {
        return (m_value != static_cast<value_type>(0));
    }

    OSL_INLINE bool any_off() const
    {
        return (m_value < (0xFFFFFFFF >> (32-WidthT));
    }


    // Setters
    OSL_INLINE void set(int lane, bool flag)
    {
        if (flag) {
            m_value |= (1<<lane);
        } else {
            m_value &= (~(1<<lane));
        }
    }

    OSL_INLINE void set_on(int lane)
    {
        m_value |= (1<<lane);
    }

    OSL_INLINE void set_all_on()
    {
        m_value = (0xFFFFFFFF >> (32-WidthT));
    }

    OSL_INLINE void set_off(int lane)
    {
        m_value &= (~(1<<lane));
    }

    OSL_INLINE void set_all_off()
    {
        m_value = 0;
    }
};

typedef WideMask<SimdLaneCount> Mask;



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


template <typename BuiltinT, int WidthT>
struct WideBuiltin
{
	typedef BuiltinT value_type;
	static constexpr int width = WidthT; 
	
	value_type data[WidthT];
	
	OSL_INLINE void
	set(int index, value_type value) 
	{
		data[index] = value;
	}

	OSL_INLINE void
	set_all(value_type value) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
					data[i] = value;
			}
		}
	}
	
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const WideBuiltin & other) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					data[i] = other.get(i);
				}
			}
		}
	}

	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, value_type value) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					data[i] = value;
				}
			}
		}
	}
	
protected:
	template<int HeadIndexT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT>, const value_type & value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... BuiltinListT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT, TailIndexListT...>, value_type headValue, BuiltinListT... tailValues)
	{
		set(HeadIndexT, headValue);
		set(internal::int_sequence<TailIndexListT...>(), tailValues...);
		return;
	}
public:
	
	WideBuiltin() = default;

	template<typename... BuiltinListT, typename = internal::enable_if_type<(sizeof...(BuiltinListT) == WidthT)> >
	OSL_INLINE
	WideBuiltin(const BuiltinListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(BuiltinListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	
	OSL_INLINE explicit  
	WideBuiltin(const value_type & uniformValue) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				data[i] = uniformValue;
			}
		}
	}
	
	
	
	OSL_INLINE BuiltinT 
	get(int index) const 
	{
		return data[index];
	}	
	
	void dump(const char *name) const
	{
		if (name != nullptr) {
			std::cout << name << " = ";
		}
		std::cout << "{";				
		for(int i=0; i < WidthT; ++i)
		{
			std::cout << data[i];
			if (i < (WidthT-1))
				std::cout << ",";
				
		}
		std::cout << "}" << std::endl;
	}
};





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
	typedef Vec3 value_type;
	static constexpr int width = WidthT; 
	float x[WidthT];
	float y[WidthT];
	float z[WidthT];
	
	OSL_INLINE void 
	set(int index, const Vec3 & value) 
	{
		x[index] = value.x;
		y[index] = value.y;
		z[index] = value.z;
	}
	
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const Wide & other) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[i] = other.x.get(i);
					y[i] = other.y.get(i);
					z[i] = other.z.get(i);
				}
			}
		}
	}
	
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const value_type & value) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[i] = value.x;
					y[i] = value.y;
					z[i] = value.z;
				}
			}
		}
	}
	
	
protected:
	template<int HeadIndexT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT>, const Vec3 & value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... Vec3ListT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT, TailIndexListT...>, Vec3 headValue, Vec3ListT... tailValues)
	{
		set(HeadIndexT, headValue);
		set(internal::int_sequence<TailIndexListT...>(), tailValues...);
		return;
	}
public:
	
	Wide() = default;
	// We want to avoid accidentially copying these when the intent was to just have
	// a reference
	Wide(const Wide &other) = delete;

	template<typename... Vec3ListT, typename = internal::enable_if_type<(sizeof...(Vec3ListT) == WidthT)> >
	OSL_INLINE
	Wide(const Vec3ListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(Vec3ListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	

	OSL_INLINE Vec3 
	get(int index) const 
	{
		return Vec3(x[index], y[index], z[index]);
	}		
	
	void dump(const char *name) const
	{
		if (name != nullptr) {
			std::cout << name << " = ";
		}
		std::cout << "x{";				
		for(int i=0; i < WidthT; ++i)
		{
			std::cout << x[i];
			if (i < (WidthT-1))
				std::cout << ",";
				
		}
		std::cout << "}" << std::endl;
		std::cout << "y{";				
		for(int i=0; i < WidthT; ++i)
		{
			std::cout << y[i];
			if (i < (WidthT-1))
				std::cout << ",";
				
		}
		std::cout << "}" << std::endl;
		std::cout << "z{"	;			
		for(int i=0; i < WidthT; ++i)
		{
			std::cout << z[i];
			if (i < (WidthT-1))
				std::cout << ",";
				
		}
		std::cout << "}" << std::endl;
	}
	
};

template <int WidthT>
struct Wide<Color3, WidthT>
{
	typedef Color3 value_type;	
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];

    OSL_INLINE void
    set(int index, const Color3 & value)
    {
        x[index] = value.x;
        y[index] = value.y;
        z[index] = value.z;
    }

	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const Wide & other) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[i] = other.x.get(i);
					y[i] = other.y.get(i);
					z[i] = other.z.get(i);
				}
			}
		}
	}
    
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const value_type & value) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[i] = value.x;
					y[i] = value.y;
					z[i] = value.z;
				}
			}
		}
	}
	
protected:
    template<int HeadIndexT>
    OSL_INLINE void
    set(internal::int_sequence<HeadIndexT>, const Color3 & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Color3ListT>
    OSL_INLINE void
    set(internal::int_sequence<HeadIndexT, TailIndexListT...>, Color3 headValue, Color3ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(internal::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    Wide() = default;
    Wide(const Wide &other) = delete;

    template<typename... Color3ListT, typename = internal::enable_if_type<(sizeof...(Color3ListT) == WidthT)> >
    OSL_INLINE
    Wide(const Color3ListT &...values)
    {
        typedef internal::make_int_sequence<sizeof...(Color3ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_INLINE Color3
    get(int index) const
    {
        return Color3(x[index], y[index], z[index]);
    }

    void dump(const char *name) const
    {
        if (name != nullptr) {
            std::cout << name << " = ";
        }
        std::cout << "x{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << x[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "y{";
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << y[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
        std::cout << "z{"	;
        for(int i=0; i < WidthT; ++i)
        {
            std::cout << z[i];
            if (i < (WidthT-1))
                std::cout << ",";

        }
        std::cout << "}" << std::endl;
    }

};

template <int WidthT>
struct Wide<Matrix44, WidthT>
{	
	typedef Matrix44 value_type;
	static constexpr int width = WidthT; 
	Wide<float, WidthT> x[4][4];
	
	OSL_INLINE void 
	set(int index, const Matrix44 & value) 
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

	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const Wide & other) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[0][0].set(i, other.x[0][0].get(i));
					x[0][1].set(i, other.x[0][1].get(i));
					x[0][2].set(i, other.x[0][2].get(i));
					x[0][3].set(i, other.x[0][3].get(i));
					x[1][0].set(i, other.x[1][0].get(i));
					x[1][1].set(i, other.x[1][1].get(i));
					x[1][2].set(i, other.x[1][2].get(i));
					x[1][3].set(i, other.x[1][3].get(i));
					x[2][0].set(i, other.x[2][0].get(i));
					x[2][1].set(i, other.x[2][1].get(i));
					x[2][2].set(i, other.x[2][2].get(i));
					x[2][3].set(i, other.x[2][3].get(i));
					x[3][0].set(i, other.x[3][0].get(i));
					x[3][1].set(i, other.x[3][1].get(i));
					x[3][2].set(i, other.x[3][2].get(i));
					x[3][3].set(i, other.x[3][3].get(i));
				}
			}
		}
	}
	
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const value_type & value) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[0][0].set(i, value.x[0][0]);
					x[0][1].set(i, value.x[0][1]);
					x[0][2].set(i, value.x[0][2]);
					x[0][3].set(i, value.x[0][3]);
					x[1][0].set(i, value.x[1][0]);
					x[1][1].set(i, value.x[1][1]);
					x[1][2].set(i, value.x[1][2]);
					x[1][3].set(i, value.x[1][3]);
					x[2][0].set(i, value.x[2][0]);
					x[2][1].set(i, value.x[2][1]);
					x[2][2].set(i, value.x[2][2]);
					x[2][3].set(i, value.x[2][3]);
					x[3][0].set(i, value.x[3][0]);
					x[3][1].set(i, value.x[3][1]);
					x[3][2].set(i, value.x[3][2]);
					x[3][3].set(i, value.x[3][3]);
				}
			}
		}
	}
	
	
	OSL_INLINE Matrix44 
	get(int index) const 
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
	
	OSL_INLINE void 
	set(int index, const value_type & value) 
	{
		x[index] = value.val();
		dx[index] = value.dx();
		dy[index] = value.dy();
	}
	
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const Wide & other) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[i] = other.x.get(i);
					dx[i] = other.dx.get(i);
					dy[i] = other.dy.get(i);
				}
			}
		}
	}

	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const value_type & value) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x[i] = value.x;
					dx[i] = value.dx;
					dy[i] = value.dy;
				}
			}
		}
	}
	
protected:
	template<int HeadIndexT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT>, const value_type &value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT, TailIndexListT...>, value_type headValue, ValueListT... tailValues)
	{
		set(HeadIndexT, headValue);
		set(internal::int_sequence<TailIndexListT...>(), tailValues...);
		return;
	}
public:
	
	// TODO:  should other wide types delete their copy constructors?
	Wide() = default;
	Wide(const Wide &other) = delete;

	template<typename... ValueListT, typename = internal::enable_if_type<(sizeof...(ValueListT) == WidthT)> >
	OSL_INLINE 
	Wide(const ValueListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	

	OSL_INLINE value_type 
	get(int index) const 
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
	
	OSL_INLINE void 
	set(int index, const value_type & value) 
	{
		x.set(index, value.val());
		dx.set(index, value.dx());
		dy.set(index, value.dy());
	}
	
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const Wide & other) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x.set(i, other.x.get(i);
					dx.set(i, other.dx.get(i);
					dy.set(i, other.dy.get(i);
				}
			}
		}
	}
	
	OSL_INLINE void 
	blendin(WideMask<WidthT> mask, const value_type & value) 
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i)
			{
				if (mask[i]) {
					x.set(i, value.x;
					dx.set(i, value.dx;
					dy.set(i, value.dy;
				}
			}
		}
	}
	
	
protected:
	template<int HeadIndexT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT>, const value_type &value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... ValueListT>
	OSL_INLINE void 
	set(internal::int_sequence<HeadIndexT, TailIndexListT...>, value_type headValue, ValueListT... tailValues)
	{
		set(HeadIndexT, headValue);
		set(internal::int_sequence<TailIndexListT...>(), tailValues...);
		return;
	}
public:
	
	Wide() = default;
	Wide(const Wide &other) = delete;

	template<typename... ValueListT, typename = internal::enable_if_type<(sizeof...(ValueListT) == WidthT)> >
	OSL_INLINE
	Wide(const ValueListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(ValueListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	

	OSL_INLINE value_type 
	get(int index) const 
	{
		return value_type(x.get(index), dx.get(index), dy.get(index));
	}		
};


template <typename DataT, int WidthT>
struct WideUniformProxy
{
	OSL_INLINE
	WideUniformProxy(Wide<DataT, WidthT> & ref_wide_data)
	: m_ref_wide_data(ref_wide_data)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members througk back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	WideUniformProxy(const WideUniformProxy &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	{}
	
	// Sets all data lanes of wide to the value
	OSL_INLINE const DataT & 
	operator = (const DataT & value)  
	{
		OSL_INTEL_PRAGMA("forceinline recursive")
		{
			OSL_INTEL_PRAGMA("ivdep")
			OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
			for(int i = 0; i < WidthT; ++i) {
				m_ref_wide_data.set(i, value);
			}
		}
		
		return value;
	}
	
private:
	Wide<DataT, WidthT> & m_ref_wide_data;
};

template <typename DataT, int WidthT>
OSL_INLINE void 
make_uniform(Wide<DataT, WidthT> &wide_data, const DataT &value)
{
	OSL_INTEL_PRAGMA("forceinline recursive")
	{
		OSL_INTEL_PRAGMA("ivdep")
		OSL_INTEL_PRAGMA("simd vectorlength(WidthT)")
		for(int i = 0; i < WidthT; ++i) {
			wide_data.set(i, value);
		}
	}
}




template <typename DataT, int WidthT>
struct WideProxy
{
	OSL_INLINE 
	WideProxy(Wide<DataT, WidthT> & ref_wide_data, const int index)
	: m_ref_wide_data(ref_wide_data)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members througk back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	WideProxy(const WideProxy &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	, m_index(other.m_index)
	{}
	
	OSL_INLINE 
	operator DataT const () const 
	{
		return m_ref_wide_data.get(m_index);
	}

	OSL_INLINE const DataT &
	operator = (const DataT & value)  
	{
		m_ref_wide_data.set(m_index, value);
		return value;
	}
	
	OSL_INLINE WideUniformProxy<DataT, WidthT> 
	uniform() 
	{
		return WideUniformProxy<DataT, WidthT>(m_ref_wide_data); 
	}
	
private:
	Wide<DataT, WidthT> & m_ref_wide_data;
	const int m_index;
};


template <typename DataT, int WidthT>
struct ConstWideProxy
{
	OSL_INLINE
	ConstWideProxy(const Wide<DataT, WidthT> & ref_wide_data, const int index)
	: m_ref_wide_data(ref_wide_data)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members througk back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstWideProxy(const ConstWideProxy &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	, m_index(other.m_index)
	{}	
	
	OSL_INLINE
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
	OSL_INLINE
	WideAccessor(const void *ptr_wide_data)
	: m_ref_wide_data(*reinterpret_cast<const Wide<DataT, WidthT> *>(ptr_wide_data))
	{}
	
	OSL_INLINE
	WideAccessor(const Wide<DataT, WidthT> & ref_wide_data)
	: m_ref_wide_data(ref_wide_data)
	{}
	
	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members througk back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	WideAccessor(const WideAccessor &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	{}	
	
	
	typedef ConstWideProxy<DataT, WidthT> Proxy;
	
	OSL_INLINE Proxy const 
	operator[](int index) const
	{
		return Proxy(m_ref_wide_data, index);
	}
		
private:
	const Wide<DataT, WidthT> & m_ref_wide_data;	
};



OSL_INLINE void robust_multVecMatrix(const Wide<Matrix44>& wx, const Wide< Imath::Vec3<float> >& wsrc, Wide< Imath::Vec3<float> >& wdst)
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

OSL_INLINE void
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
OSL_INLINE void
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
OSL_INLINE void
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
