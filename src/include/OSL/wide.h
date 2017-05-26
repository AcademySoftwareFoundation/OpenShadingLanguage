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
public:
    typedef unsigned int value_type;
    static_assert(sizeof(value_type)*8 > WidthT, "unsupported WidthT");
	static constexpr int width = WidthT; 

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
private:
    value_type m_value;
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
	
	OSL_INLINE WideBuiltin() = default;

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
	
	OSL_INLINE Wide() = default;
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

    OSL_INLINE Wide() = default;
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
struct Wide<ustring, WidthT>
{	
	static constexpr int width = WidthT; 
    ustring str[WidthT];
    static_assert(sizeof(ustring) == sizeof(char*), "ustring must be pointer size");
	
	OSL_INLINE void 
	set(int index, const ustring& value) 
	{
        str[index] = value;
	}

	OSL_INLINE ustring 
	get(int index) const 
	{
        return str[index];
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
	OSL_INLINE Wide() = default;
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
	
	OSL_INLINE Wide() = default;
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
	// data members through back to original object
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
struct LaneProxy
{
	OSL_INLINE 
	LaneProxy(Wide<DataT, WidthT> & ref_wide_data, const int index)
	: m_ref_wide_data(ref_wide_data)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	LaneProxy(const LaneProxy &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	, m_index(other.m_index)
	{}
	
	OSL_INLINE 
	operator DataT const () const 
	{
		return m_ref_wide_data.get(m_index);
	}

	OSL_INLINE 
	DataT const get() const 
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
DataT const 
unproxy(const LaneProxy<DataT,WidthT> &proxy)
{
	return proxy.operator DataT const ();
}


template <typename DataT, int WidthT>
struct ConstLaneProxy
{
	OSL_INLINE
	ConstLaneProxy(const Wide<DataT, WidthT> & ref_wide_data, const int index)
	: m_ref_wide_data(ref_wide_data)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstLaneProxy(const ConstLaneProxy &other)
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
DataT const 
unproxy(const ConstLaneProxy<DataT,WidthT> &proxy)
{
	return proxy.operator DataT const ();
}


template <typename DataT, int WidthT>
struct WideAccessor
{
	static constexpr int width = WidthT; 
	
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
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	WideAccessor(const WideAccessor &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	{}	
	
	
	typedef ConstLaneProxy<DataT, WidthT> Proxy;
	
	OSL_INLINE Proxy const 
	operator[](int index) const
	{
		return Proxy(m_ref_wide_data, index);
	}
		
private:
	const Wide<DataT, WidthT> & m_ref_wide_data;	
};


template <typename DataT, int WidthT>
struct MaskedLaneProxy
{
	explicit OSL_INLINE 
	MaskedLaneProxy(Wide<DataT, WidthT> & ref_wide_data, const Mask & mask, const int index)
	: m_ref_wide_data(ref_wide_data)
	, m_mask(mask)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	MaskedLaneProxy(const MaskedLaneProxy &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	, m_mask(other.m_mask)
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
		if (m_mask[m_index]) {
			m_ref_wide_data.set(m_index, value);
		}
		return value;
	}
	
	// Although having free helper functions
	// might be cleaner, we choose to expose
	// this functionality here to increase 
	// visibility to end user whose IDE
	// might display these methods vs. free 
	// functions
    OSL_INLINE bool 
    is_on() const
    {
        return m_mask.is_on(m_index);
    }

    OSL_INLINE bool 
    is_off()
    {
        return m_mask.is_off(m_index);
    }
    
	OSL_INLINE 
	DataT const get() const 
	{
		return m_ref_wide_data.get(m_index);
	}
	
private:
	Wide<DataT, WidthT> & m_ref_wide_data;
	const Mask &m_mask;
	const int m_index;
};

template <typename DataT, int ArrayLenT, int WidthT>
struct MaskedArrayLaneProxy
{
	explicit OSL_INLINE 
	MaskedArrayLaneProxy(Wide<DataT, WidthT> * array_of_wide_data, const Mask & mask, const int index)
	: m_array_of_wide_data(array_of_wide_data)
	, m_mask(mask)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	MaskedArrayLaneProxy(const MaskedArrayLaneProxy &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_mask(other.m_mask)
	, m_index(other.m_index)
	{}
	
	OSL_INLINE 
	MaskedArrayLaneProxy &
	operator = (const DataT (&value) [ArrayLenT] )  
	{
		if (m_mask[m_index]) {
			for(int i=0; i < ArrayLenT; ++i) {
				m_array_of_wide_data[i].set(m_index, value[i]);
			}
		}
		return *this;
	}
	
	// Although having free helper functions
	// might be cleaner, we choose to expose
	// this functionality here to increase 
	// visibility to end user whose IDE
	// might display these methods vs. free 
	// functions
    OSL_INLINE bool 
    is_on() const
    {
        return m_mask.is_on(m_index);
    }

    OSL_INLINE bool 
    is_off()
    {
        return m_mask.is_off(m_index);
    }
 
	OSL_INLINE MaskedLaneProxy<DataT, WidthT> 
	operator[](int array_index) const 
	{
		return MaskedLaneProxy<DataT, WidthT>(m_array_of_wide_data[array_index], m_mask, m_index);
	}
	
	OSL_INLINE void 
	get(DataT (&value) [ArrayLenT]) const 
	{
		for(int i=0; i < ArrayLenT; ++i) {
			value[i] = m_array_of_wide_data[i].get(m_index);
		}
		return;
	}
	
private:
	Wide<DataT, WidthT> * m_array_of_wide_data;
	const Mask &m_mask;
	const int m_index;
};



template <typename DataT, int WidthT = SimdLaneCount>
struct MaskedAccessor
{
	static constexpr int width = WidthT; 
	
	explicit OSL_INLINE
	MaskedAccessor(void *ptr_wide_data, Mask mask, int derivIndex=0)
	: m_ref_wide_data(reinterpret_cast<Wide<DataT, WidthT> *>(ptr_wide_data)[derivIndex])
	, m_mask(mask)
	{}
	
	explicit OSL_INLINE
	MaskedAccessor(Wide<DataT, WidthT> & ref_wide_data, Mask mask)
	: m_ref_wide_data(ref_wide_data)
	, m_mask(mask)
	{}
	
	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	MaskedAccessor(const MaskedAccessor &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	, m_mask(other.m_mask)
	{}	
	
	
	typedef MaskedLaneProxy<DataT, WidthT> Proxy;
	
	OSL_INLINE Proxy  
	operator[](int index) 
	{
		return Proxy(m_ref_wide_data, m_mask, index);
	}

	OSL_INLINE Proxy const  
	operator[](int index) const
	{
		return Proxy(m_ref_wide_data, m_mask, index);
	}
	
private:
	Wide<DataT, WidthT> & m_ref_wide_data;
	Mask m_mask;
};

template <typename DataT, int ArrayLenT, int WidthT>
struct MaskedArrayAccessor
{
	static_assert(ArrayLenT > 0, "OSL logic bug");
	static constexpr int width = WidthT; 
	
	explicit OSL_INLINE
	MaskedArrayAccessor(void *ptr_wide_data, int derivIndex, Mask mask)
	: m_array_of_wide_data(&reinterpret_cast<Wide<DataT, WidthT> *>(ptr_wide_data)[ArrayLenT*derivIndex])
	, m_mask(mask)
	{}
	
	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	MaskedArrayAccessor(const MaskedArrayAccessor &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_mask(other.m_mask)
	{}	
	
	
	typedef MaskedArrayLaneProxy<DataT, ArrayLenT, WidthT> Proxy;
	
	OSL_INLINE Proxy  
	operator[](int index) 
	{
		return Proxy(m_array_of_wide_data, m_mask, index);
	}

	OSL_INLINE Proxy const  
	operator[](int index) const
	{
		return Proxy(m_array_of_wide_data, m_mask, index);
	}
	
private:
	Wide<DataT, WidthT> * m_array_of_wide_data;
	Mask m_mask;
};


template <int WidthT = SimdLaneCount>
class MaskedData
{
    void *m_ptr;
    TypeDesc m_type;
    Mask m_mask;
    bool m_has_derivs; 
public:
   MaskedData() = delete;
   
   explicit OSL_INLINE MaskedData(TypeDesc type, bool has_derivs, Mask mask, void *ptr)
   : m_ptr(ptr)
   , m_type(type)
   , m_mask(mask)
   , m_has_derivs(has_derivs)
   {}
   
	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
   OSL_INLINE MaskedData(const MaskedData &other)
   : m_ptr(other.m_ptr)
   , m_type(other.m_type)
   , m_mask(other.m_mask)
   , m_has_derivs(other.m_has_derivs)
   {}

   OSL_INLINE void *ptr() const { return m_ptr; }
   OSL_INLINE TypeDesc type() { return m_type; }
   OSL_INLINE bool has_derivs() { return m_has_derivs; }
   OSL_INLINE Mask mask() { return m_mask; }
   
protected:
   
   
   OSL_INLINE bool is_impl(float) const { return m_type == TypeDesc::TypeFloat; }
   OSL_INLINE bool is_impl(int) const { return m_type == TypeDesc::TypeInt; }
   OSL_INLINE bool is_impl(char *) const { return m_type == TypeDesc::TypeString; }
   OSL_INLINE bool is_impl(ustring) const { return m_type == TypeDesc::TypeString; }
   OSL_INLINE bool is_impl(Vec3) const { return m_type == TypeDesc::TypeVector; }
   OSL_INLINE bool is_impl(Color3) const { return m_type == TypeDesc::TypeColor; }
   OSL_INLINE bool is_impl(Matrix33) const { return m_type == TypeDesc::TypeMatrix33; }
   OSL_INLINE bool is_impl(Matrix44) const { return m_type == TypeDesc::TypeMatrix44; }

   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(float) const { return m_type == TypeDesc(TypeDesc::FLOAT, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(int) const { return m_type == TypeDesc(TypeDesc::INT32, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(char *) const { return m_type == TypeDesc(TypeDesc::STRING, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(ustring) const { return m_type == TypeDesc(TypeDesc::STRING, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Vec3) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::VECTOR, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Color3) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::COLOR, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Matrix33) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::MATRIX33, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Matrix44) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::MATRIX44, ArrayLenT); }

   
   template<typename DataT>
   OSL_INLINE bool 
   is_array_selector(std::false_type) {
	   return is_impl(DataT());
   }

   template<typename DataT>
   OSL_INLINE bool 
   is_array_selector(std::true_type) {
	   return is_array_impl<std::extent<DataT>::value>(std::remove_all_extents<DataT>::type());
   }
   
  
   template<typename DataT, int DerivIndexT>
   OSL_INLINE MaskedAccessor<DataT, WidthT>
   masked_impl(std::false_type) 
   { 
	   DASSERT(is<DataT>());
	   return MaskedAccessor<DataT, WidthT>(m_ptr, m_mask, DerivIndexT);
   }

   template<typename DataT, int DerivIndexT>
   OSL_INLINE MaskedArrayAccessor<typename std::remove_all_extents<DataT>::type, std::extent<DataT>::value, WidthT>
   masked_impl(std::true_type) 
   { 
	   DASSERT(is<DataT>());
	   return MaskedArrayAccessor<typename std::remove_all_extents<DataT>::type, std::extent<DataT>::value, WidthT>(m_ptr, DerivIndexT, m_mask);
   }
   
public:
   template<typename DataT>
   OSL_INLINE bool 
   is() {
	   return is_array_selector<DataT>(std::is_array<DataT>::type());
   }
   
   template<typename DataT>
   OSL_INLINE decltype(std::declval<MaskedData<WidthT>>().masked_impl<DataT, 0>(typename std::is_array<DataT>::type()))
   masked() 
   { 
	   return masked_impl<DataT, 0>(typename std::is_array<DataT>::type());
   }

   
   template<typename DataT>
   OSL_INLINE decltype(std::declval<MaskedData<WidthT>>().masked_impl<DataT, 1>(typename std::is_array<DataT>::type()))
   maskedDx() 
   {
	   DASSERT(has_derivs());
	   return masked_impl<DataT, 1>(typename std::is_array<DataT>::type());
   }

   template<typename DataT>
   OSL_INLINE decltype(std::declval<MaskedData<WidthT>>().masked_impl<DataT, 2>(typename std::is_array<DataT>::type()))
   maskedDy() 
   { 
	   DASSERT(has_derivs());
	   return masked_impl<DataT, 2>(typename std::is_array<DataT>::type());
   }
   
   template<typename DataT>
   OSL_INLINE decltype(std::declval<MaskedData<WidthT>>().masked_impl<DataT, 3>(typename std::is_array<DataT>::type()))
   maskedDz() 
   { 
	   DASSERT(has_derivs());
	   return masked_impl<DataT, 3>(typename std::is_array<DataT>::type());
   }   
};

typedef MaskedData<SimdLaneCount> MaskedDataRef;


// The RefProxy pretty much just allows "auto" to be used on the stack to 
// keep a reference vs. a copy of DataT
template <typename DataT>
struct RefProxy
{
	explicit OSL_INLINE 
	RefProxy(DataT & ref_data)
	: m_ref_data(ref_data)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	RefProxy(const RefProxy &other)
	: m_ref_data(other.m_ref_data)
	{}

	OSL_INLINE 
	operator DataT & () 
	{
		return m_ref_data;
	}
	
	OSL_INLINE 
	operator DataT const () const 
	{
		return m_ref_data;
	}

	OSL_INLINE const DataT &
	operator = (const DataT & value)  
	{
		m_ref_data = value;
		return value;
	}
private:
	DataT & m_ref_data;
};


template <typename DataT, int ArrayLenT>
struct RefArrayProxy
{
	OSL_INLINE 
	explicit RefArrayProxy(DataT (&ref_array_data)[ArrayLenT])
	: m_ref_array_data(ref_array_data)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	RefArrayProxy(const RefArrayProxy &other)
	: m_ref_array_data(other.m_ref_array_data)
	{}
	
	OSL_INLINE 
	RefArrayProxy &
	operator = (DataT (&value) [ArrayLenT] )
	{
		for(int i=0; i < ArrayLenT; ++i) {
			m_ref_array_data[i] = value[i];
		}
		return *this;
	}

	typedef DataT (&ArrayRefType)[ArrayLenT];
	
	OSL_INLINE
	operator ArrayRefType()
	{
		return m_ref_array_data;
	}

	
	OSL_INLINE DataT & 
	operator[](int array_index)  
	{
		DASSERT(array_index >= 0 && array_index < ArrayLenT);
		return m_ref_array_data[array_index];
	}

	OSL_INLINE DataT const & 
	operator[](int array_index) const  
	{
		DASSERT(array_index >= 0 && array_index < ArrayLenT);
		return m_ref_array_data[array_index];
	}
	
	OSL_INLINE void
	get(DataT (&value) [ArrayLenT]) const
	{
		for(int i=0; i < ArrayLenT; ++i) {
			value[i] = m_ref_array_data[i];
		}
		return;
	}
	
private:
	DataT (&m_ref_array_data)[ArrayLenT];
};


class DataRef
{
    void *m_ptr;
    TypeDesc m_type;
    bool m_has_derivs; 
public:
   DataRef() = delete;
   
   explicit OSL_INLINE DataRef(TypeDesc type, bool has_derivs, void *ptr)
   : m_ptr(ptr)
   , m_type(type)
   , m_has_derivs(has_derivs)
   {}
   
	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
   OSL_INLINE DataRef(const DataRef &other)
   : m_ptr(other.m_ptr)
   , m_type(other.m_type)
   , m_has_derivs(other.m_has_derivs)
   {}
    
   OSL_INLINE void *ptr() const { return m_ptr; }
   OSL_INLINE TypeDesc type() { return m_type; }
   OSL_INLINE bool has_derivs() { return m_has_derivs; }
   
protected:
   
   // TODO: see if impl can be shared with MaskedData
   OSL_INLINE bool is_impl(float) const { return m_type == TypeDesc::TypeFloat; }
   OSL_INLINE bool is_impl(int) const { return m_type == TypeDesc::TypeInt; }
   OSL_INLINE bool is_impl(char *) const { return m_type == TypeDesc::TypeString; }
   OSL_INLINE bool is_impl(ustring) const { return m_type == TypeDesc::TypeString; }
   OSL_INLINE bool is_impl(Vec3) const { return m_type == TypeDesc::TypeVector; }
   OSL_INLINE bool is_impl(Color3) const { return m_type == TypeDesc::TypeColor; }
   OSL_INLINE bool is_impl(Matrix33) const { return m_type == TypeDesc::TypeMatrix33; }
   OSL_INLINE bool is_impl(Matrix44) const { return m_type == TypeDesc::TypeMatrix44; }

   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(float) const { return m_type == TypeDesc(TypeDesc::FLOAT, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(int) const { return m_type == TypeDesc(TypeDesc::INT32, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(char *) const { return m_type == TypeDesc(TypeDesc::STRING, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(ustring) const { return m_type == TypeDesc(TypeDesc::STRING, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Vec3) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::VECTOR, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Color3) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::VEC3, TypeDesc::COLOR, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Matrix33) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::MATRIX33, ArrayLenT); }
   template <int ArrayLenT>
   OSL_INLINE bool is_array_impl(Matrix44) const { return m_type == TypeDesc(TypeDesc::FLOAT, TypeDesc::MATRIX44, ArrayLenT); }

   
   template<typename DataT>
   OSL_INLINE bool 
   is_array_selector(std::false_type) {
	   return is_impl(DataT());
   }

   template<typename DataT>
   OSL_INLINE bool 
   is_array_selector(std::true_type) {
	   return is_array_impl<std::extent<DataT>::value>(std::remove_all_extents<DataT>::type());
   }
   
  
   template<typename DataT, int DerivIndexT>
   //OSL_INLINE DataT &
   OSL_INLINE RefProxy<DataT>
   ref_impl(std::false_type) 
   { 
	   DASSERT(is<DataT>());	   
	   //return reinterpret_cast<DataT *>(m_ptr)[DerivIndexT];
	   return RefProxy<DataT>(reinterpret_cast<DataT *>(m_ptr)[DerivIndexT]);
   }

   // TODO: consider returning a proxy to enable array index bounds checking
   template<typename DataT, int DerivIndexT>
   OSL_INLINE RefArrayProxy<typename std::remove_all_extents<DataT>::type, std::extent<DataT>::value> 
   ref_impl(std::true_type) 
   { 
	   DASSERT(is<DataT>());
	   //return &(reinterpret_cast<DataT *>(m_ptr)[std::extent<DataT>::value*derivIndex]);
	   
	   return RefArrayProxy<typename std::remove_all_extents<DataT>::type, std::extent<DataT>::value>(reinterpret_cast<DataT *>(m_ptr)[DerivIndexT]);
   }
   
public:
   template<typename DataT>
   OSL_INLINE bool 
   is() {
	   return is_array_selector<DataT>(std::is_array<DataT>::type());
   }
   
   template<typename DataT>
   OSL_INLINE decltype(std::declval<DataRef>().ref_impl<DataT, 0>(typename std::is_array<DataT>::type()))
   ref() 
   { 
	   return ref_impl<DataT, 0>(typename std::is_array<DataT>::type());
   }

   
   template<typename DataT>
   OSL_INLINE decltype(std::declval<DataRef>().ref_impl<DataT, 1>(typename std::is_array<DataT>::type()))
   refDx() 
   {
	   DASSERT(has_derivs());
	   return ref_impl<DataT, 1>(typename std::is_array<DataT>::type());
   }

   template<typename DataT>
   OSL_INLINE decltype(std::declval<DataRef>().ref_impl<DataT, 2>(typename std::is_array<DataT>::type()))
   refDy() 
   { 
	   DASSERT(has_derivs());
	   return ref_impl<DataT, 2>(typename std::is_array<DataT>::type());
   }
   
   template<typename DataT>
   OSL_INLINE decltype(std::declval<DataRef>().ref_impl<DataT, 3>(typename std::is_array<DataT>::type()))
   refDz() 
   { 
	   DASSERT(has_derivs());
	   return ref_impl<DataT, 3>(typename std::is_array<DataT>::type());
   }   
};



OSL_NAMESPACE_EXIT
