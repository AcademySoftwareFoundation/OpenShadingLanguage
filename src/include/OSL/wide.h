/*
Copyright (c) 2017 Intel Inc., et al.
All Rights Reserved.
 
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
#include "Imathx.h"

OSL_NAMESPACE_ENTER

// TODO: add conditional compilation to change this
static constexpr int SimdLaneCount = 16;
#define __OSL_SIMD_LANE_COUNT 16


/// Type for an opaque pointer to whatever the renderer uses to represent a
/// coordinate transformation.
typedef const void * TransformationPtr;

// Simple wrapper to identify a single lane index vs. a mask_value
class Lane
{
	const int m_index;
public:
	explicit OSL_INLINE
	Lane(int index)
	: m_index(index)
	{}

	Lane() = delete;

    OSL_INLINE Lane(const Lane &other)
        : m_index(other.m_index)
    {}

    OSL_INLINE int
	value() const {
    	return m_index;
    }
};

template <int WidthT>
class WideMask
{
public:
    typedef unsigned int value_type;
    static_assert(sizeof(value_type)*8 >= WidthT, "unsupported WidthT");
	static constexpr int width = WidthT; 

    OSL_INLINE WideMask()
    {}

    explicit OSL_INLINE
	WideMask(Lane lane)
    : m_value(1<<lane.value())
    {}

    explicit OSL_INLINE
	WideMask(bool all_on_or_off)
    : m_value((all_on_or_off) ? (0xFFFFFFFF >> (32-WidthT)) : 0)
    {}
    
    explicit OSL_INLINE
	WideMask(value_type value_)
        : m_value(value_)
    {}

    explicit OSL_INLINE
	WideMask(int value_)
        : m_value(static_cast<value_type>(value_))
    {}

    OSL_INLINE WideMask(const WideMask &other)
        : m_value(other.m_value)
    {}

    OSL_INLINE value_type value() const
    { return m_value; }

    // count number of active bits
    OSL_INLINE int count() const {
        value_type m(m_value);
        int count = 0;
        for (count = 0; m != 0; ++count) {
            m &= m-1;
        }
        return count;
    }

    
    OSL_INLINE WideMask invert() const
    {
    	return WideMask((~m_value)&(0xFFFFFFFF >> (32-WidthT)));
    }

    OSL_INLINE WideMask invert(const WideMask &mask) const
    {
        return WideMask(mask.m_value&((~m_value)&(0xFFFFFFFF >> (32-WidthT))));
    }

    
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
        return (m_value >= (0xFFFFFFFF >> (32-WidthT)));
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
        return (m_value < (0xFFFFFFFF >> (32-WidthT)));
    }

    OSL_INLINE bool any_off(const WideMask &mask) const
    {
        return m_value != (m_value & mask.m_value);
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

    OSL_INLINE bool
    operator == (const WideMask &other) const
    {
        return m_value == other.m_value;
    }

    OSL_INLINE bool
    operator != (const WideMask &other) const
    {
        return m_value != other.m_value;
    }
    
    OSL_INLINE WideMask & 
    operator &=(const WideMask &other)
    {
        m_value = m_value&other.m_value;
        return *this;
    }

    OSL_INLINE WideMask & 
    operator |=(const WideMask &other)
    {
        m_value = m_value|other.m_value;
        return *this;
    }

    OSL_INLINE WideMask  
    operator & (const WideMask &other) const
    {
        return WideMask(m_value&other.m_value);
    }

    OSL_INLINE WideMask  
    operator | (const WideMask &other) const
    {
        return WideMask(m_value|other.m_value);
    }

    OSL_INLINE WideMask  
    operator ~() const
    {
        return invert();
    }
    
private:
    value_type m_value;
};

typedef WideMask<SimdLaneCount> Mask;
// Technically identical to Mask, but intended use is that 
// the implementor may ignore the mask and populate
// all data lanes of the destination object, however
// implementor may still find it usefull to avoid
// pulling/gathering data for that lane.
// Intent is for self documenting code
typedef WideMask<SimdLaneCount> WeakMask;



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
		OSL_INTEL_PRAGMA(forceinline recursive)
		{
			OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
			for(int i = 0; i < WidthT; ++i)
			{
					data[i] = value;
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
	explicit OSL_INLINE
	WideBuiltin(const BuiltinListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(BuiltinListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}
	
	OSL_INLINE explicit  
	WideBuiltin(const value_type & uniformValue) 
	{
		OSL_INTEL_PRAGMA(forceinline recursive)
		{
			OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
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


// Vec4 isn't used by external interfaces, but some internal
// noise functions utilize a wide version of it.
typedef Imath::Vec4<Float>     Vec4;

template <int WidthT>
struct Wide<Vec4, WidthT>
{
	typedef Vec4 value_type;
	static constexpr int width = WidthT;
	float x[WidthT];
	float y[WidthT];
	float z[WidthT];
	float w[WidthT];

	OSL_INLINE void
	set(int index, const Vec4 & value)
	{
		x[index] = value.x;
		y[index] = value.y;
		z[index] = value.z;
		w[index] = value.w;
	}

protected:
	template<int HeadIndexT>
	OSL_INLINE void
	set(internal::int_sequence<HeadIndexT>, const Vec4 & value)
	{
		set(HeadIndexT, value);
	}

	template<int HeadIndexT, int... TailIndexListT, typename... Vec4ListT>
	OSL_INLINE void
	set(internal::int_sequence<HeadIndexT, TailIndexListT...>, Vec4 headValue, Vec4ListT... tailValues)
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

	template<typename... Vec4ListT, typename = internal::enable_if_type<(sizeof...(Vec4ListT) == WidthT)> >
	explicit OSL_INLINE
	Wide(const Vec4ListT &...values)
	{
		typedef internal::make_int_sequence<sizeof...(Vec4ListT)> int_seq_type;
		set(int_seq_type(), values...);
		return;
	}


	OSL_INLINE Vec4
	get(int index) const
	{
		return Vec4(x[index], y[index], z[index], w[index]);
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
		std::cout << "w{"	;
		for(int i=0; i < WidthT; ++i)
		{
			std::cout << w[i];
			if (i < (WidthT-1))
				std::cout << ",";

		}
		std::cout << "}" << std::endl;
	}

};

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
	explicit OSL_INLINE
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
struct Wide<Vec2, WidthT>
{
    typedef Vec2 value_type;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];

    OSL_INLINE void
    set(int index, const Vec2 & value)
    {
        x[index] = value.x;
        y[index] = value.y;
    }

protected:
    template<int HeadIndexT>
    OSL_INLINE void
    set(internal::int_sequence<HeadIndexT>, const Vec2 & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Vec2ListT>
    OSL_INLINE void
    set(internal::int_sequence<HeadIndexT, TailIndexListT...>, Vec2 headValue, Vec2ListT... tailValues)
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

    template<typename... Vec2ListT, typename = internal::enable_if_type<(sizeof...(Vec2ListT) == WidthT)> >
    explicit OSL_INLINE
    Wide(const Vec2ListT &...values)
    {
        typedef internal::make_int_sequence<sizeof...(Vec2ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_INLINE Vec2
    get(int index) const
    {
        return Vec2(x[index], y[index]);
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
    explicit OSL_INLINE
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


#if 0
// Considering having functionally equivalent versions of Vec3, Color3, Matrix44
// with slight modifications to inlining and implmentation to avoid aliasing and
// improve likelyhood of proper privation of local variables within a SIMD loop

template <int WidthT>
struct Wide<fast::Color3, WidthT>
{
	typedef fast::Color3 value_type;
    static constexpr int width = WidthT;
    float x[WidthT];
    float y[WidthT];
    float z[WidthT];

    OSL_INLINE void
    set(int index, const value_type & value)
    {
        x[index] = value.x;
        y[index] = value.y;
        z[index] = value.z;
    }

protected:
    template<int HeadIndexT>
    OSL_INLINE void
    set(internal::int_sequence<HeadIndexT>, const value_type & value)
    {
        set(HeadIndexT, value);
    }

    template<int HeadIndexT, int... TailIndexListT, typename... Color3ListT>
    OSL_INLINE void
    set(internal::int_sequence<HeadIndexT, TailIndexListT...>, value_type headValue, Color3ListT... tailValues)
    {
        set(HeadIndexT, headValue);
        set(internal::int_sequence<TailIndexListT...>(), tailValues...);
        return;
    }
public:

    OSL_INLINE Wide() = default;
    Wide(const Wide &other) = delete;

    template<typename... Color3ListT, typename = internal::enable_if_type<(sizeof...(Color3ListT) == WidthT)> >
    explicit OSL_INLINE
    Wide(const Color3ListT &...values)
    {
        typedef internal::make_int_sequence<sizeof...(Color3ListT)> int_seq_type;
        set(int_seq_type(), values...);
        return;
    }


    OSL_INLINE value_type
    get(int index) const
    {
        return value_type(x[index], y[index], z[index]);
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
#endif

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
	explicit OSL_INLINE
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
	Wide<Vec3, WidthT> x;
	Wide<Vec3, WidthT> dx;
	Wide<Vec3, WidthT> dy;
	
	OSL_INLINE void 
	set(int index, const value_type & value) 
	{
		x.set(index, value.val());
		dx.set(index, value.dx());
		dy.set(index, value.dy());
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
	explicit OSL_INLINE
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

template <int WidthT>
struct Wide<Dual2<Color3>, WidthT>
{
	typedef Dual2<Color3> value_type;
	static constexpr int width = WidthT;
	Wide<Color3, WidthT> x;
	Wide<Color3, WidthT> dx;
	Wide<Color3, WidthT> dy;

	OSL_INLINE void
	set(int index, const value_type & value)
	{
		x.set(index, value.val());
		dx.set(index, value.dx());
		dy.set(index, value.dy());
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
	explicit OSL_INLINE
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
	explicit OSL_INLINE
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
		OSL_INTEL_PRAGMA(forceinline recursive)
		{
			OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
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
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
		for(int i = 0; i < WidthT; ++i) {
			wide_data.set(i, value);
		}
	}
}


template <typename DataT, int WidthT>
struct LaneProxy
{
	explicit OSL_INLINE
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
	explicit OSL_INLINE
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
struct ConstDual2LaneProxy
{
	explicit OSL_INLINE
	ConstDual2LaneProxy(const Wide<DataT, WidthT> *array_of_wide_data, const int lane_index, const int array_index, const int array_length)
	: m_array_of_wide_data(array_of_wide_data)
	, m_array_index(array_index)
    , m_lane_index(lane_index)
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstDual2LaneProxy(const ConstDual2LaneProxy &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_array_index(other.m_array_index)
    , m_lane_index(other.m_lane_index)
	, m_array_length(other.m_array_length)
	{}

	OSL_INLINE
	operator Dual2<DataT> const () const
	{
		return Dual2<DataT>(m_array_of_wide_data[m_array_index].get(m_lane_index),
							m_array_of_wide_data[m_array_length + m_array_index].get(m_lane_index),
							m_array_of_wide_data[2*m_array_length + m_array_index].get(m_lane_index));
	}

private:
	const Wide<DataT, WidthT> * m_array_of_wide_data;
	const int m_array_index;
	const int m_lane_index;
	const int m_array_length;
};



template <typename DataT, int WidthT>
Dual2<DataT> const
unproxy(const ConstDual2LaneProxy<DataT,WidthT> &proxy)
{
	return proxy.operator Dual2<DataT> const ();
}


template <typename DataT, int WidthT = SimdLaneCount>
struct WideAccessor
{
	template <typename OtherDataT, int OtherWidthT>
	friend struct ConstWideAccessor;

	static constexpr int width = WidthT; 
	typedef DataT value_type;
	
	explicit OSL_INLINE
	WideAccessor(void *ptr_wide_data, int derivIndex=0)
	: m_ref_wide_data(reinterpret_cast<Wide<DataT, WidthT> *>(ptr_wide_data)[derivIndex])
	{}
	
	// Allow implicit construction
	OSL_INLINE
	WideAccessor(Wide<DataT, WidthT> & ref_wide_data)
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
	
	
	OSL_INLINE Wide<DataT, WidthT> &  data() const { return m_ref_wide_data; }

	typedef LaneProxy<DataT, WidthT> Proxy;
	typedef ConstLaneProxy<DataT, WidthT> ConstProxy;
	
	OSL_INLINE ConstProxy const 
	operator[](int index) const
	{
		return ConstProxy(m_ref_wide_data, index);
	}

	OSL_INLINE Proxy
	operator[](int index)
	{
		return Proxy(m_ref_wide_data, index);
	}

private:
	Wide<DataT, WidthT> & m_ref_wide_data;
};


template <typename DataT, int WidthT = SimdLaneCount>
struct ConstWideAccessor
{
	static constexpr int width = WidthT; 
	typedef DataT value_type;
	
	explicit OSL_INLINE
	ConstWideAccessor(const void *ptr_wide_data, int derivIndex=0)
	: m_ref_wide_data(reinterpret_cast<const Wide<DataT, WidthT> *>(ptr_wide_data)[derivIndex])
	{}
	
	// Allow implicit construction
	OSL_INLINE
	ConstWideAccessor(const Wide<DataT, WidthT> & ref_wide_data)
	: m_ref_wide_data(ref_wide_data)
	{}

	// Allow implicit conversion construction
	OSL_INLINE
	ConstWideAccessor(const WideAccessor<DataT, WidthT> &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	{}
	
	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstWideAccessor(const ConstWideAccessor &other)
	: m_ref_wide_data(other.m_ref_wide_data)
	{}	
	
	
	typedef ConstLaneProxy<DataT, WidthT> ConstProxy;
	
    OSL_INLINE const Wide<DataT, WidthT> &  data() const { return m_ref_wide_data; }

	OSL_INLINE ConstProxy const 
	operator[](int index) const
	{
		return ConstProxy(m_ref_wide_data, index);
	}

private:
	const Wide<DataT, WidthT> & m_ref_wide_data;
};


template <typename DataT>
struct ConstUniformProxy
{
	explicit OSL_INLINE
	ConstUniformProxy(const DataT & ref_uniform_data)
	: m_ref_uniform_data(ref_uniform_data)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstUniformProxy(const ConstUniformProxy &other)
	: m_ref_uniform_data(other.m_ref_uniform_data)
	{}

	OSL_INLINE
	operator const DataT & () const
	{
		return m_ref_uniform_data;
	}

private:
	const DataT & m_ref_uniform_data;
};

template <typename DataT, int WidthT = SimdLaneCount>
struct ConstUniformAccessor
{
	static constexpr int width = WidthT;
	typedef DataT value_type;

	explicit OSL_INLINE
	ConstUniformAccessor(const void *ptr_uniform_data, int derivIndex=0)
	: m_ref_uniform_data(reinterpret_cast<const DataT *>(ptr_uniform_data)[derivIndex])
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstUniformAccessor(const ConstUniformAccessor &other)
	: m_ref_uniform_data(other.m_ref_uniform_data)
	{}

	OSL_INLINE const DataT & data() const { return m_ref_uniform_data; }


	typedef ConstUniformProxy<DataT> ConstProxy;

	OSL_INLINE ConstProxy const
	operator[](int index) const
	{
		return ConstProxy(m_ref_uniform_data);
	}

private:
	const DataT & m_ref_uniform_data;
};


template <typename DataT, int WidthT>
struct ConstWideUnboundedArrayLaneProxy
{
	explicit OSL_INLINE
	ConstWideUnboundedArrayLaneProxy(const Wide<DataT, WidthT> * array_of_wide_data, int array_length, int index)
	: m_array_of_wide_data(array_of_wide_data)
	, m_array_length(array_length)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstWideUnboundedArrayLaneProxy(const ConstWideUnboundedArrayLaneProxy &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_array_length(other.m_array_length)
	, m_index(other.m_index)
	{}

	OSL_INLINE int
	length() const { return m_array_length; }

	OSL_INLINE ConstLaneProxy<DataT, WidthT>
	operator[](int array_index) const
	{
		DASSERT(array_index < m_array_length);
		return ConstLaneProxy<DataT, WidthT>(m_array_of_wide_data[array_index], m_index);
	}

private:
	const Wide<DataT, WidthT> * m_array_of_wide_data;
	int m_array_length;
	const int m_index;
};



template <typename DataT, int WidthT>
struct ConstWideDual2UnboundedArrayLaneProxy
{
	explicit OSL_INLINE
	ConstWideDual2UnboundedArrayLaneProxy(const Wide<DataT, WidthT> * array_of_wide_data, int array_length, int lane_index)
	: m_array_of_wide_data(array_of_wide_data)
	, m_array_length(array_length)
	, m_lane_index(lane_index)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstWideDual2UnboundedArrayLaneProxy(const ConstWideDual2UnboundedArrayLaneProxy &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_array_length(other.m_array_length)
	, m_lane_index(other.m_lane_index)
	{}

	OSL_INLINE int
	length() const { return m_array_length; }

	OSL_INLINE ConstDual2LaneProxy<DataT, WidthT>
	operator[](int array_index) const
	{
		DASSERT(array_index < m_array_length);
		return ConstDual2LaneProxy<DataT, WidthT>(m_array_of_wide_data, m_lane_index, array_index, m_array_length);
	}

private:
	const Wide<DataT, WidthT> * m_array_of_wide_data;
	int m_array_length;
	const int m_lane_index;
};

template <typename DataT, int WidthT = SimdLaneCount>
struct ConstWideUnboundArrayAccessor
{
	static constexpr int width = WidthT;
    typedef DataT value_type;

	explicit OSL_INLINE
	ConstWideUnboundArrayAccessor(const void *ptr_wide_data, int array_length)
	: m_array_of_wide_data(reinterpret_cast<const Wide<DataT, WidthT> *>(ptr_wide_data))
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstWideUnboundArrayAccessor(const ConstWideUnboundArrayAccessor &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_array_length(other.m_array_length)
	{}


	typedef ConstWideUnboundedArrayLaneProxy<DataT, WidthT> Proxy;

	OSL_INLINE Proxy const
	operator[](int index) const
	{
		return Proxy(m_array_of_wide_data, m_array_length, index);
	}

private:
	const Wide<DataT, WidthT> * m_array_of_wide_data;
	int m_array_length;
};


template <typename DataT, int WidthT>
struct ConstWideUnboundArrayAccessor<Dual2<DataT>, WidthT>
{
	static constexpr int width = WidthT;
    typedef Dual2<DataT> value_type;

	explicit OSL_INLINE
	ConstWideUnboundArrayAccessor(const void *ptr_wide_data, int array_length)
	: m_array_of_wide_data(reinterpret_cast<const Wide<DataT, WidthT> *>(ptr_wide_data))
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstWideUnboundArrayAccessor(const ConstWideUnboundArrayAccessor &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_array_length(other.m_array_length)
	{}


	typedef ConstWideDual2UnboundedArrayLaneProxy<DataT, WidthT> Proxy;

	OSL_INLINE Proxy const
	operator[](int lane_index) const
	{
		return Proxy(m_array_of_wide_data, m_array_length, lane_index);
	}

private:
	const Wide<DataT, WidthT> * m_array_of_wide_data;
	int m_array_length;
};

template <typename DataT, int WidthT>
struct ConstUniformUnboundedArrayLaneProxy
{
	explicit OSL_INLINE
	ConstUniformUnboundedArrayLaneProxy(const DataT * array_of_data, int array_length)
	: m_array_of_data(array_of_data)
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstUniformUnboundedArrayLaneProxy(const ConstUniformUnboundedArrayLaneProxy &other)
	: m_array_of_data(other.m_array_of_data)
	, m_array_length(other.m_array_length)
	{}

	OSL_INLINE int
	length() const { return m_array_length; }

	OSL_INLINE const DataT &
	operator[](int array_index) const
	{
		DASSERT(array_index < m_array_length);
		return m_array_of_data[array_index];
	}

private:
	const DataT * m_array_of_data;
	int m_array_length;
};



template <typename DataT, int WidthT>
struct ConstUniformUnboundedDual2ArrayLaneProxy
{
	explicit OSL_INLINE
	ConstUniformUnboundedDual2ArrayLaneProxy(const DataT * array_of_data, int array_length)
	: m_array_of_data(array_of_data)
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstUniformUnboundedDual2ArrayLaneProxy(const ConstUniformUnboundedDual2ArrayLaneProxy &other)
	: m_array_of_data(other.m_array_of_data)
	, m_array_length(other.m_array_length)
	{}

	OSL_INLINE int
	length() const { return m_array_length; }

	OSL_INLINE const Dual2<DataT>
	operator[](int array_index) const
	{
		DASSERT(array_index < m_array_length);
		return Dual2<DataT>(m_array_of_data[array_index],
				            m_array_of_data[1*m_array_length + array_index],
							m_array_of_data[2*m_array_length + array_index]);
	}

private:
	const DataT * m_array_of_data;
	int m_array_length;
};

template <typename DataT, int WidthT = SimdLaneCount>
struct ConstUniformUnboundedArrayAccessor
{
	static constexpr int width = WidthT;
    typedef DataT value_type;


	explicit OSL_INLINE
	ConstUniformUnboundedArrayAccessor(const void *ptr_data, int array_length)
	: m_array_of_data(reinterpret_cast<const DataT *>(ptr_data))
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstUniformUnboundedArrayAccessor(const ConstUniformUnboundedArrayAccessor &other)
	: m_array_of_data(other.m_array_of_data)
	, m_array_length(other.m_array_length)
	{}


	typedef ConstUniformUnboundedArrayLaneProxy<DataT, WidthT> Proxy;

	OSL_INLINE Proxy const
	operator[](int index) const
	{
		return Proxy(m_array_of_data, m_array_length);
	}

private:
	const DataT * m_array_of_data;
	int m_array_length;
};


template <typename DataT, int WidthT>
struct ConstUniformUnboundedArrayAccessor<Dual2<DataT>, WidthT>
{
	static constexpr int width = WidthT;
    typedef Dual2<DataT> value_type;

	explicit OSL_INLINE
	ConstUniformUnboundedArrayAccessor(const void *ptr_data, int array_length)
	: m_array_of_data(reinterpret_cast<const DataT *>(ptr_data))
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to
	// get compiler to be able to follow individual
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	ConstUniformUnboundedArrayAccessor(const ConstUniformUnboundedArrayAccessor &other)
	: m_array_of_data(other.m_array_of_data)
	, m_array_length(other.m_array_length)
	{}


	typedef ConstUniformUnboundedDual2ArrayLaneProxy<DataT, WidthT> Proxy;

	OSL_INLINE Proxy const
	operator[](int index) const
	{
		return Proxy(m_array_of_data, m_array_length);
	}

private:
	const DataT * m_array_of_data;
	int m_array_length;
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
		if (m_mask[m_index])
		{
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

template <typename DataT, int WidthT>
DataT const
unproxy(const MaskedLaneProxy<DataT,WidthT> &proxy)
{
	return proxy.operator DataT const ();
}



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


template <typename DataT, int WidthT>
struct MaskedUnboundedArrayLaneProxy
{
	explicit OSL_INLINE 
	MaskedUnboundedArrayLaneProxy(Wide<DataT, WidthT> * array_of_wide_data, int array_length, const Mask & mask, int index)
	: m_array_of_wide_data(array_of_wide_data)
	, m_array_length(array_length)
	, m_mask(mask)
	, m_index(index)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	MaskedUnboundedArrayLaneProxy(const MaskedUnboundedArrayLaneProxy &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_array_length(other.m_array_length)
	, m_mask(other.m_mask)
	, m_index(other.m_index)
	{}
	
	OSL_INLINE int 
	length() const { return m_array_length; }
	
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
		DASSERT(array_index < m_array_length);
		return MaskedLaneProxy<DataT, WidthT>(m_array_of_wide_data[array_index], m_mask, m_index);
	}
	
	
private:
	Wide<DataT, WidthT> * m_array_of_wide_data;
	int m_array_length;
	const Mask &m_mask;
	const int m_index;
};



template <typename DataT, int WidthT = SimdLaneCount>
struct MaskedAccessor
{
	static constexpr int width = WidthT; 
	typedef DataT value_type;
	
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
	
	
    OSL_INLINE Wide<DataT, WidthT> &  data() const { return m_ref_wide_data; }
    OSL_INLINE Mask mask() const { return m_mask; }

    OSL_INLINE MaskedAccessor operator & (const Mask &mask) const
	{
    	return MaskedAccessor(m_ref_wide_data, m_mask & mask);
	}

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

template <typename DataT, int WidthT>
OSL_INLINE void
make_uniform(MaskedAccessor<DataT, WidthT> &wide_data, const DataT &value)
{
	OSL_INTEL_PRAGMA(forceinline recursive)
	{
		OSL_OMP_PRAGMA(omp simd simdlen(WidthT))
		for(int i = 0; i < WidthT; ++i) {
			wide_data[i] = value;
		}
	}
}


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

template <typename DataT, int WidthT>
struct MaskedUnboundArrayAccessor
{
	static constexpr int width = WidthT; 
	
	explicit OSL_INLINE
	MaskedUnboundArrayAccessor(void *ptr_wide_data, int derivIndex, int array_length, Mask mask)
	: m_array_of_wide_data(&reinterpret_cast<Wide<DataT, WidthT> *>(ptr_wide_data)[array_length*derivIndex])
	, m_array_length(array_length)
	, m_mask(mask)
	{}
	
	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	MaskedUnboundArrayAccessor(const MaskedUnboundArrayAccessor &other)
	: m_array_of_wide_data(other.m_array_of_wide_data)
	, m_array_length(other.m_array_length)
	, m_mask(other.m_mask)
	{}	
	
	
	typedef MaskedUnboundedArrayLaneProxy<DataT, WidthT> Proxy;
	
	OSL_INLINE Proxy  
	operator[](int index) 
	{
		return Proxy(m_array_of_wide_data, m_array_length, m_mask, index);
	}

	OSL_INLINE Proxy const  
	operator[](int index) const
	{
		return Proxy(m_array_of_wide_data, m_array_length, m_mask, index);
	}
	
private:
	Wide<DataT, WidthT> * m_array_of_wide_data;
	int m_array_length;
	Mask m_mask;
};


// End users can add specialize wide for their own types
// and specialize traits to enable them to be used in the proxies
// NOTE: array detection is handled separately
template <typename DataT>
struct WideTraits; // undefined, all types used should be specialized
//{
	//static bool mathes(const TypeDesc &) { return false; }
//};

template <>
struct WideTraits<float> {
	static bool matches(const TypeDesc &type_desc) { 
		return type_desc.basetype == TypeDesc::FLOAT && 
		       type_desc.aggregate == TypeDesc::SCALAR; 
	}
};

template <>
struct WideTraits<int> {
	static bool matches(const TypeDesc &type_desc) { 
		return type_desc.basetype == TypeDesc::INT && 
		       type_desc.aggregate == TypeDesc::SCALAR; 
	}
};

template <>
struct WideTraits<char *> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::STRING && 
               type_desc.aggregate == TypeDesc::SCALAR; 
	}
};

template <>
struct WideTraits<ustring> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::STRING && 
               type_desc.aggregate == TypeDesc::SCALAR; 
	}
};

// We let Vec3 match any vector semantics as we don't have a seperate Point or Normal classes
template <>
struct WideTraits<Vec3> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::FLOAT && 
		    type_desc.aggregate == TypeDesc::VEC3; 
	}
};

template <>
struct WideTraits<Vec2> {
    static bool matches(const TypeDesc &type_desc) {
        return type_desc.basetype == TypeDesc::FLOAT &&
            type_desc.aggregate == TypeDesc::VEC2;
    }
};

template <>
struct WideTraits<Color3> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::FLOAT && 
		    type_desc.aggregate == TypeDesc::VEC3 &&
			type_desc.vecsemantics == TypeDesc::COLOR; 
	}
};

#if 0
template <>
struct WideTraits<fast::Color3> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::FLOAT &&
		    type_desc.aggregate == TypeDesc::VEC3 &&
			type_desc.vecsemantics == TypeDesc::COLOR;
	}
};
#endif

template <>
struct WideTraits<Matrix33> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::FLOAT && 
		type_desc.aggregate == TypeDesc::MATRIX33;
	}
};

template <>
struct WideTraits<Matrix44> {
	static bool matches(const TypeDesc &type_desc) {
		return type_desc.basetype == TypeDesc::FLOAT && 
		    type_desc.aggregate == TypeDesc::MATRIX44; }
};



template <int WidthT = SimdLaneCount>
class MaskedData
{
    void *m_ptr;
    TypeDesc m_type;
    Mask m_mask;
    bool m_has_derivs; 
public:

   static constexpr int width = WidthT;

   MaskedData() = delete;
   
   explicit OSL_INLINE
   MaskedData(TypeDesc type, bool has_derivs, Mask mask, void *ptr)
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
   OSL_INLINE TypeDesc type() const { return m_type; }
   OSL_INLINE bool has_derivs() const { return m_has_derivs; }
   OSL_INLINE Mask mask() const { return m_mask; }
   OSL_INLINE bool valid() const { return m_ptr != nullptr; }

protected:
   
   // C++11 doesn't support auto return types, so we must use
   // template specialization to determine the return type of the
   // masked accessor.  Once C++14 is baseline requirement, we could
   // simplify this a little bit

   template <typename DataT, bool IsArrayT, bool IsArrayUnboundedT>
   struct AccessorFactory; // undefined

   template <typename DataT>
   struct AccessorFactory<DataT, false, true>
   {
	  typedef MaskedAccessor<DataT, WidthT> value_type;

	  static OSL_INLINE bool
	  matches(MaskedData &md)
	  {
		  return (md.m_type.arraylen == 0) &&
				  WideTraits<DataT>::matches(md.m_type);
	  }

	  template<int DerivIndexT>
	  static OSL_INLINE value_type
	  build(MaskedData &md)
	  {
		  DASSERT(matches(md));
	   	  return value_type(md.m_ptr, md.m_mask, DerivIndexT);
	  }
   };

   template <typename DataT>
   struct AccessorFactory<DataT, true /* IsArrayT */, false /* IsArrayUnboundedT */>
   {
	  typedef typename std::remove_all_extents<DataT>::type ElementType;
	  typedef MaskedArrayAccessor<ElementType, std::extent<DataT>::value, WidthT> value_type;

	  static OSL_INLINE bool
	  matches(MaskedData &md)
	  {
		  return (md.m_type.arraylen == std::extent<DataT>::value) &&
		  		  WideTraits<ElementType>::matches(md.m_type);
	  }

	  template<int DerivIndexT>
	  static OSL_INLINE value_type
	  build(MaskedData &md)
	  {
		  DASSERT(matches(md));
		  return value_type(md.m_ptr, DerivIndexT, md.m_mask);
	  }
   };

   template <typename DataT>
   struct AccessorFactory<DataT, true/* IsArrayT */, true /* IsArrayUnboundedT */>
   {
	  typedef typename std::remove_all_extents<DataT>::type ElementType;
	  typedef MaskedUnboundArrayAccessor<ElementType, WidthT> value_type;

	  static OSL_INLINE bool
	  matches(MaskedData &md)
	  {
		  return (md.m_type.arraylen != 0) &&
				  WideTraits<ElementType>::matches(md.m_type);
	  }

	  template<int DerivIndexT>
	  static OSL_INLINE value_type
	  build(MaskedData &md)
	  {
		  DASSERT(matches(md));
		  return value_type(md.m_ptr, DerivIndexT, md.m_type.arraylen, md.m_mask);
	  }
   };
   
   template<typename DataT>
   using Factory = AccessorFactory<DataT, std::is_array<DataT>::value, (std::extent<DataT>::value == 0)>;
public:
   template<typename DataT>
   OSL_INLINE bool 
   is() {
	   return Factory<DataT>::matches(*this);
   }
   
   template<typename DataT>
   OSL_INLINE typename Factory<DataT>::value_type
   masked() 
   { 
	   return Factory<DataT>::template build<0 /*DerivIndexT*/>(*this);
   }

   
   template<typename DataT>
   OSL_INLINE typename Factory<DataT>::value_type
   maskedDx() 
   {
	   DASSERT(has_derivs());
	   return Factory<DataT>::template build<1 /*DerivIndexT*/>(*this);
   }

   template<typename DataT>
   OSL_INLINE typename Factory<DataT>::value_type
   maskedDy() 
   { 
	   DASSERT(has_derivs());
	   return Factory<DataT>::template build<2 /*DerivIndexT*/>(*this);
   }
   
   template<typename DataT>
   OSL_INLINE typename Factory<DataT>::value_type
   maskedDz() 
   { 
	   DASSERT(has_derivs());
	   return Factory<DataT>::template build<3 /*DerivIndexT*/>(*this);
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
	explicit OSL_INLINE
	RefArrayProxy(DataT (&ref_array_data)[ArrayLenT])
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



template <typename DataT>
struct RefUnboundedArrayProxy
{
	explicit OSL_INLINE
	RefUnboundedArrayProxy(DataT *array_data, int array_length)
	: m_array_data(array_data)
	, m_array_length(array_length)
	{}

	// Must provide user defined copy constructor to 
	// get compiler to be able to follow individual 
	// data members through back to original object
	// when fully inlined the proxy should disappear
	OSL_INLINE
	RefUnboundedArrayProxy(const RefUnboundedArrayProxy &other)
	: m_array_data(other.m_array_data)
	, m_array_length(other.m_array_length)
	{}
	
	OSL_INLINE int 
	length() const { return m_array_length; }
	
	OSL_INLINE DataT & 
	operator[](int array_index)  
	{
		DASSERT(array_index >= 0 && array_index < m_array_length);
		return m_array_data[array_index];
	}

	OSL_INLINE DataT const & 
	operator[](int array_index) const  
	{
		DASSERT(array_index >= 0 && array_index < m_array_length);
		return m_array_data[array_index];
	}
	
private:
	DataT * m_array_data;
	int m_array_length;
};



class DataRef
{
    void *m_ptr;
    TypeDesc m_type;
    bool m_has_derivs; 
public:
	DataRef() = delete;

	explicit OSL_INLINE
	DataRef(TypeDesc type, bool has_derivs, void *ptr)
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
	OSL_INLINE TypeDesc type() const { return m_type; }
	OSL_INLINE bool has_derivs() const { return m_has_derivs; }
	OSL_INLINE bool valid() const { return m_ptr != nullptr; }

protected:
   
   // C++11 doesn't support auto return types, so we must use
   // template specialization to determine the return type of the
   // Ref accessor.  Once C++14 is baseline requirement, we could
   // simplify this a little bit
	template <typename DataT, bool IsArrayT, bool IsArrayUnboundedT>
	struct RefFactory; // undefined
   
	template <typename DataT>
	struct RefFactory<DataT, false, true>
	{
		typedef RefProxy<DataT> value_type;

		static OSL_INLINE bool
		matches(DataRef &dr)
		{
		  return ((dr.m_type.arraylen == 0) &&
				   WideTraits<DataT>::matches(dr.m_type));
		}

		template<int DerivIndexT>
		static OSL_INLINE value_type
		build(DataRef &dr)
		{
		  DASSERT(matches(dr));
		  return value_type(reinterpret_cast<DataT *>(dr.m_ptr)[DerivIndexT]);
		}
	};

	template <typename DataT>
	struct RefFactory<DataT, true /* IsArrayT */, false /* IsArrayUnboundedT */>
	{
		typedef typename std::remove_all_extents<DataT>::type ElementType;
		typedef RefArrayProxy<ElementType, std::extent<DataT>::value> value_type;


		static OSL_INLINE bool
		matches(DataRef &dr)
		{
			return (dr.m_type.arraylen == std::extent<DataT>::value) &&
				   WideTraits<ElementType>::matches(dr.m_type);
		}

		template<int DerivIndexT>
		static OSL_INLINE value_type
		build(DataRef &dr)
		{
		  DASSERT(matches(dr));
		  return value_type(reinterpret_cast<DataT *>(dr.m_ptr)[DerivIndexT]);
		}
	};

	template <typename DataT>
	struct RefFactory<DataT, true/* IsArrayT */, true /* IsArrayUnboundedT */>
	{
		typedef typename std::remove_all_extents<DataT>::type ElementType;
		typedef RefUnboundedArrayProxy<ElementType> value_type;

		static OSL_INLINE bool
		matches(DataRef &dr)
		{
			return (dr.m_type.arraylen != 0)
						  && WideTraits<ElementType>::matches(dr.m_type);
		}

		template<int DerivIndexT>
		static OSL_INLINE value_type
		build(DataRef &dr)
		{
		  DASSERT(matches(dr));
		  return value_type(&(reinterpret_cast<ElementType *>(dr.m_ptr)[DerivIndexT*dr.m_type.arraylen]), dr.m_type.arraylen);
		}
	};


	template<typename DataT>
	using Factory = RefFactory<DataT, std::is_array<DataT>::value, (std::extent<DataT>::value == 0)>;

public:
	template<typename DataT>
	OSL_INLINE bool
	is() {
		return Factory<DataT>::matches(*this);
	}

	template<typename DataT>
	OSL_INLINE typename Factory<DataT>::value_type
	ref()
	{
		return Factory<DataT>::template build<0 /*DerivIndexT*/>(*this);
	}


	template<typename DataT>
	OSL_INLINE typename Factory<DataT>::value_type
	refDx()
	{
		DASSERT(has_derivs());
		return Factory<DataT>::template build<1 /*DerivIndexT*/>(*this);
	}

	template<typename DataT>
	OSL_INLINE typename Factory<DataT>::value_type
	refDy()
	{
		DASSERT(has_derivs());
		return Factory<DataT>::template build<2 /*DerivIndexT*/>(*this);
	}

	template<typename DataT>
	OSL_INLINE typename Factory<DataT>::value_type
	refDz()
	{
		DASSERT(has_derivs());
		return Factory<DataT>::template build<3 /*DerivIndexT*/>(*this);
	}
};

OSL_NAMESPACE_EXIT
