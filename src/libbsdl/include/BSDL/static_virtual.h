// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#pragma once

#include <BSDL/config.h>
#include <tuple>

// This header defines the class StaticVirtual which is used for dynamic dispatch
// of subclass methods using a switch/case statement instead of virtual methods.
// It is inspired on PBRT4 and the tagged pointers, except we don't use tagged
// pointers, instead the subclass ID is stored by StaticVirtual as a member. When
// defineing a base class the list of possible subclasses is explicitly stated like
//
//   struct Base : public StaticVirtual<DerivedOne, DerivedTwo> { ... }
//
// Then methods can be dispatched from Base using the dispatch() method. See an
// example at the end of this file. StaticVirtual has a templated constructor, this
// way at construction time we feed it the specific subclass just passing the this
// pointer (see example).

BSDL_ENTER_NAMESPACE

// First tool is to get the index of a type within a type list. We use std::tuple
// as a type list, and with this recursive template we can say
//
//   Index<B, std::tuple<A, B, C>::value == 1
//
// to map a type to an index
//
template<typename T, typename Tuple> struct Index;

template<typename T, typename... Types>
struct Index<T, std::tuple<T, Types...>> {
    static constexpr std::size_t value = 0;
};

template<typename T, typename U, typename... Types>
struct Index<T, std::tuple<U, Types...>> {
    static constexpr std::size_t value
        = 1 + Index<T, std::tuple<Types...>>::value;
};

// Then we define the Dispatcher struct, which takes an object of type F with a
// templated call() method and calls this method with the class corresponding to
// the requested index.

template<typename F, typename Tuple> struct Dispatcher;

// Trivial case, just one subclass A0, disptch is a no brainer
template<typename F, typename A0> struct Dispatcher<F, std::tuple<A0>> {
    template<size_t BASE> BSDL_INLINE_METHOD static auto call(size_t index, F f)
    {
        return f.template call<A0>();
    }
};

// These macros help us emitting the switch/case statement to save some typing

#define CASE1(offset) \
    case offset:      \
        return f      \
            .template call<typename std::tuple_element<offset, pack>::type>();
#define CASE2(offset)  CASE1(offset) CASE1(offset + 1)
#define CASE4(offset)  CASE2(offset) CASE2(offset + 2)
#define CASE8(offset)  CASE4(offset) CASE4(offset + 4)
#define CASE16(offset) CASE8(offset) CASE8(offset + 8)
#define CASEDEF(offset) \
    default:            \
        return f        \
            .template call<typename std::tuple_element<offset, pack>::type>();

// Now we define ARGSXX(D) macros to emit argument lists with a decorator D to
// optionally prefix the arg name. Arguments are called A0, A1, ... AN. This would be
// shorter if we had recursive macros.

#define ARGS2(D)  D(A0), D(A1)
#define ARGS3(D)  ARGS2(D), D(A2)
#define ARGS4(D)  ARGS3(D), D(A3)
#define ARGS5(D)  ARGS4(D), D(A4)
#define ARGS6(D)  ARGS5(D), D(A5)
#define ARGS7(D)  ARGS6(D), D(A6)
#define ARGS8(D)  ARGS7(D), D(A7)
#define ARGS9(D)  ARGS8(D), D(A8)
#define ARGS10(D) ARGS9(D), D(A9)
#define ARGS11(D) ARGS10(D), D(A10)
#define ARGS12(D) ARGS11(D), D(A11)
#define ARGS13(D) ARGS12(D), D(A12)
#define ARGS14(D) ARGS13(D), D(A13)
#define ARGS15(D) ARGS14(D), D(A14)
#define ARGS16(D) ARGS15(D), D(A15)
#define ARGS17(D) ARGS16(D), D(A16)
#define ARGS18(D) ARGS17(D), D(A17)
#define ARGS19(D) ARGS18(D), D(A18)
#define ARGS20(D) ARGS19(D), D(A19)
#define ARGS21(D) ARGS20(D), D(A20)
#define ARGS22(D) ARGS21(D), D(A21)
#define ARGS23(D) ARGS22(D), D(A22)
#define ARGS24(D) ARGS23(D), D(A23)
#define ARGS25(D) ARGS24(D), D(A24)
#define ARGS26(D) ARGS25(D), D(A25)
#define ARGS27(D) ARGS26(D), D(A26)
#define ARGS28(D) ARGS27(D), D(A27)
#define ARGS29(D) ARGS28(D), D(A28)
#define ARGS30(D) ARGS29(D), D(A29)
#define ARGS31(D) ARGS30(D), D(A30)
#define ARGS32(D) ARGS31(D), D(A31)

// We also need helpers to emit the case statements.

#define CASES2  CASE1(0) CASEDEF(1)
#define CASES3  CASE2(0) CASEDEF(2)
#define CASES4  CASE2(0) CASE1(2) CASEDEF(3)
#define CASES5  CASE4(0) CASEDEF(4)
#define CASES6  CASE4(0) CASE1(4) CASEDEF(5)
#define CASES7  CASE4(0) CASE2(4) CASEDEF(6)
#define CASES8  CASE4(0) CASE2(4) CASE1(6) CASEDEF(7)
#define CASES9  CASE8(0) CASEDEF(8)
#define CASES10 CASE8(0) CASE1(8) CASEDEF(9)
#define CASES11 CASE8(0) CASE2(8) CASEDEF(10)
#define CASES12 CASE8(0) CASE2(8) CASE1(10) CASEDEF(11)
#define CASES13 CASE8(0) CASE4(8) CASEDEF(12)
#define CASES14 CASE8(0) CASE4(8) CASE1(12) CASEDEF(13)
#define CASES15 CASE8(0) CASE4(8) CASE2(12) CASEDEF(14)
#define CASES16 CASE8(0) CASE4(8) CASE2(12) CASE1(14) CASEDEF(15)
#define CASES17 CASE16(0) CASEDEF(16)
#define CASES18 CASE16(0) CASE1(16) CASEDEF(17)
#define CASES19 CASE16(0) CASE2(16) CASEDEF(18)
#define CASES20 CASE16(0) CASE2(16) CASE1(18) CASEDEF(19)
#define CASES21 CASE16(0) CASE4(16) CASEDEF(20)
#define CASES22 CASE16(0) CASE4(16) CASE1(20) CASEDEF(21)
#define CASES23 CASE16(0) CASE4(16) CASE2(20) CASEDEF(22)
#define CASES24 CASE16(0) CASE4(16) CASE2(20) CASE1(22) CASEDEF(23)
#define CASES25 CASE16(0) CASE8(16) CASEDEF(24)
#define CASES26 CASE16(0) CASE8(16) CASE1(24) CASEDEF(25)
#define CASES27 CASE16(0) CASE8(16) CASE2(24) CASEDEF(26)
#define CASES28 CASE16(0) CASE8(16) CASE2(24) CASE1(26) CASEDEF(27)
#define CASES29 CASE16(0) CASE8(16) CASE4(24) CASEDEF(28)
#define CASES30 CASE16(0) CASE8(16) CASE4(24) CASE1(28) CASEDEF(29)
#define CASES31 CASE16(0) CASE8(16) CASE4(24) CASE2(28) CASEDEF(30)
#define CASES32 CASE16(0) CASE8(16) CASE4(24) CASE2(28) CASE1(30) CASEDEF(31)

// And we are ready to create dispatchers up to 32 entries. We use two decorators:
// one that adds the typename keyword and another one that does nothing.

#define TYPENAME(X) typename X
#define ARG(X)      X

#define DECL_DISPATCH(_ARGS, _CASES)                           \
    template<typename F, _ARGS(TYPENAME)>                      \
    struct Dispatcher<F, std::tuple<_ARGS(ARG)>> {             \
        using pack = std::tuple<_ARGS(ARG)>;                   \
        template<size_t BASE>                                  \
        static BSDL_INLINE_METHOD auto call(size_t index, F f) \
        {                                                      \
            switch (index - BASE) {                            \
                _CASES                                         \
            }                                                  \
        }                                                      \
    }

DECL_DISPATCH(ARGS2, CASES2);
DECL_DISPATCH(ARGS3, CASES3);
DECL_DISPATCH(ARGS4, CASES4);
DECL_DISPATCH(ARGS5, CASES5);
DECL_DISPATCH(ARGS6, CASES6);
DECL_DISPATCH(ARGS7, CASES7);
DECL_DISPATCH(ARGS8, CASES8);
DECL_DISPATCH(ARGS9, CASES9);
DECL_DISPATCH(ARGS10, CASES10);
DECL_DISPATCH(ARGS11, CASES11);
DECL_DISPATCH(ARGS12, CASES12);
DECL_DISPATCH(ARGS13, CASES13);
DECL_DISPATCH(ARGS14, CASES14);
DECL_DISPATCH(ARGS15, CASES15);
DECL_DISPATCH(ARGS16, CASES16);
DECL_DISPATCH(ARGS17, CASES17);
DECL_DISPATCH(ARGS18, CASES18);
DECL_DISPATCH(ARGS19, CASES19);
DECL_DISPATCH(ARGS20, CASES20);
DECL_DISPATCH(ARGS21, CASES21);
DECL_DISPATCH(ARGS22, CASES22);
DECL_DISPATCH(ARGS23, CASES23);
DECL_DISPATCH(ARGS24, CASES24);
DECL_DISPATCH(ARGS25, CASES25);
DECL_DISPATCH(ARGS26, CASES26);
DECL_DISPATCH(ARGS27, CASES27);
DECL_DISPATCH(ARGS28, CASES28);
DECL_DISPATCH(ARGS29, CASES29);
DECL_DISPATCH(ARGS30, CASES30);
DECL_DISPATCH(ARGS31, CASES31);
DECL_DISPATCH(ARGS32, CASES32);

#define MAX_SWITCH 32

// And then if we have more than 32 types we divide them in smaller switches with
// if statements
template<typename F, ARGS32(TYPENAME), typename... Types>
struct Dispatcher<F, std::tuple<ARGS32(ARG), Types...>> {
    template<size_t BASE> static BSDL_INLINE_METHOD auto call(size_t index, F f)
    {
        if (index < BASE + MAX_SWITCH)
            return Dispatcher<F, std::tuple<ARGS32(ARG)>>::template call<BASE>(
                index, f);
        else
            return Dispatcher<F, std::tuple<Types...>>::template call<
                BASE + MAX_SWITCH>(index, f);
    }
};

template<typename T, typename... Types> struct Includes;
template<typename T> struct Includes<T> {
    static constexpr bool value = false;
};
template<typename T, typename O, typename... Types>
struct Includes<T, O, Types...> {
    static constexpr bool value = std::is_same<T, O>::value
                                  || Includes<T, Types...>::value;
};

// Finally, yes, we are ready to define our StaticVirtual class
template<typename... Types> struct StaticVirtual {
    // The subclasses type list
    using types = std::tuple<Types...>;
    // A templated constructor to store the type index in idx, we pass a pointer
    // to infer the type because the compiler has problems with templated constructors.
    template<typename T>
    BSDL_INLINE_METHOD StaticVirtual(const T*) : idx(GET_ID<T>())
    {
    }

    // Simple handler for the Dispatch helper that takes a callable F, possible
    // lambda function that receives obj cast to the particular type.
    template<typename F> struct Handler {
        template<typename T> BSDL_INLINE_METHOD auto call()
        {
            return f(*static_cast<T*>(obj));
        }
        F f;
        StaticVirtual<Types...>* obj;
    };
    // const version
    template<typename F> struct ConstHandler {
        template<typename T> BSDL_INLINE_METHOD auto call()
        {
            return f(*static_cast<const T*>(obj));
        }
        F f;
        const StaticVirtual<Types...>* obj;
    };

    // Actual dispatch function that you can feed a lambda
    template<typename F> BSDL_INLINE_METHOD auto dispatch(F f)
    {
        Handler<F> h = { f, this };
        return Dispatcher<Handler<F>, types>::template call<0>(idx, h);
    }
    // Const version
    template<typename F> BSDL_INLINE_METHOD auto dispatch(F f) const
    {
        ConstHandler<F> h = { f, this };
        return Dispatcher<ConstHandler<F>, types>::template call<0>(idx, h);
    }
    // For inspecting a type alone (no object), f should only use its arg type
    template<typename F>
    static BSDL_INLINE_METHOD auto dispatch(unsigned idx, F f)
    {
        StaticVirtual tmp(idx);
        return tmp.dispatch(f);
    }

    // Static mapping from type to index
    template<typename T> static BSDL_INLINE_METHOD constexpr unsigned GET_ID()
    {
        constexpr bool included = Includes<T, Types...>::value;
        static_assert(included,
                      "Constructed type is not in StaticVirtual declaration");
        if constexpr (included)
            return Index<T, types>::value;
        else
            return 0;
    }
    // Dynamic version
    unsigned BSDL_INLINE_METHOD get_id() const { return idx; }

private:
    // For static dispatch
    BSDL_INLINE_METHOD StaticVirtual(unsigned idx) : idx(idx) {}

    uint16_t idx;
};

// Tuples as type packs tools, this one gives you the maximum size
// and align for a collection of types should you put them in a union
template<typename Tuple> struct SizeAlign;

template<> struct SizeAlign<std::tuple<>> {
    static constexpr std::size_t size  = 0;
    static constexpr std::size_t align = 0;
};

template<typename T, typename... Types>
struct SizeAlign<std::tuple<T, Types...>> {
    using Other                        = SizeAlign<std::tuple<Types...>>;
    static constexpr std::size_t size  = std::max(sizeof(T), Other::size);
    static constexpr std::size_t align = std::max(alignof(T), Other::align);
};

// And this is a very useful tool to map an std::tuple of types to another with a
// template Filter that maps Filter<OriginalType>::ToSomeOtherType. The Filter
// template also needs to define a constexpr bool "keep", which is used to remove
// types from the tuple if set to false.

template<template<typename> typename Filter,
         typename Orig,                 // Original tuple
         typename Dest = std::tuple<>>  // Placeholder for the result
struct MapTuple;
// Trivial case, empty tuple
template<template<typename> typename Filter, typename... Ds>
struct MapTuple<Filter, std::tuple<>, std::tuple<Ds...>> {
    using type = std::tuple<Ds...>;  // We are done, result is whatever is in Ds
};
// Recursion: there is a list of original types <O, Os...> and a list of already
// convrted types <Ds...>
template<template<typename> typename Filter, typename O, typename... Os,
         typename... Ds>
struct MapTuple<Filter, std::tuple<O, Os...>, std::tuple<Ds...>> {
    using D = typename Filter<O>::type;  // Convert the head of the list
    using Grown =
        typename std::  // If "keep" append D to Ds, otherwise leave Ds alone
        conditional<Filter<O>::keep, std::tuple<D, Ds...>,
                    std::tuple<Ds...>>::type;
    // Recursive "call" to itself with a shorter list of Os... (without the head O)
    using type = typename MapTuple<Filter, std::tuple<Os...>, Grown>::type;
};

// Just another tool to represent type lists that provides a static method
// ::apply(someobject) to run all the types T through someobject.visit<T>()
template<typename... Ts> struct TypeList;
template<> struct TypeList<> {
    template<typename F> static void apply(F f) {}
};
template<typename T, typename... Ts> struct TypeList<T, Ts...> {
    template<typename F> static void apply(F f)
    {
        f.template visit<T>();
        TypeList<Ts...>::apply(f);
    }
};


BSDL_LEAVE_NAMESPACE

#if 0
// Example usage
#    include "static_virtual.h"
#    include <cstdio>

struct One;
struct Two;
struct Three;

struct Base : public StaticVirtual<One, Two, Three> {
    template <typename T> Base(const T* o): StaticVirtual(o) {}
    int mymethod(float v)const;
};
struct One : public Base {
    One(): Base(this) {}
    int mymethod(float v) const
    {
        printf("I'm one\n");
        return 1;
    }
};
struct Two : public Base {
    Two(): Base(this) {}
    int mymethod(float v)const
    {
        printf("I'm two\n");
        return 2;
    }
};
struct Three : public Base {
    Three(): Base(this) {}
    int mymethod(float v)const
    {
        printf("I'm three\n");
        return 3;
    }
};
// Dispatch definition has to go after subclasses declarations
int Base::mymethod(float v)const
{
    return dispatch([&](auto& obj) { return obj.mymethod(v); });
}

int main()
{

    One   one;
    Two   two;
    Base* obj = &one;
    printf("got %d\n", obj->mymethod(1.0f));
    obj = &two;
    printf("got %d\n", obj->mymethod(1.0f));
    return 0;
}
#endif

// Cleanup macro mess

#undef CASE1
#undef CASE2
#undef CASE4
#undef CASE8
#undef CASE16
#undef CASEDEF
#undef ARGS2
#undef ARGS3
#undef ARGS4
#undef ARGS5
#undef ARGS6
#undef ARGS7
#undef ARGS8
#undef ARGS9
#undef ARGS10
#undef ARGS11
#undef ARGS12
#undef ARGS13
#undef ARGS14
#undef ARGS15
#undef ARGS16
#undef ARGS17
#undef ARGS18
#undef ARGS19
#undef ARGS20
#undef ARGS21
#undef ARGS22
#undef ARGS23
#undef ARGS24
#undef ARGS25
#undef ARGS26
#undef ARGS27
#undef ARGS28
#undef ARGS29
#undef ARGS30
#undef ARGS31
#undef ARGS32
#undef CASES2
#undef CASES3
#undef CASES4
#undef CASES5
#undef CASES6
#undef CASES7
#undef CASES8
#undef CASES9
#undef CASES10
#undef CASES11
#undef CASES12
#undef CASES13
#undef CASES14
#undef CASES15
#undef CASES16
#undef CASES17
#undef CASES18
#undef CASES19
#undef CASES20
#undef CASES21
#undef CASES22
#undef CASES23
#undef CASES24
#undef CASES25
#undef CASES26
#undef CASES27
#undef CASES28
#undef CASES29
#undef CASES30
#undef CASES31
#undef CASES32
#undef TYPENAME
#undef ARG
#undef MAX_SWITCH
