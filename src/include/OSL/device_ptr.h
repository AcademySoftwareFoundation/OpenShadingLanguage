// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <OSL/oslconfig.h>



OSL_NAMESPACE_ENTER


/// Wrapper class for holding a "device" pointer -- GPU or whatnot. It
/// provides protections so that the pointer cannot easily be accessed on the
/// host side, where presumably it would not be valid memory.
template<class T> class device_ptr {
public:
    device_ptr() = default;

    // Copy ctr from another device_ptr of the same type
    device_ptr(const device_ptr& other) = default;

    /// On device, device_ptr can construct from a pointer.
    /// On host, it must be explicitly constructed -- use with caution.
#ifdef __CUDA_ARCH__
    device_ptr(T* ptr) : m_ptr(ptr) {}
#else
    explicit device_ptr(T* ptr) : m_ptr(ptr) {}
#endif

#ifdef __CUDA_ARCH__
    /// On device, act as a pointer. None of these things are allowed on the
    /// host.
    T* operator->() const { return m_ptr; }
    T& operator*() const { return *m_ptr; }
#endif

    /// Extract the raw device-side pointer. Use with caution! On the host,
    /// this will not point to valid memory.
    T* d_get() const { return m_ptr; }

    /// Evaluate as bool is a null pointer check.
    operator bool() const noexcept { return m_ptr != nullptr; }

    /// Reset the pointer to `dptr`, which must be a device-side raw pointer,
    /// or null. Since this device_ptr is non-owning, any previous value is
    /// simply overwritten.
    void reset(T* dptr = nullptr) { m_ptr = dptr; }

private:
    T* m_ptr = nullptr;  // underlying pointer, initializes to null
};



OSL_NAMESPACE_EXIT
