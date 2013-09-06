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

#ifndef OSLCLOSURE_H
#define OSLCLOSURE_H

#include <cstring>
#include <OpenImageIO/ustring.h>
#include "oslconfig.h"

OSL_NAMESPACE_ENTER

// Ids for the builtin provided closures

enum {
    CLOSURE_BSDF_DIFFUSE_ID,
    CLOSURE_BSDF_TRANSLUCENT_ID,
    CLOSURE_BSDF_REFLECTION_ID,
    CLOSURE_BSDF_REFRACTION_ID,
    CLOSURE_BSDF_DIELECTRIC_ID,
    CLOSURE_BSDF_TRANSPARENT_ID,
    CLOSURE_BSDF_MICROFACET_GGX_ID,
    CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
    CLOSURE_BSDF_MICROFACET_BECKMANN_ID,
    CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID,
    CLOSURE_BSDF_WARD_ID,
    CLOSURE_BSDF_PHONG_ID,
    CLOSURE_BSDF_PHONG_RAMP_ID,
    CLOSURE_BSDF_HAIR_DIFFUSE_ID,
    CLOSURE_BSDF_HAIR_SPECULAR_ID,
    CLOSURE_BSDF_ASHIKHMIN_VELVET_ID,
    CLOSURE_BSDF_CLOTH_ID,
    CLOSURE_BSDF_CLOTH_SPECULAR_ID,
    CLOSURE_BSDF_FAKEFUR_DIFFUSE_ID,
    CLOSURE_BSDF_FAKEFUR_SPECULAR_ID,
    CLOSURE_BSDF_FAKEFUR_SKIN_ID,
    CLOSURE_BSDF_WESTIN_BACKSCATTER_ID,
    CLOSURE_BSDF_WESTIN_SHEEN_ID,
    CLOSURE_BSSRDF_CUBIC_ID,
    CLOSURE_EMISSION_ID,
    CLOSURE_DEBUG_ID,
    CLOSURE_BACKGROUND_ID,
    CLOSURE_HOLDOUT_ID,
    CLOSURE_SUBSURFACE_ID,

    NBUILTIN_CLOSURES };


/// Labels for light walks
///
/// This is the leftover of a class which used to hold all the labels
/// Now just acts as a little namespace for the basic definitions
///
/// NOTE: you still can assign these labels as keyword arguments to
///       closures. But we have removed the old labels array in the
///       primitives.
class OSLEXECPUBLIC Labels {
public:

    static const ustring NONE;
    // Event type
    static const ustring CAMERA;
    static const ustring LIGHT;
    static const ustring BACKGROUND;
    static const ustring TRANSMIT;
    static const ustring REFLECT;
    static const ustring VOLUME;
    static const ustring OBJECT;
    // Scattering
    static const ustring DIFFUSE;  // typical 2PI hemisphere
    static const ustring GLOSSY;   // blurry reflections and transmissions
    static const ustring SINGULAR; // perfect mirrors and glass
    static const ustring STRAIGHT; // Special case for transparent shadows

    static const ustring STOP;     // end of a surface description

};

/// Base class representation of a radiance color closure. These are created on
/// the fly during rendering via placement new. Therefore derived classes should
/// only use POD types as members.
class OSLEXECPUBLIC ClosurePrimitive {
public:
    /// The categories of closure primitives we can have.  It's possible
    /// to customize/extend this list as long as there is coordination
    /// between the closure primitives and the integrators.
    enum Category {
        BSDF,           ///< Reflective and/or transmissive surface
        BSSRDF,         ///< Sub-surface light transfer
        Emissive,       ///< Light emission
        Background,     ///< Background emission
        Volume,         ///< Volume scattering
        Holdout,        ///< Holdout from alpha
        Debug,          ///< For debug and masks
    };

    // Describe a closure's sidedness
    enum Sidedness {
        None  = 0,
        Front = 1,
        Back  = 2,
        Both  = 3
    };

    ClosurePrimitive (Category category) :
        m_category (category) { }

    virtual ~ClosurePrimitive () { }

    virtual void setup() {};

    /// Return the category of material this primitive represents.
    ///
    int category () const { return m_category; }

    /// How many bytes of parameter storage will this primitive need?
    ///
    virtual size_t memsize () const = 0;

    /// The name of the closure primitive.  Must be unique so that prims
    /// with non-matching names are definitley not the same kind of
    /// closure primitive.
    virtual const char * name () const = 0;

    /// Stream operator output (for debugging)
    ///
    virtual void print_on (std::ostream &out) const = 0;

    friend std::ostream& operator<< (std::ostream& o, const ClosurePrimitive& b);

    /// Helper function: sample cosine-weighted hemisphere.
    ///
    static void sample_cos_hemisphere (const Vec3 &N, const Vec3 &omega_out,
                        float randu, float randv, Vec3 &omega_in, float &pdf);

    /// Helper function: sample uniform-weighted hemisphere.
    ///
    static void sample_uniform_hemisphere (const Vec3 &N, const Vec3 &omega_out,
                        float randu, float randv, Vec3 &omega_in, float &pdf);

    /// Helper function: make two unit vectors that are orthogonal to N and
    /// each other.  This assumes that N is already normalized.  We get the
    /// first orthonormal by taking the cross product of N and (1,1,1), unless N
    /// is 1,1,1, in which case we cross with (-1,1,1).  Either way, we get
    /// something orthogonal.  Then N x a is mutually orthogonal to the other two.
    static void make_orthonormals (const Vec3 &N, Vec3 &a, Vec3 &b);

    /// Helper function: make two unit vectors that are orthogonal to N and
    /// each other. The x vector will point roughly in the same direction as the
    /// tangent vector T. We assume that T and N are already normalized.
    static void make_orthonormals (const Vec3 &N, const Vec3& T, Vec3 &x, Vec3& y);

    /// Helper function to compute fresnel reflectance R of a dieletric. The
    /// transmission can be computed as 1-R. This routine accounts for total
    /// internal reflection. eta is the ratio of the indices of refraction
    /// (inside medium over outside medium - for example ~1.333 for water from
    /// air). The inside medium is defined as the region of space the normal N
    /// is pointing away from.
    /// This routine also computes the refracted direction T from the incoming
    /// direction I (which should be pointing away from the surface). The normal
    /// should always point in its canonical direction so that this routine can
    /// flip the refraction coefficient as needed.
    static float fresnel_dielectric (float eta, const Vec3 &N,
            const Vec3 &I, const Vec3 &dIdx, const Vec3 &dIdy,
            Vec3 &R, Vec3 &dRdx, Vec3 &dRdy,
            Vec3 &T, Vec3 &dTdx, Vec3 &dTdy,
            bool &is_inside);

    /// Helper function to compute fresnel reflectance R of a dielectric. This
    /// formulation does not explicitly compute the refracted vector so should
    /// only be used for reflective materials. cosi is the angle between the
    /// incomming ray and the surface normal, eta gives the index of refraction
    /// of the surface.
    static float fresnel_dielectric (float cosi, float eta);

    /// Helper function to compute fresnel reflectance R of a conductor. These
    /// materials do not transmit any light. cosi is the angle between the
    /// incomming ray and the surface normal, eta and k give the complex index
    /// of refraction of the surface.
    static float fresnel_conductor (float cosi, float eta, float k);

    /// Are 'this' and 'other' identical closure primitives and thus can
    /// be merged simply by summing their weights?  We expect every subclass
    /// to overload this (and call back to the parent as well) -- if they
    /// don't, component merges will happen inappropriately!
    virtual bool mergeable (const ClosurePrimitive *other) const {
        return true;
    }

private:
    Category m_category;
};



/// Subclass of ClosurePrimitive that contains the methods needed
/// for a BSDF-like material.
class BSDFClosure : public ClosurePrimitive {
public:
    BSDFClosure (ustring scattering, Sidedness eval_sidedness = Front) :
        ClosurePrimitive (BSDF),
        m_eval_sidedness (eval_sidedness),
        m_scattering_label (scattering) { }
    ~BSDFClosure () { }


    /// Assuming the side of the viewer is always Front, return which side
    /// it is sensitive to light on.
    ///
    Sidedness get_light_side() const {
        return m_eval_sidedness;
    }
    /// Return the scattering label for this primitive
    ///
    ustring scattering () const { return m_scattering_label; }

    /// Albedo function, returns the integral (or an approximation) of
    /// the reflected light in the given out direction. It is expected to
    /// be less than or equal to 1.0 and it is not guaranteed to be accurate.
    /// It is meant to be used for sampling decisions. When two or more
    /// closures are present at the same time, their albedos will be used
    /// to compute a probability of one being chosen. So this value must reflect
    /// the "importance" of the closure when compared to others. And as a
    /// convention we use the integral of the radiance to omega_out for all
    /// the possible omega_in of the eval function. An approximation is enough,
    /// the accuracy of this number only affects the quality of the sampling.
    ///
    /// This value will by no means affect the apperance of the render in any
    /// other way than sampling noise. Unless 0.0 is returned, which excludes
    /// the bsdf from indirect lighting. And we use this value for those bsdf's
    /// that we don't know how to sample.
    ///
    /// Most bsdf's are designed to integrate to 1.0, except the fresnel affected
    /// ones. And returning 1.0 is also a safe value for when eval is too complicated
    /// to integrate.
    virtual float albedo (const Vec3 &omega_out) const = 0;

    /// Evaluate the extended BRDF and BTDF kernels -- Given viewing direction
    /// omega_out and lighting direction omega_in (both pointing away from the
    /// surface), compute the amount of radiance to be transfered between these
    /// two directions. This also computes the probability of sampling the
    /// direction omega_in from the sample method.
    virtual Color3 eval_reflect  (const Vec3 &omega_out, const Vec3 &omega_in, float &pdf) const = 0;
    virtual Color3 eval_transmit (const Vec3 &omega_out, const Vec3 &omega_in, float &pdf) const = 0;

    /// Sample the BSDF -- Given instance parameters, viewing direction omega_out
    /// (pointing away from the surface), and random deviates randu and
    /// randv on [0,1), return a sampled direction omega_in, the PDF value
    /// in that direction and the evaluation of the color.
    /// Unlike the other methods, this routine can be called even if the
    /// if the scattering routine returned SINGULAR. This is to allow singular
    /// BRDFs to pick directions from infinitely small cones.
    /// The caller is responsible for initializing the values of the output
    /// arguments with zeros so that early exits from this function are
    /// simplified. Returns the direction label (R or T).
    virtual ustring sample (const Vec3 &Ng,
                         const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                         float randu, float randv,
                         Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                         float &pdf, Color3 &eval) const = 0;

protected:

private:
    Sidedness m_eval_sidedness; // which canonical sides are sensitive to light?
    // A bsdf can only perform one type of scattering
    ustring  m_scattering_label;
};

/// Subclass of ClosurePrimitive that contains the methods needed
/// for a BSSSRDF-like material.
class BSSRDFClosure : public ClosurePrimitive {
public:
    BSSRDFClosure() : ClosurePrimitive(BSSRDF) { }
    ~BSSRDFClosure() { }

    /// Evaluate the amount of light transfered between two points seperated by
    /// a distance r.
    virtual Color3 eval(float r) const = 0;

    /// Return the maximum distance for which eval returns a non-zero value.
    virtual float max_radius() const = 0;
};


class VolumeClosure : public ClosurePrimitive {
public:
    VolumeClosure() : ClosurePrimitive(Volume), m_ior(1.0f), m_sigma_s(0.0f), m_sigma_a(0.0f) {}

    // FIXME: should ior be stored here?
    float ior() const { return m_ior; }
    void ior(float i) { m_ior = i; }

    // Scattering properties
    Color3 sigma_s() const { return m_sigma_s; }
    Color3 sigma_a() const { return m_sigma_a; }
    void sigma_s(const Color3 &s) { m_sigma_s = s; }
    void sigma_a(const Color3 &a) { m_sigma_s = a; }

    // phase function
    virtual Color3 eval_phase(const Vec3 &omega_in, const Vec3 &omega_out) const = 0;

    bool mergeable (const ClosurePrimitive *other) const {
        const VolumeClosure *comp = (const VolumeClosure *) other;
        return m_ior == comp->m_ior &&
               m_sigma_s == comp->m_sigma_s &&
               m_sigma_a == comp->m_sigma_a &&
               ClosurePrimitive::mergeable(other);
    }

private:
    float m_ior;
    Color3 m_sigma_s;   ///< Scattering coefficient
    Color3 m_sigma_a;   ///< Absorption coefficient
};


/// Subclass of ClosurePrimitive that contains the methods needed
/// for an emissive material.
class EmissiveClosure : public ClosurePrimitive {
public:
    EmissiveClosure () : ClosurePrimitive (Emissive) { }
    ~EmissiveClosure () { }

    /// Evaluate the emission -- Given instance parameters, the light's surface
    /// normal N and the viewing direction omega_out, compute the outgoing
    /// radiance along omega_out (which points away from the light's
    /// surface).
    virtual Color3 eval (const Vec3 &Ng, const Vec3 &omega_out) const = 0;

    /// Sample the emission direction -- Given instance parameters, the light's
    /// surface normal and random deviates randu and randv on [0,1), return a
    /// sampled direction omega_out (pointing away from the light's surface) and
    /// the PDF value in that direction.
    virtual void sample (const Vec3 &Ng,
                         float randu, float randv,
                         Vec3 &omega_out, float &pdf) const = 0;

    /// Return the probability distribution function in the direction omega_out,
    /// given the parameters and the light's surface normal.  This MUST match
    /// the PDF computed by sample().
    virtual float pdf (const Vec3 &Ng,
                       const Vec3 &omega_out) const = 0;
};



/// Subclass of ClosurePrimitive that contains the serves to
/// flag a background color. No methods needed yet, only the
/// weight is going to be used
class BackgroundClosure : public ClosurePrimitive {
public:
    BackgroundClosure () : ClosurePrimitive (Background) { }
    ~BackgroundClosure () { }

};

namespace pvt {
class ShadingSystemImpl;
}



/// ClosureColor is the base class for a lightweight tree representation
/// of OSL closures for the sake of the executing OSL shader.
///
/// Remember that a closure color really just boils down to a flat list
/// of weighted closure primitive components (such as diffuse,
/// transparent, etc.).  But it's expensive to construct these
/// dynamically as we execute the shader, so instead of maintaining a
/// properly flattened list as we go, we just manipulate a very
/// lightweight data structure that looks like a tree, where leaf nodes
/// are a single primitive component, and internal nodes of the tree are
/// are either 'add' (two closures) or 'mul' (of a closure with a
/// weight) and just reference their operands by pointers.
/// 
/// We are extremely careful to make these classes resemble POD (plain
/// old data) so they can be easily "placed" anywhere, including a memory
/// pool.  So no virtual functions!
///
/// The base class ClosureColor just provides the type, and it's
/// definitely one of the three kinds of subclasses: ClosureComponent,
/// ClosureMul, ClosureAdd.
struct ClosureColor {
    enum ClosureType { COMPONENT, MUL, ADD };

    ClosureType type;
};



/// ClosureComponent is a subclass of ClosureColor that holds the ID and
/// parameter data for a single primitive closure component (such as
/// diffuse, translucent, etc.).  The declaration leaves 4 bytes for
/// parameter data (mem), but it's expected that the structure be
/// allocated with enough space to house all the parameter data for
/// whatever type of custom primitive component it actually is.
struct ClosureComponent : public ClosureColor
{
    struct Attr
    {
        ustring   key;
        union {
            int     integer;
            float   flt;
            float   triple[3];  // This will fake Color3 and Vec3 which C++ doesn't allow in unions
            void   *str;        // And this fakes a ustring (not allowed in unions either)
        }         value;

        // This members are just to avoid having to typecast all the time
        int           & integer()       { return value.integer; }
        const int     & integer() const { return value.integer; }
        float         & flt()           { return value.flt; }
        const float   & flt()     const { return value.flt; }
        Color3        & color()         { return *(Color3*)       raw_data(); }
        const Color3  & color()   const { return *(const Color3*) raw_data(); }
        Vec3          & vector()        { return *(Vec3*)         raw_data(); }
        const Vec3    & vector()  const { return *(const Vec3*)   raw_data(); }
        ustring       & str()           { return *(ustring*)      raw_data(); }
        const ustring & str()     const { return *(const ustring*)raw_data(); }

    private:
        char*       raw_data()       { return reinterpret_cast<char*>(&value); }
        const char* raw_data() const { return reinterpret_cast<const char*>(&value); }
    };

    int    id;       ///< Id of the component
    int    size;     ///< Memory used by the primitive
    int    nattrs;   ///< Number of keyword attributes
    Vec3   w;        ///< Weight of this component
    char   mem[4];   ///< Memory for the primitive
                     ///  4 is the minimum, allocation
                     ///  will be scaled to requirements
                     ///  of the primitive

    /// Handy method for getting the parameter memory as a void*.
    ///
    void *data () { return &mem; }
    const void *data () const { return &mem; }
    /// Attributes are always allocated at the end of the data block
    Attr *attrs() { return (Attr *)((char *)data() + size); }
    const Attr *attrs() const { return (Attr *)((char *)data() + size); }
};


/// ClosureMul is a subclass of ClosureColor that provides a lightweight
/// way to represent a closure multiplied by a scalar or color weight.
struct ClosureMul : public ClosureColor
{
    Color3 weight;
    const ClosureColor *closure;
};


/// ClosureAdd is a subclass of ClosureColor that provides a lightweight
/// way to represent a closure that is a sum of two other closures.
struct ClosureAdd : public ClosureColor
{
    const ClosureColor *closureA;
    const ClosureColor *closureB;
};


namespace pvt {
class ShadingSystemImpl;
}

//std::ostream &operator<< (std::ostream &out, const ClosureColor &closure);
void print_closure (std::ostream &out, const ClosureColor *closure, pvt::ShadingSystemImpl *ss);

OSL_NAMESPACE_EXIT

#endif /* OSLCLOSURE_H */
