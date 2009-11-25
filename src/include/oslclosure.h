/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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

#include <OpenImageIO/ustring.h>

#include "oslconfig.h"

#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {

/// Labels for light walks
///
/// This is the leftover of a class which used to hold all the labels
/// Now just acts as a little namespace for the basic definitions
///
class Labels {
public:

    static const ustring NONE;
    // Event type
    static const ustring CAMERA;
    static const ustring LIGHT;
    static const ustring BACKGROUND;
    static const ustring SURFACE;
    static const ustring VOLUME;
    // Direction
    static const ustring TRANSMIT;
    static const ustring REFLECT;
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
class ClosurePrimitive {
public:
    /// The categories of closure primitives we can have.  It's possible
    /// to customize/extend this list as long as there is coordination
    /// between the closure primitives and the integrators.
    enum Category {
        BSDF,           ///< It's reflective and/or transmissive
        Emissive,       ///< It's emissive (like a light)
        Background,     ///< It's the background
    };

    // Describe a closure's sidedness
    enum Sidedness {
        None  = 0,
        Front = 1,
        Back  = 2,
        Both  = 3
    };

    // Maximum allowed custom labels
    static const int MAXCUSTOM = 5;

    ClosurePrimitive (Category category) :
        m_category (category) { m_custom_labels[0] = Labels::NONE; }

    virtual ~ClosurePrimitive () { }

    /// Meant to be used by the opcode to set custom labels
    ///
    void set_custom_label (int i, ustring label) { m_custom_labels[i] = label; }

    /// Return the category of material this primitive represents.
    ///
    int category () const { return m_category; }
    /// Get the custom labels (Labels::NONE terminated)
    ///
    const ustring *get_custom_labels()const { return m_custom_labels; }

    /// Stream operator output (for debugging)
    ///
    virtual void print_on (std::ostream &out) const = 0;

    friend std::ostream& operator<< (std::ostream& o, const ClosurePrimitive& b);

    /// Helper function: sample cosine-weighted hemisphere.
    ///
    static void sample_cos_hemisphere (const Vec3 &N, const Vec3 &omega_out,
                        float randu, float randv, Vec3 &omega_in, float &pdf);

    /// Helper function: return the PDF for cosine-weighted hemisphere.
    ///
    static float pdf_cos_hemisphere (const Vec3 &N, const Vec3 &omega_in);

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
            Vec3& T, Vec3 &dTdx, Vec3 &dTdy);

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

private:
    Category m_category;
    // Labels::NONE terminated custom label list
    ustring  m_custom_labels[MAXCUSTOM + 1];
};



/// Subclass of ClosurePrimitive that contains the methods needed
/// for a BSDF-like material: eval(), sample(), pdf().
class BSDFClosure : public ClosurePrimitive {
public:
    BSDFClosure (Sidedness side, ustring scattering, bool needs_eval = true, bool reflective_eval = true) :
        ClosurePrimitive (BSDF),
        m_sidedness(side),
        m_needs_eval(needs_eval),
        m_reflective_eval (reflective_eval),
        m_scattering_label(scattering) { }
    ~BSDFClosure () { }


    /// Given the side from which we are viewing this closure, return which side
    /// it is sensitive to light on.
    Sidedness get_light_side(Sidedness viewing_side) const {
        if (!m_needs_eval)
            return None;
        return m_reflective_eval ?
                Sidedness (m_sidedness & viewing_side) :
                Sidedness (m_sidedness ^ viewing_side);
    }
    /// Return the scattering label for this primitive
    ///
    ustring scattering () const { return m_scattering_label; }

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
    /// get_cone routine returned false. This is to allow singular BRDFs to pick
    /// directions from infinitely small cones.
    /// The caller is responsible for initializing the values of the output
    /// arguments with zeros so that early exits from this function are
    /// simplified. Returns the direction label (R or T).
    virtual ustring sample (const Vec3 &Ng,
                         const Vec3 &omega_out, const Vec3 &domega_out_dx, const Vec3 &domega_out_dy,
                         float randu, float randv,
                         Vec3 &omega_in, Vec3 &domega_in_dx, Vec3 &domega_in_dy,
                         float &pdf, Color3 &eval) const = 0;

protected:
    /// Helper function to perform a faceforward on the geometric and shading
    /// normals according to what is allowed by the sidedness flag
    bool faceforward (const Vec3 &omega_out,
                      const Vec3 &Ng, const Vec3 &N,
                      Vec3 &Ngf, Vec3 &Nf) const {
        // figure out sidedness
        float cosNgO = Ng.dot(omega_out);
        if (cosNgO > 0 && (m_sidedness & Front)) {
            Nf = N;
            Ngf = Ng;
            return true;
        } else if (cosNgO < 0 && (m_sidedness & Back)) {
            // we are behind the surface
            Nf = -N;
            Ngf = -Ng;
            return true;
        } else {
            // on the wrong side of the surface
            return false;
        }
    }

private:
    Sidedness m_sidedness;
    bool m_needs_eval;
    bool m_reflective_eval;
    // A bsdf can only perform one type of scattering
    ustring  m_scattering_label;
};



/// Subclass of ClosurePrimitive that contains the methods needed
/// for an emissive material.
class EmissiveClosure : public ClosurePrimitive {
public:
    EmissiveClosure (Sidedness side) : ClosurePrimitive (Emissive), m_sidedness (side) { }
    ~EmissiveClosure () { }

    /// Returns true if light is emitted on the specified side of this closure
    bool is_light_side(Sidedness viewing_side) const {
        return (viewing_side & m_sidedness) != 0;
    }

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
private:
    Sidedness m_sidedness;
};



/// Subclass of ClosurePrimitive that contains the serves to
/// flag a background color. No methods needed yet, only the
/// weight is going to be used
class BackgroundClosure : public ClosurePrimitive {
public:
    BackgroundClosure () : ClosurePrimitive (Background) { }
    ~BackgroundClosure () { }

};



/// Representation of an OSL 'closure color'.  It houses a linear
/// combination of weights * components (the components are references
/// to closure primitives and instance parameters).
class ClosureColor {
public:
    ClosureColor () { }
    ~ClosureColor () { }

    void clear () {
        m_components.clear ();
        m_mem.clear ();
    }

    /// Clears this closures and adds a single component with unit weight. A new
    /// ClosurePrimitive object must be placement new'd into the returned
    /// memory location
    ///
    char* allocate_component (size_t num_bytes);

    /// *this += A
    ///
    void add (const ClosureColor &A);
    const ClosureColor & operator+= (const ClosureColor &A) {
        add (A);
        return *this;
    }

    /// *this = a+b
    ///
    void add (const ClosureColor &a, const ClosureColor &b);

    /// *this *= f
    ///
    void mul (float f);
    void mul (const Color3 &w);
    const ClosureColor & operator*= (float w) { mul(w); return *this; }
    const ClosureColor & operator*= (const Color3 &w) { mul(w); return *this; }

    /// Stream output (for debugging)
    ///
    friend std::ostream & operator<< (std::ostream &out, const ClosureColor &c);

    /// Return the number of primitive components of this closure.
    ///
    int ncomponents () const { return (int) m_components.size(); }

    /// Return the weight of the i-th primitive component of this closure.
    ///
    const Color3 & weight (int i) const { return component(i).weight; }

    /// Return a pointer to the ClosurePrimitive of the i-th primitive
    /// component of this closure.
    const ClosurePrimitive * prim (int i) const {
        return reinterpret_cast<const ClosurePrimitive*>(&m_mem[component(i).memoffset]);
    }

    /// This allows for fast stealing of closure data avoiding reallocation
    void swap(ClosureColor &source) { m_components.swap(source.m_components);
                                      m_mem.swap(source.m_mem); }

private:

    /// Light-weight struct to hold a single component of the Closure.
    ///
    struct Component {
        Color3 weight;   ///< Weight of this component
        int memoffset;   ///< Offset at which we can find a ClosurePrimitive*

        Component (const Color3 &weight, int memoffset) :
            weight(weight), memoffset(memoffset) { }
    };

    /// Return the i-th component of this closure.
    ///
    const Component & component (int i) const { return m_components[i]; }

    std::vector<Component> m_components;   ///< weight + location in memory
    std::vector<char> m_mem;               ///< memory used to store components
};




}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSLCLOSURE_H */
