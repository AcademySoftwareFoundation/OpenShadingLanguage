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


#include <OpenImageIO/refcnt.h>
#include <OpenImageIO/ustring.h>


#ifdef OSL_NAMESPACE
namespace OSL_NAMESPACE {
#endif

namespace OSL {


struct ClosureColorComponent;  // forward declaration



/// Representation of a radiance color closure.
/// For each BSDF or emission profile, the renderer should create a
/// subclass of ClosurePrimitive and create a single object (which
/// automatically "registers" it with the system).
/// Each subclass (BSDF) needs to overload eval() and sample().
class ClosurePrimitive {
public:
    ClosurePrimitive (ustring name, int nargs, ustring argtypes);
    virtual ~ClosurePrimitive ();

    /// Return the name of the primitive
    ///
    ustring name () const { return m_name; }

    /// Return the number of arguments that the primitive expects
    ///
    int nargs () const { return m_nargs; }

    /// Return a ustring giving the argument types expected.  For example,
    /// "vff" means that it takes arguments (vector, float, float).
    ustring argtypes () const { return m_argtypes; }

    /// Evaluate the BSDF -- Given instance parameters found in comp,
    /// incoming radiance El in the direction L, and reflection
    /// direction R, compute the outgoing radiance Er in the direction
    /// of R.  Return true if there is any (non-zero) outgoing radiance,
    /// false if there is no outgoing radiance (this allows the caller
    /// to take various shortcuts without needing to check the value of
    /// Er.  It is assumed that L and R are already normalized and point
    /// away from the surface.  It's up to the implementor of a
    /// ClosurePrimitive subclass to ensure that it conserves energy
    /// (unless it's intended to be emissive) and observes reciprocity.
    virtual bool eval (const ClosureColorComponent &comp,
                       const Vec3 &L, const Color3 &El,
                       const Vec3 &R, Color3 &Er) const {
        Er.setValue (0.0f, 0.0f, 0.0f);
        return false;
    }

    /// Sample the BSDF -- Given instance parameters found in comp,
    /// incident direction I (pointing toward the surface), and random
    /// deviates randu and randv on [0,1), return an importance-sampled
    /// direction R and the PDF.
    virtual void sample (const ClosureColorComponent &comp,
                         const Vec3 &I, float randu, float randv,
                         Vec3 &R, float &pdf) const {
        R = -I;
        pdf = 1;
    }
    
private:
    ustring m_name;
    int m_nargs;
    ustring m_argtypes;
};



/// A single component of a ClosureColor consisting of a primitive
/// function and concrete parameter values.  It's reference-counted.
struct ClosureColorComponent : public RefCnt {
public:

    /// Return the argument types this component expects
    ///
    ustring argtypes () const { return m_cprim->argtypes(); }

    /// Add a float argument
    ///
    void addarg (float f) { m_fparams.push_back (f); ++m_nargs; }
    /// Add a vector (or color, normal, point) argument
    ///
    void addarg (const Vec3 &v) {
        for (int i = 0;  i < 3;  ++i)
            m_fparams.push_back (v[i]);
        ++m_nargs;
    }
    /// Add a matrix argument
    ///
    void addarg (const Matrix44 &m) {
        for (int i = 0;  i < 16;  ++i)
            m_fparams.push_back (((float *)&m)[i]);
        ++m_nargs;
    }
    /// Add a string argument
    ///
    void addarg (ustring s) { m_sparams.push_back (s); ++m_nargs; }

    /// Compiler two components -- are they the same primitive and have
    /// all the same arguments?
    bool operator== (const ClosureColorComponent &a) const {
        return (m_cprim == a.m_cprim && argtypes() == a.argtypes() &&
                m_fparams == a.m_fparams && m_sparams == a.m_sparams);
    }

    /// Stream output (for debugging)
    ///
    friend std::ostream & operator<< (std::ostream &out,
                                      const ClosureColorComponent &comp);

private:
    const ClosurePrimitive *m_cprim; ///< Which closure primitive
    int m_nargs;                     ///< Number of arguments
    std::vector<float> m_fparams;    ///< float parameters
    std::vector<ustring> m_sparams;  ///< string parameters

    // Make the only constructor be private, so nobody can create one
    // of these except for ClosureColor::primitive.
    ClosureColorComponent (const ClosurePrimitive &prim) 
        : m_cprim(&prim), m_nargs(0) { }

    friend class ClosureColor;
};



/// Representation of an OSL 'closure color'.  It houses a linear
/// combination of weights * components (the components are references
/// to closure primitives and instance parameters).
class ClosureColor {
public:
    ClosureColor () : m_ncomps(0) { }
    ~ClosureColor () { }

    /// Typedef a reference-counted pointer to a ClosureColorComponent.
    ///
    typedef boost::intrusive_ptr<ClosureColorComponent> compref_t;

    void clear () {
        m_components.clear ();
        m_weight.clear ();
        m_ncomps = 0;
    }

    /// Add to our sum the primitive closure component with the given
    /// weight.  If we already have the identical component, just add
    /// the weights rather than add redundant components.
    void add (const compref_t &comp, const Color3 &weight);

    /// *this += A
    ///
    void add (const ClosureColor &A);

    /// *this = a+b
    ///
    void add (const ClosureColor &a, const ClosureColor &b);

    /// Assemble a primitive by name
    ///
    static compref_t primitive (ustring name);

    /// Stream output (for debugging)
    ///
    friend std::ostream & operator<< (std::ostream &out, const ClosureColor &c);

    int ncomponents () const { return m_ncomps; }
    const compref_t & component (int i) const { return m_components[i]; }
    const Color3 & weight (int i) const { return m_weight[i]; }

private:
    typedef std::vector<compref_t> ComponentVec;
    int m_ncomps;                      ///< Number of components
    ComponentVec m_components;         ///< The primitive components
    std::vector<Color3> m_weight;      ///< Spectral weight per component
};




}; // namespace OSL

#ifdef OSL_NAMESPACE
}; // end namespace OSL_NAMESPACE
using namespace OSL_NAMESPACE;
#endif


#endif /* OSLCLOSURE_H */
