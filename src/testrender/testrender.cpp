/*
 * Originally based on smallpt, Copyright(c) 2006-2008 Kevin Beason 
 * (kevin.beason@gmail.com) and licensed as follows:
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files(the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <string>
#include <cmath>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/strutil.h>

#include "oslexec.h"
#include "oslclosure.h"
#include "simplerend.h"

using namespace OSL;
using namespace OpenImageIO;

static std::string outputfile = "image.png";
static std::string cameraraytype = "camera";
static std::string shadowraytype = "shadow";
static std::string reflectionraytype = "reflection";
static std::string refractionraytype = "refraction";
static std::string diffuseraytype = "diffuse";
static std::string glossyraytype = "glossy";
static int w = 1024, h = 768;
static int samples = 4;
static int allraytype = 2+4+8+16+32; // does not include "camera"
static int allraytypepluscamera = 1+2+4+8+16+32;

struct ThreadInfo {
	ShadingSystem *shadingsys;
	PerThreadInfo *handle;
	ShadingContext *ctx;
	unsigned short Xi[3];
};

struct Ray {
	Vec3 o, d;
	
	Ray(Vec3 o_, Vec3 d_)
	: o(o_), d(d_) {}
};

typedef Imath::Vec3<double> Vec3d;

struct Sphere {
	double radius;
	Vec3d p;
	ShadingAttribStateRef shaderstate;
	const char *shadername;
	bool light;

	Sphere(double radius_, Vec3d p_, const char *shadername_, bool light_)
	: radius(radius_), p(p_), shadername(shadername_), light(light_) {}

	// returns distance, 0 if no hit
	double intersect(const Ray &r) const
	{
		// Solve t^2*d.d + 2*t*(o-p).d +(o-p).(o-p)-R^2 = 0
		Vec3d op = p - Vec3d(r.o.x, r.o.y, r.o.z);
		double t, eps = 1e-4;
		double b = op.dot(Vec3(r.d.x, r.d.y, r.d.z));
		double det = b*b - op.dot(op) + radius*radius;

		if(det<0) return 0;
		else {
			det=sqrt(det);
			return(t=b-det) > eps ? t :((t=b+det)>eps ? t : 0);
		}
	}

	Vec3 normal(const Vec3& P)
	{
		return Vec3(P[0] - p[0], P[1] - p[1], P[2] - p[2]).normalized();
	}
};

// Scene(modified cornell box): radius, position, shader name
static Sphere spheres[] = {
	Sphere(1e5,  Vec3d(1e5+1, 40.8, 81.6),   "cornell_wall", false), // left
	Sphere(1e5,  Vec3d(-1e5+99, 40.8, 81.6), "cornell_wall", false), // right
	Sphere(1e5,  Vec3d(50, 40.8, 1e5),       "cornell_wall", false), // back
	Sphere(1e5,  Vec3d(50, 1e5, 81.6),       "cornell_wall", false), // front
	Sphere(1e5,  Vec3d(50, -1e5+81.6, 81.6), "cornell_wall", false), // bottom
	Sphere(1e5,  Vec3d(50, 40.8, -1e5+170),  "cornell_wall", false), // top
	Sphere(16.5, Vec3d(27, 16.5, 47),        "mirror",       false), // left ball
	Sphere(16.5, Vec3d(73, 16.5, 78),        "glass",        false), // right ball
	Sphere(600,  Vec3d(50, 681.6-1.27, 81.6),"emitter",      true)   // light
};

int numSpheres = sizeof(spheres)/sizeof(Sphere);

static inline bool intersect(const Ray &r, double &t, int &id)
{
	double d, inf=t=1e20;

	for(int i=sizeof(spheres)/sizeof(Sphere); i--;)
		if((d=spheres[i].intersect(r))&&d<t) {
			t=d;
			id=i;
		}

	return t<inf;
}

static const ClosureColor *execute_shader(ThreadInfo& thread_info, ShaderGlobals& globals, const Ray& ray, float t, int id, int depth)
{
	Sphere &obj = spheres[id];
	Vec3 P = ray.o + t*ray.d;

	memset(&globals, 0, sizeof(globals));

	// vectors
	globals.P = P;
	globals.I = -ray.d;
	globals.Ng = obj.normal(P);
	globals.N = globals.Ng;

	// uv
	float phi = atan2(P.y, P.x);
	if(phi < 0.) phi += 2*M_PI;

	globals.u = phi/(2*M_PI);
	globals.v = acos(P.z/obj.radius)/M_PI;

	// tangents
	float invzr = 1./sqrt(P.x*P.x + P.y*P.y);
	float cosphi = P.x*invzr;
	float sinphi = P.y*invzr;

	globals.dPdu = Vec3(-2*M_PI*P.y, -2*M_PI*P.x, 0.);
	globals.dPdv = Vec3(P.z*cosphi, P.z*sinphi, -obj.radius*sin(globals.v*M_PI)) * M_PI;

	// other
	globals.raytype = (depth == 0) ? allraytypepluscamera : allraytype ; // not exactly right, but...
	globals.surfacearea = 1.0f;
	globals.backfacing = (globals.Ng.dot(globals.I) < 0.0f);
	globals.Ci = NULL;

	// execute shader
	thread_info.shadingsys->execute(*(thread_info.ctx), *(obj.shaderstate), globals);

	return globals.Ci;
}

static void sample_primitive_recurse(const ClosurePrimitive*& r_prim, Color3& r_weight, const ClosureColor *closure,
	const Color3& weight, float& totw, float& r)
{
	if(closure->type == ClosureColor::COMPONENT) {
		ClosureComponent *comp = (ClosureComponent*)closure;
		ClosurePrimitive *prim = (ClosurePrimitive*)comp->data();
		float p, w = fabsf(weight[0]) + fabsf(weight[1]) + fabsf(weight[2]);

		if(w == 0.0f)
			return;

		totw += w;

		if(!r_prim) {
			// no primitive was found yet, so use this
			r_prim = prim;
			r_weight = weight/w;
		}
		else {
			p = w/totw;

			if(r < p) {
				// pick other primitive
				r_prim = prim;
				r_weight = weight/w;

				r = r/p;
			}
			else {
				// keep existing primitive
				r = (r + p)/(1.0f - p);
			}
		}
	}
	else if(closure->type == ClosureColor::MUL) {
		ClosureMul *mul = (ClosureMul*)closure;

		sample_primitive_recurse(r_prim, r_weight, mul->closure, mul->weight * weight, totw, r);
	}
	else if(closure->type == ClosureColor::ADD) {
		ClosureAdd *add = (ClosureAdd*)closure;

		sample_primitive_recurse(r_prim, r_weight, add->closureA, weight, totw, r);
		sample_primitive_recurse(r_prim, r_weight, add->closureB, weight, totw, r);
	}
}

static const ClosurePrimitive *sample_primitive(Color3& weight, const ClosureColor *closure, float r)
{
	if(closure) {
		const ClosurePrimitive *prim = NULL;
		float totw = 0.0f;

		sample_primitive_recurse(prim, weight, closure, Color3(1.0f), totw, r);
		weight *= totw;

		return prim;
	}

	return NULL;
}

static Color3 radiance(ThreadInfo& thread_info, const Ray &ray, int depth)
{
	double t; // distance to intersection
	int id = 0; // id of intersected object

	if(depth == 5)
		return Color3(0.0f);

	if(!intersect(ray, t, id))
		return Color3(0.0f, 0.0f, 0.0f); // if miss, return black

	// execute shader
	ShaderGlobals globals;
	const ClosureColor *closure = execute_shader(thread_info, globals, ray, t, id, depth);

	// sample primitive from closure tree
	Color3 weight;
	const ClosurePrimitive *prim = sample_primitive(weight, closure, erand48(thread_info.Xi));

	if(prim) {
		if(prim->category() == OSL::ClosurePrimitive::BSDF) {
			// sample BSDF closure
			BSDFClosure *bsdf = (BSDFClosure*)prim;
			Vec3 omega_in, zero(0.0f);
			Color3 eval;
			float pdf = 0.0;

			bsdf->sample(globals.Ng, globals.I, zero, zero, erand48(thread_info.Xi), erand48(thread_info.Xi),
				omega_in, zero, zero, pdf, eval);
			
			if(pdf != 0.0f) {
				Ray new_ray(globals.P, omega_in);
				Color3 r = (weight*eval/pdf)*radiance(thread_info, new_ray, depth+1);

				return r;
			}
		}
		else if(prim->category() == OSL::ClosurePrimitive::Emissive) {
			// evaluate emissive closure
			EmissiveClosure *emissive = (EmissiveClosure*)prim;
			return weight*emissive->eval(globals.Ng, globals.I);
		}
		else if(prim->category() == OSL::ClosurePrimitive::Background) {
			// background closure just returns weight
			return weight;
		}
	}

	return Color3(0.0f);
}

static void initshaders(ShadingSystem *shadingsys)
{
	// load shaders
	shadingsys->attribute("optimize", 2);
	shadingsys->attribute("lockgeom", 1);

	for(int i = 0;  i < numSpheres; i++) {
		shadingsys->ShaderGroupBegin();
		shadingsys->Shader("surface", spheres[i].shadername, NULL);
		shadingsys->ShaderGroupEnd();

		spheres[i].shaderstate = shadingsys->state();
		shadingsys->clear_state();
	}
}

static void getargs(int argc, const char *argv[])
{
	static bool help = false;
	ArgParse ap;
	ap.options("Usage:  testrender [options]",
				"--help", &help, "Print help message",
				"-d %d %d", &w, &h, "X x Y dimensions of output image",
				"-s %d", &samples, "Number of samples",
				"-o %s", &outputfile, "Output filename",
				NULL);

	if(ap.parse(argc, argv) < 0) {
		std::cerr << ap.geterror() << std::endl;
		ap.usage();
		exit(EXIT_FAILURE);
	}

	if(help) {
		std::cout << "testrender -- Use Open Shading Language with the Cornell Box\n";
		ap.usage();
		exit(EXIT_SUCCESS);
	}
}

static void write_image(float *buffer, int w, int h)
{
	// write image using OIIO
	ImageOutput *out = ImageOutput::create(outputfile);
	ImageSpec spec(w, h, 3, TypeDesc::UINT8);

	out->open(outputfile, spec);
	out->write_image(TypeDesc::FLOAT, buffer);

	out->close();
	delete out;
}

int main(int argc, const char *argv[])
{
	// parse arguments
	getargs (argc, argv);

	// create shading system
	SimpleRenderer rend;
	ErrorHandler errhandler;
	ShadingSystem *shadingsys;

	shadingsys = ShadingSystem::create (&rend, NULL, &errhandler);
	
	// create shaders
	initshaders (shadingsys);

	// create thread info
	ThreadInfo thread_info;
	thread_info.shadingsys = shadingsys;
	thread_info.handle = shadingsys->create_thread_info ();
	thread_info.ctx = shadingsys->get_context (thread_info.handle);

	// camera parameters
	Vec3 cam_o = Vec3(50, 52, 295.6);
	Vec3 cam_d = Vec3(0, -0.042612, -1).normalized();
	Vec3 cam_x = Vec3(w*.5135/h, 0.0, 0.0);
	Vec3 cam_y = (cam_x.cross(cam_d)).normalized()*.5135;

	// render
	Color3 *buffer = new Color3[w*h];
	float scale = 10.0f;
	int samps = (samples > 4)? samples/4: 1;

	for (int y=0; y<h; y++) {
		fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps*4, 100.*y/(h-1));

		thread_info.Xi[0] = 0;
		thread_info.Xi[1] = 0;
		thread_info.Xi[2] = y*y*y;

		for (int x=0; x<w; x++) {
			int i = (h - y - 1)*w + x;

			// 2x2 subixel with tent filter
			for(int sy = 0; sy < 2; sy++) {
				for(int sx = 0; sx < 2; sx++) {
					Vec3 r(0.0f);

					for (int s=0; s<samps; s++) {
						double r1 = 2*erand48(thread_info.Xi);
						double r2 = 2*erand48(thread_info.Xi);
						
						double dx = (r1 < 1)? sqrt(r1)-1: 1-sqrt(2-r1);
						double dy = (r2 < 1)? sqrt(r2)-1: 1-sqrt(2-r2);

						Vec3 d = cam_x*(((sx+.5 + dx)/2 + x)/w - .5) +
								 cam_y*(((sy+.5 + dy)/2 + y)/h - .5) + cam_d;

						Ray ray(cam_o + d*130, d.normalized());
						r = r + radiance(thread_info, ray, 0)*(1.0f/samps);
					}

					r = Vec3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
					buffer[i] += r*scale*0.25f;
				}
			}

			// gamma
			buffer[i] = Vec3(pow(buffer[i].x, 1.0/2.2), pow(buffer[i].y, 1.0/2.2), pow(buffer[i].z, 1.0/2.2));
		}
	}

	fprintf(stderr, "\n");

	// free OSL data
	shadingsys->release_context (thread_info.ctx);
	shadingsys->destroy_thread_info (thread_info.handle);

	ShadingSystem::destroy (shadingsys);

	// write image
	write_image ((float*)buffer, w, h);
	delete buffer;

	return EXIT_SUCCESS;
}
