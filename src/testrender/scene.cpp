#include "rapidobj/rapidobj.hpp"
#include "raytracer.h"

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/strutil.h>

OSL_NAMESPACE_ENTER

void Scene::add_model(const std::string& filename, const ShaderMap& shadermap, int shaderID, OIIO::ErrorHandler& errhandler) {
    OIIO::Timer timer;
    rapidobj::MaterialLibrary materials = rapidobj::MaterialLibrary::Default(rapidobj::Load::Optional);
    rapidobj::Result obj_file = rapidobj::ParseFile(filename, materials);
    if (obj_file.error) {
        // we were unable to parse the scene
        errhandler.errorfmt("Error while reading {} - {}", filename, obj_file.error.code.message());
        return;
    }
    if (!Triangulate(obj_file)) {
        errhandler.errorfmt("Unable to triangulate model from {}", filename);
        return;
    }
    std::vector<int> material_ids;
    bool use_material_ids = false;
    if (!shadermap.empty()) {
        // try to match up obj materials with existing shader names in our shadermap
        for (auto&& mat : obj_file.materials) {
            auto it = shadermap.find(mat.name);
            if (it != shadermap.end()) {
                errhandler.infofmt("Found material {}", mat.name);
                material_ids.emplace_back(it->second);
                use_material_ids = true;
            } else {
                material_ids.emplace_back(shaderID);
            }
        }
    }
    int nverts = obj_file.attributes.positions.size() / 3;
    unsigned base_idx = verts.size();
    verts.reserve(verts.size() + nverts);
    for (int i = 0, i3 = 0; i < nverts; i++, i3 += 3) {
        float x = obj_file.attributes.positions[i3 + 0];
        float y = obj_file.attributes.positions[i3 + 1];
        float z = obj_file.attributes.positions[i3 + 2];
        verts.emplace_back(x, y, z);
    }        
    int ntris = 0;
    for (auto&& shape : obj_file.shapes) {
        int shapeShaderID = shaderID;
        if (!use_material_ids) {
            // if we couldn't find material names to use, try to lookup matching shaders by the mesh name instead
            auto it = shadermap.find(shape.name);
            if (it != shadermap.end()) {
                errhandler.infofmt("Found mesh {}", shape.name);
                shapeShaderID = it->second;
            }
        }
        OSL_ASSERT(
            shape.mesh.material_ids.size() == 0 ||
            shape.mesh.material_ids.size() == shape.mesh.indices.size() / 3
        );
        for (int i = 0, n = shape.mesh.indices.size(), f = 0; i < n; i += 3, f++) {
            int a = shape.mesh.indices[i + 0].position_index;
            int b = shape.mesh.indices[i + 1].position_index;
            int c = shape.mesh.indices[i + 2].position_index;
            triangles.emplace_back(TriangleIndices{
                base_idx + a,
                base_idx + b,
                base_idx + c});
            if (use_material_ids && !shape.mesh.material_ids.empty()) {
                // remap the material ID to our indexing
                int obj_mat_id = shape.mesh.material_ids[f];
                OSL_ASSERT(obj_mat_id >= 0);
                OSL_ASSERT(obj_mat_id < material_ids.size());
                shaderids.emplace_back(material_ids[obj_mat_id]);
            } else
                shaderids.emplace_back(shapeShaderID);
            ntris++;
        }
    }
    double loadtime = timer();
    errhandler.infofmt("Parsed {} vertices and {} triangles from {} in {}", nverts, ntris, filename, OIIO::Strutil::timeintervalformat(loadtime, 2));
}

void Scene::add_sphere(const Vec3& c, float r, int shaderID, int resolution) {
    const int W = 2 * resolution;
    const int H = resolution;
    const int NV = 2 + W * H; // poles + grid = total vertex count
    unsigned base_idx = verts.size();
    // vertices
    verts.emplace_back(c - Vec3(0, 0, r)); // pole -z
    // W * H grid of points
    for (int y = 0; y < H; y++) {
        float s = float(y + 0.5f) / float(H);
        float z = cosf((1 - s) * float(M_PI));
        float q = sqrtf(1 - z * z);
        for (int x = 0; x < W; x++) {
            const float a = float(2 * M_PI) * float(x) / float(W);
            const Vec3 p(q * cosf(a), q * sinf(a), z);
            verts.emplace_back(c + r * p);
        }
    }
    verts.emplace_back(c + Vec3(0, 0, r)); // pole +z

    for (int x0 = 0, x1 = W - 1; x0 < W; x1 = x0, x0++) {
        // tri to pole
        triangles.emplace_back(TriangleIndices{
            base_idx,
            base_idx + 1 + x0,
            base_idx + 1 + x1
        });
        shaderids.emplace_back(shaderID);
        for (int y = 0; y < H - 1; y++) {
            // quads
            unsigned i00 = base_idx + 1 + (x0 + W * (y + 0));
            unsigned i10 = base_idx + 1 + (x1 + W * (y + 0));
            unsigned i11 = base_idx + 1 + (x1 + W * (y + 1));
            unsigned i01 = base_idx + 1 + (x0 + W * (y + 1));

            triangles.emplace_back(TriangleIndices{
                i00,
                i11,
                i10});
            triangles.emplace_back(TriangleIndices{
                i00,
                i01,
                i11});
            shaderids.emplace_back(shaderID);
            shaderids.emplace_back(shaderID);
        }
        triangles.emplace_back(TriangleIndices{
            base_idx + NV - 1,
            base_idx + NV - 1 - W + x1,
            base_idx + NV - 1 - W + x0});
        shaderids.emplace_back(shaderID);
    }
}

void Scene::add_quad(const Vec3& p, const Vec3& ex, const Vec3& ey, int shaderID, int resolution) {
    // add vertices
    unsigned base_idx = verts.size();
    for (int v = 0; v <= resolution; v++)
    for (int u = 0; u <= resolution; u++) {
        float s = float(u) / float(resolution);
        float t = float(v) / float(resolution);
        verts.emplace_back(p + s * ex + t * ey);
    }
    for (int v = 0; v < resolution; v++) 
    for (int u = 0; u < resolution; u++) {
        unsigned i00 = base_idx + (u + 0) + (v + 0) * (resolution + 1);
        unsigned i10 = base_idx + (u + 1) + (v + 0) * (resolution + 1);
        unsigned i11 = base_idx + (u + 1) + (v + 1) * (resolution + 1);
        unsigned i01 = base_idx + (u + 0) + (v + 1) * (resolution + 1);
        triangles.emplace_back(TriangleIndices{
            i00,
            i10,
            i11});

        triangles.emplace_back(TriangleIndices{
            i00,
            i11,
            i01});
        shaderids.emplace_back(shaderID);
        shaderids.emplace_back(shaderID);
    }
}


OSL_NAMESPACE_EXIT
