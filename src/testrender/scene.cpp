#include "rapidobj/rapidobj.hpp"
#include "raytracer.h"

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/timer.h>
#include <OpenImageIO/strutil.h>

OSL_NAMESPACE_ENTER

void Scene::add_model(const std::string& filename, OIIO::ErrorHandler& errhandler) {
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
    int nverts = obj_file.attributes.positions.size() / 3;
    unsigned base_idx = verts.size();
    verts.reserve(verts.size() + nverts);
    for (int i = 0, i3 = 0; i < nverts; i++, i3 += 3) {
        float x = obj_file.attributes.positions[i3 + 0];
        float y = obj_file.attributes.positions[i3 + 1];
        float z = obj_file.attributes.positions[i3 + 2];
        verts.emplace_back(x, y, z);
    }
    for (auto&& mat : obj_file.materials)
        errhandler.infofmt("Found material {}", mat.name);
    int ntris = 0;
    for (auto&& shape : obj_file.shapes) {
        errhandler.infofmt("Found mesh {}", shape.name);
        for (int i = 0, n = shape.mesh.indices.size(); i < n; i += 3) {
            int a = shape.mesh.indices[i + 0].position_index;
            int b = shape.mesh.indices[i + 1].position_index;
            int c = shape.mesh.indices[i + 2].position_index;
            indices.push_back(base_idx + a);
            indices.push_back(base_idx + b);
            indices.push_back(base_idx + c);
            ntris++;
        }
    }
    double loadtime = timer();
    errhandler.infofmt("Parsed {} vertices and {} triangles from {} in {}", nverts, ntris, filename, OIIO::Strutil::timeintervalformat(loadtime, 2));
}

OSL_NAMESPACE_EXIT
