/*

rapidobj - Fast Wavefront .obj file loader

Licensed under the MIT License <http://opensource.org/licenses/MIT>
SPDX-License-Identifier: MIT
Copyright (c) 2024 Slobodan Pavlic

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef RAPID_OBJ_INCLUDE_HEADER_GUARD
#define RAPID_OBJ_INCLUDE_HEADER_GUARD

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#ifdef __linux__

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#elif _WIN32

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#elif __APPLE__

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

#define RAPIDOBJ_VERSION_MAJOR 1
#define RAPIDOBJ_VERSION_MINOR 1
#define RAPIDOBJ_VERSION_PATCH 0

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Public API Begins
//

namespace rapidobj {

static constexpr struct {
    int Major;
    int Minor;
    int Patch;
} Version = { RAPIDOBJ_VERSION_MAJOR, RAPIDOBJ_VERSION_MINOR, RAPIDOBJ_VERSION_PATCH };

enum class Load { Mandatory, Optional };

struct MaterialLibrary final {
    static MaterialLibrary Default() { return MaterialLibrary(); }
    static MaterialLibrary Default(Load policy) { return MaterialLibrary(policy); }

    static MaterialLibrary SearchPath(std::filesystem::path path, Load policy = Load::Mandatory)
    {
        return MaterialLibrary({ std::move(path) }, policy);
    }

    static MaterialLibrary SearchPaths(std::vector<std::filesystem::path> paths, Load policy = Load::Mandatory)
    {
        return MaterialLibrary(std::move(paths), policy);
    }

    static MaterialLibrary String(std::string_view text) { return MaterialLibrary(text); }

    static MaterialLibrary Ignore() { return MaterialLibrary(nullptr); }

    const auto& Value() const noexcept { return m_value; }
    const auto& Policy() const noexcept { return m_policy; }

  private:
    MaterialLibrary() noexcept = default;
    MaterialLibrary(Load policy) noexcept : m_policy(policy) {}
    MaterialLibrary(std::vector<std::filesystem::path>&& paths, Load policy) noexcept
        : m_value(std::move(paths)), m_policy(policy)
    {}
    MaterialLibrary(std::string_view text) noexcept : m_value(text) {}
    MaterialLibrary(std::nullptr_t) noexcept : m_value(nullptr) {}

    using Variant = std::variant<std::monostate, std::nullptr_t, std::vector<std::filesystem::path>, std::string_view>;

    Variant             m_value{};
    std::optional<Load> m_policy{};
};

template <typename T>
class Array final {
  public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    Array() noexcept {}
    Array(size_type size) : m_state{ size ? std::unique_ptr<T[]>(new T[size]) : nullptr, size } {}

    template <typename Iter>
    Array(Iter begin, Iter end)
    {
        static_assert(std::is_same_v<std::random_access_iterator_tag, typename Iter::iterator_category>);
        static_assert(std::is_same_v<value_type, typename Iter::value_type>);

        const auto size = static_cast<size_type>(end - begin);
        m_state.data.reset(new value_type[size]);
        memcpy(&*begin, m_state.data.get(), size * sizeof(value_type));
        m_state.size = size;
    }

    Array(const Array&)            = delete;
    Array& operator=(const Array&) = delete;

    Array(Array&& other) noexcept { m_state = std::exchange(other.m_state, {}); }
    Array& operator=(Array&& other) noexcept
    {
        if (this != &other) {
            m_state = std::exchange(other.m_state, {});
        }
        return *this;
    }

    constexpr reference       operator[](size_type index) noexcept { return m_state.data[index]; }
    constexpr const_reference operator[](size_type index) const noexcept { return m_state.data[index]; }
    constexpr reference       front() noexcept { return m_state.data[0]; }
    constexpr const_reference front() const noexcept { return m_state.data[0]; }
    constexpr reference       back() noexcept { return m_state.data[m_state.size - 1]; }
    constexpr const_reference back() const noexcept { return m_state.data[m_state.size - 1]; }
    constexpr pointer         data() noexcept { return m_state.data.get(); }
    constexpr const_pointer   data() const noexcept { return m_state.data.get(); }
    constexpr iterator        begin() noexcept { return m_state.data.get(); }
    constexpr const_iterator  begin() const noexcept { return m_state.data.get(); }
    constexpr const_iterator  cbegin() const noexcept { return m_state.data.get(); }
    constexpr iterator        end() noexcept { return m_state.data.get() + m_state.size; }
    constexpr const_iterator  end() const noexcept { return m_state.data.get() + m_state.size; }
    constexpr const_iterator  cend() const noexcept { return m_state.data.get() + m_state.size; }
    constexpr bool            empty() const noexcept { return m_state.size == 0; }
    constexpr size_type       size() const noexcept { return m_state.size; }
    void                      swap(Array& other) noexcept { std::swap(m_state.state, other.m_state.state); }

  private:
    struct final {
        std::unique_ptr<T[]> data;
        size_type            size;
    } m_state{};

    static_assert(
        std::is_trivially_constructible_v<value_type> && std::is_trivially_destructible_v<value_type> &&
        std::is_trivially_copyable_v<value_type>);
};

template <typename Iter>
Array(Iter b, Iter e) -> Array<typename std::iterator_traits<Iter>::value_type>;

template <typename T>
bool operator==(const Array<T>& lhs, const Array<T>& rhs)
{
    return (lhs.size() == rhs.size()) && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T>
bool operator!=(const Array<T>& lhs, const Array<T>& rhs)
{
    return !operator==(lhs, rhs);
}

struct Attributes final {
    Array<float> positions; // 'v'  (xyz)
    Array<float> texcoords; // 'vt' (uv)
    Array<float> normals;   // 'vn' (xyz)
    Array<float> colors;    //  vertex color extension (see http://paulbourke.net/dataformats/obj/colour.html)
};

struct Index final {
    int position_index;
    int texcoord_index;
    int normal_index;
};

struct Mesh final {
    Array<Index>    indices;             // Position/Texture/Normal indices
    Array<uint8_t>  num_face_vertices;   // Number of vertices per face: 3 (triangle), 4 (quad), ... , 255
    Array<int32_t>  material_ids;        // Material ID per face
    Array<uint32_t> smoothing_group_ids; // Smoothing group ID per face (group id 0 means off)
};

struct Lines final {
    Array<Index>   indices;           // Polyline indices
    Array<int32_t> num_line_vertices; // Number of vertices per polyline
};

struct Points final {
    Array<Index> indices; // Points indices
};

struct Shape final {
    std::string name;
    Mesh        mesh;
    Lines       lines;
    Points      points;
};

using Shapes = std::vector<Shape>;

enum class TextureType { None, Sphere, CubeTop, CubeBottom, CubeFront, CubeBack, CubeLeft, CubeRight };

using Float3 = std::array<float, 3>;

// see https://en.wikipedia.org/wiki/Wavefront_.obj_file#Texture_options
struct TextureOption final {
    TextureType type               = TextureType::None; // -type
    float       sharpness          = 1.0f;              // -boost
    float       brightness         = 0.0f;              // -mm
    float       contrast           = 1.0f;              // -mm
    Float3      origin_offset      = { 0, 0, 0 };       // -o
    Float3      scale              = { 1, 1, 1 };       // -s
    Float3      turbulence         = { 0, 0, 0 };       // -t
    int         texture_resolution = -1;                // -texres
    bool        clamp              = false;             // -clamp
    char        imfchan            = 'm';               // -imfchan
    bool        blendu             = true;              // -blendu
    bool        blendv             = true;              // -blendv
    float       bump_multiplier    = 1.0f;              // -bm
};

struct Material final {
    std::string name;

    // Material parameters (see http://www.fileformat.info/format/material/)
    Float3 ambient       = { 0, 0, 0 }; // Ka
    Float3 diffuse       = { 0, 0, 0 }; // Kd
    Float3 specular      = { 0, 0, 0 }; // Ks
    Float3 transmittance = { 0, 0, 0 }; // Kt
    Float3 emission      = { 0, 0, 0 }; // Ke
    float  shininess     = 1.0f;        // Ns
    float  ior           = 1.0f;        // Ni
    float  dissolve      = 1.0f;        // d
    int    illum         = 0;           // illum

    std::string ambient_texname;            // map_Ka
    std::string diffuse_texname;            // map_Kd
    std::string specular_texname;           // map_Ks
    std::string specular_highlight_texname; // map_Ns
    std::string bump_texname;               // map_bump, map_Bump, bump
    std::string displacement_texname;       // disp
    std::string alpha_texname;              // map_d
    std::string reflection_texname;         // refl

    TextureOption ambient_texopt;
    TextureOption diffuse_texopt;
    TextureOption specular_texopt;
    TextureOption specular_highlight_texopt;
    TextureOption bump_texopt;
    TextureOption displacement_texopt;
    TextureOption alpha_texopt;
    TextureOption reflection_texopt;

    // PBR extension (see http://exocortex.com/blog/extending_wavefront_mtl_to_support_pbr)
    float roughness           = 0.0f; // Pr
    float metallic            = 0.0f; // Pm
    float sheen               = 0.0f; // Ps
    float clearcoat_thickness = 0.0f; // Pc
    float clearcoat_roughness = 0.0f; // Pcr
    float anisotropy          = 0.0f; // aniso
    float anisotropy_rotation = 0.0f; // anisor

    std::string roughness_texname; // map_Pr
    std::string metallic_texname;  // map_Pm
    std::string sheen_texname;     // map_Ps
    std::string emissive_texname;  // map_Ke
    std::string normal_texname;    // norm

    TextureOption roughness_texopt;
    TextureOption metallic_texopt;
    TextureOption sheen_texopt;
    TextureOption emissive_texopt;
    TextureOption normal_texopt;
};

using Materials = std::vector<Material>;

struct Error final {
    explicit operator bool() const noexcept { return code.value(); }

    std::error_code code{};
    std::string     line{};
    size_t          line_num{};
};

struct [[nodiscard]] Result final {
    Attributes attributes;
    Shapes     shapes;
    Materials  materials;
    Error      error;
};

inline Result ParseFile(
    const std::filesystem::path& obj_filepath,
    const MaterialLibrary&       mtl_library = MaterialLibrary::Default());

inline Result ParseStream(std::istream& obj_stream, const MaterialLibrary& mtl_library = MaterialLibrary::Default());

inline bool Triangulate(Result& result);

} // namespace rapidobj

//
// Public API Ends
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Third-party includes Begin
//

// clang-format off

//
// Code below is from: https://github.com/mapbox/earcut.hpp/tree/v2.2.4
//

// ISC License
//
// Copyright (c) 2015, Mapbox
//
// Permission to use, copy, modify, and/or distribute this software for any purpose
// with or without fee is hereby granted, provided that the above copyright notice
// and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
// REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
// INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
// OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
// TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
// THIS SOFTWARE.

namespace mapbox {

namespace util {

template <std::size_t I, typename T> struct nth {
    inline static typename std::tuple_element<I, T>::type
    get(const T& t) { return std::get<I>(t); };
};

}

namespace detail {

template <typename N = uint32_t>
class Earcut {
public:
    std::vector<N> indices;
    std::size_t vertices = 0;

    template <typename Polygon>
    void operator()(const Polygon& points);

private:
    struct Node {
        Node(N index, double x_, double y_) : i(index), x(x_), y(y_) {}
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        Node(Node&&) = delete;
        Node& operator=(Node&&) = delete;

        const N i;
        const double x;
        const double y;

        // previous and next vertice nodes in a polygon ring
        Node* prev = nullptr;
        Node* next = nullptr;

        // z-order curve value
        int32_t z = 0;

        // previous and next nodes in z-order
        Node* prevZ = nullptr;
        Node* nextZ = nullptr;

        // indicates whether this is a steiner point
        bool steiner = false;
    };

    template <typename Ring> Node* linkedList(const Ring& points, const bool clockwise);
    Node* filterPoints(Node* start, Node* end = nullptr);
    void earcutLinked(Node* ear, int pass = 0);
    bool isEar(Node* ear);
    bool isEarHashed(Node* ear);
    Node* cureLocalIntersections(Node* start);
    void splitEarcut(Node* start);
    template <typename Polygon> Node* eliminateHoles(const Polygon& points, Node* outerNode);
    Node* eliminateHole(Node* hole, Node* outerNode);
    Node* findHoleBridge(Node* hole, Node* outerNode);
    bool sectorContainsSector(const Node* m, const Node* p);
    void indexCurve(Node* start);
    Node* sortLinked(Node* list);
    int32_t zOrder(const double x_, const double y_);
    Node* getLeftmost(Node* start);
    bool pointInTriangle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) const;
    bool isValidDiagonal(Node* a, Node* b);
    double area(const Node* p, const Node* q, const Node* r) const;
    bool equals(const Node* p1, const Node* p2);
    bool intersects(const Node* p1, const Node* q1, const Node* p2, const Node* q2);
    bool onSegment(const Node* p, const Node* q, const Node* r);
    int sign(double val);
    bool intersectsPolygon(const Node* a, const Node* b);
    bool locallyInside(const Node* a, const Node* b);
    bool middleInside(const Node* a, const Node* b);
    Node* splitPolygon(Node* a, Node* b);
    template <typename Point> Node* insertNode(std::size_t i, const Point& p, Node* last);
    void removeNode(Node* p);

    bool hashing;
    double minX, maxX;
    double minY, maxY;
    double inv_size = 0;

    template <typename T, typename Alloc = std::allocator<T>>
    class ObjectPool {
    public:
        ObjectPool() { }
        ObjectPool(std::size_t blockSize_) {
            reset(blockSize_);
        }
        ~ObjectPool() {
            clear();
        }
        template <typename... Args>
        T* construct(Args&&... args) {
            if (currentIndex >= blockSize) {
                currentBlock = alloc_traits::allocate(alloc, blockSize);
                allocations.emplace_back(currentBlock);
                currentIndex = 0;
            }
            T* object = &currentBlock[currentIndex++];
            alloc_traits::construct(alloc, object, std::forward<Args>(args)...);
            return object;
        }
        void reset(std::size_t newBlockSize) {
            for (auto allocation : allocations) {
                alloc_traits::deallocate(alloc, allocation, blockSize);
            }
            allocations.clear();
            blockSize = std::max<std::size_t>(1, newBlockSize);
            currentBlock = nullptr;
            currentIndex = blockSize;
        }
        void clear() { reset(blockSize); }
    private:
        T* currentBlock = nullptr;
        std::size_t currentIndex = 1;
        std::size_t blockSize = 1;
        std::vector<T*> allocations;
        Alloc alloc;
        typedef typename std::allocator_traits<Alloc> alloc_traits;
    };
    ObjectPool<Node> nodes;
};

template <typename N> template <typename Polygon>
void Earcut<N>::operator()(const Polygon& points) {
    // reset
    indices.clear();
    vertices = 0;

    if (points.empty()) return;

    double x;
    double y;
    int threshold = 80;
    std::size_t len = 0;

    for (size_t i = 0; threshold >= 0 && i < points.size(); i++) {
        threshold -= static_cast<int>(points[i].size());
        len += points[i].size();
    }

    //estimate size of nodes and indices
    nodes.reset(len * 3 / 2);
    indices.reserve(len + points[0].size());

    Node* outerNode = linkedList(points[0], true);
    if (!outerNode || outerNode->prev == outerNode->next) return;

    if (points.size() > 1) outerNode = eliminateHoles(points, outerNode);

    // if the shape is not too simple, we'll use z-order curve hash later; calculate polygon bbox
    hashing = threshold < 0;
    if (hashing) {
        Node* p = outerNode->next;
        minX = maxX = outerNode->x;
        minY = maxY = outerNode->y;
        do {
            x = p->x;
            y = p->y;
            minX = std::min<double>(minX, x);
            minY = std::min<double>(minY, y);
            maxX = std::max<double>(maxX, x);
            maxY = std::max<double>(maxY, y);
            p = p->next;
        } while (p != outerNode);

        // minX, minY and inv_size are later used to transform coords into integers for z-order calculation
        inv_size = std::max<double>(maxX - minX, maxY - minY);
        inv_size = inv_size != .0 ? (32767. / inv_size) : .0;
    }

    earcutLinked(outerNode);

    nodes.clear();
}

// create a circular doubly linked list from polygon points in the specified winding order
template <typename N> template <typename Ring>
typename Earcut<N>::Node*
Earcut<N>::linkedList(const Ring& points, const bool clockwise) {
    using Point = typename Ring::value_type;
    double sum = 0;
    const std::size_t len = points.size();
    std::size_t i, j;
    Node* last = nullptr;

    // calculate original winding order of a polygon ring
    for (i = 0, j = len > 0 ? len - 1 : 0; i < len; j = i++) {
        const auto& p1 = points[i];
        const auto& p2 = points[j];
        const double p20 = util::nth<0, Point>::get(p2);
        const double p10 = util::nth<0, Point>::get(p1);
        const double p11 = util::nth<1, Point>::get(p1);
        const double p21 = util::nth<1, Point>::get(p2);
        sum += (p20 - p10) * (p11 + p21);
    }

    // link points into circular doubly-linked list in the specified winding order
    if (clockwise == (sum > 0)) {
        for (i = 0; i < len; i++) last = insertNode(vertices + i, points[i], last);
    } else {
        for (i = len; i-- > 0;) last = insertNode(vertices + i, points[i], last);
    }

    if (last && equals(last, last->next)) {
        removeNode(last);
        last = last->next;
    }

    vertices += len;

    return last;
}

// eliminate colinear or duplicate points
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::filterPoints(Node* start, Node* end) {
    if (!end) end = start;

    Node* p = start;
    bool again;
    do {
        again = false;

        if (!p->steiner && (equals(p, p->next) || area(p->prev, p, p->next) == 0)) {
            removeNode(p);
            p = end = p->prev;

            if (p == p->next) break;
            again = true;

        } else {
            p = p->next;
        }
    } while (again || p != end);

    return end;
}

// main ear slicing loop which triangulates a polygon (given as a linked list)
template <typename N>
void Earcut<N>::earcutLinked(Node* ear, int pass) {
    if (!ear) return;

    // interlink polygon nodes in z-order
    if (!pass && hashing) indexCurve(ear);

    Node* stop = ear;
    Node* prev;
    Node* next;

    // iterate through ears, slicing them one by one
    while (ear->prev != ear->next) {
        prev = ear->prev;
        next = ear->next;

        if (hashing ? isEarHashed(ear) : isEar(ear)) {
            // cut off the triangle
            indices.emplace_back(prev->i);
            indices.emplace_back(ear->i);
            indices.emplace_back(next->i);

            removeNode(ear);

            // skipping the next vertice leads to less sliver triangles
            ear = next->next;
            stop = next->next;

            continue;
        }

        ear = next;

        // if we looped through the whole remaining polygon and can't find any more ears
        if (ear == stop) {
            // try filtering points and slicing again
            if (!pass) earcutLinked(filterPoints(ear), 1);

            // if this didn't work, try curing all small self-intersections locally
            else if (pass == 1) {
                ear = cureLocalIntersections(filterPoints(ear));
                earcutLinked(ear, 2);

            // as a last resort, try splitting the remaining polygon into two
            } else if (pass == 2) splitEarcut(ear);

            break;
        }
    }
}

// check whether a polygon node forms a valid ear with adjacent nodes
template <typename N>
bool Earcut<N>::isEar(Node* ear) {
    const Node* a = ear->prev;
    const Node* b = ear;
    const Node* c = ear->next;

    if (area(a, b, c) >= 0) return false; // reflex, can't be an ear

    // now make sure we don't have other points inside the potential ear
    Node* p = ear->next->next;

    while (p != ear->prev) {
        if (pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
            area(p->prev, p, p->next) >= 0) return false;
        p = p->next;
    }

    return true;
}

template <typename N>
bool Earcut<N>::isEarHashed(Node* ear) {
    const Node* a = ear->prev;
    const Node* b = ear;
    const Node* c = ear->next;

    if (area(a, b, c) >= 0) return false; // reflex, can't be an ear

    // triangle bbox; min & max are calculated like this for speed
    const double minTX = std::min<double>(a->x, std::min<double>(b->x, c->x));
    const double minTY = std::min<double>(a->y, std::min<double>(b->y, c->y));
    const double maxTX = std::max<double>(a->x, std::max<double>(b->x, c->x));
    const double maxTY = std::max<double>(a->y, std::max<double>(b->y, c->y));

    // z-order range for the current triangle bbox;
    const int32_t minZ = zOrder(minTX, minTY);
    const int32_t maxZ = zOrder(maxTX, maxTY);

    // first look for points inside the triangle in increasing z-order
    Node* p = ear->nextZ;

    while (p && p->z <= maxZ) {
        if (p != ear->prev && p != ear->next &&
            pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
            area(p->prev, p, p->next) >= 0) return false;
        p = p->nextZ;
    }

    // then look for points in decreasing z-order
    p = ear->prevZ;

    while (p && p->z >= minZ) {
        if (p != ear->prev && p != ear->next &&
            pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
            area(p->prev, p, p->next) >= 0) return false;
        p = p->prevZ;
    }

    return true;
}

// go through all polygon nodes and cure small local self-intersections
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::cureLocalIntersections(Node* start) {
    Node* p = start;
    do {
        Node* a = p->prev;
        Node* b = p->next->next;

        // a self-intersection where edge (v[i-1],v[i]) intersects (v[i+1],v[i+2])
        if (!equals(a, b) && intersects(a, p, p->next, b) && locallyInside(a, b) && locallyInside(b, a)) {
            indices.emplace_back(a->i);
            indices.emplace_back(p->i);
            indices.emplace_back(b->i);

            // remove two nodes involved
            removeNode(p);
            removeNode(p->next);

            p = start = b;
        }
        p = p->next;
    } while (p != start);

    return filterPoints(p);
}

// try splitting polygon into two and triangulate them independently
template <typename N>
void Earcut<N>::splitEarcut(Node* start) {
    // look for a valid diagonal that divides the polygon into two
    Node* a = start;
    do {
        Node* b = a->next->next;
        while (b != a->prev) {
            if (a->i != b->i && isValidDiagonal(a, b)) {
                // split the polygon in two by the diagonal
                Node* c = splitPolygon(a, b);

                // filter colinear points around the cuts
                a = filterPoints(a, a->next);
                c = filterPoints(c, c->next);

                // run earcut on each half
                earcutLinked(a);
                earcutLinked(c);
                return;
            }
            b = b->next;
        }
        a = a->next;
    } while (a != start);
}

// link every hole into the outer loop, producing a single-ring polygon without holes
template <typename N> template <typename Polygon>
typename Earcut<N>::Node*
Earcut<N>::eliminateHoles(const Polygon& points, Node* outerNode) {
    const size_t len = points.size();

    std::vector<Node*> queue;
    for (size_t i = 1; i < len; i++) {
        Node* list = linkedList(points[i], false);
        if (list) {
            if (list == list->next) list->steiner = true;
            queue.push_back(getLeftmost(list));
        }
    }
    std::sort(queue.begin(), queue.end(), [](const Node* a, const Node* b) {
        return a->x < b->x;
    });

    // process holes from left to right
    for (size_t i = 0; i < queue.size(); i++) {
        outerNode = eliminateHole(queue[i], outerNode);
    }

    return outerNode;
}

// find a bridge between vertices that connects hole with an outer ring and and link it
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::eliminateHole(Node* hole, Node* outerNode) {
    Node* bridge = findHoleBridge(hole, outerNode);
    if (!bridge) {
        return outerNode;
    }

    Node* bridgeReverse = splitPolygon(bridge, hole);

    // filter collinear points around the cuts
    filterPoints(bridgeReverse, bridgeReverse->next);

    // Check if input node was removed by the filtering
    return filterPoints(bridge, bridge->next);
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::findHoleBridge(Node* hole, Node* outerNode) {
    Node* p = outerNode;
    double hx = hole->x;
    double hy = hole->y;
    double qx = -std::numeric_limits<double>::infinity();
    Node* m = nullptr;

    // find a segment intersected by a ray from the hole's leftmost Vertex to the left;
    // segment's endpoint with lesser x will be potential connection Vertex
    do {
        if (hy <= p->y && hy >= p->next->y && p->next->y != p->y) {
          double x = p->x + (hy - p->y) * (p->next->x - p->x) / (p->next->y - p->y);
          if (x <= hx && x > qx) {
            qx = x;
            m = p->x < p->next->x ? p : p->next;
            if (x == hx) return m; // hole touches outer segment; pick leftmost endpoint
          }
        }
        p = p->next;
    } while (p != outerNode);

    if (!m) return 0;

    // look for points inside the triangle of hole Vertex, segment intersection and endpoint;
    // if there are no points found, we have a valid connection;
    // otherwise choose the Vertex of the minimum angle with the ray as connection Vertex

    const Node* stop = m;
    double tanMin = std::numeric_limits<double>::infinity();
    double tanCur = 0;

    p = m;
    double mx = m->x;
    double my = m->y;

    do {
        if (hx >= p->x && p->x >= mx && hx != p->x &&
            pointInTriangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p->x, p->y)) {

            tanCur = std::abs(hy - p->y) / (hx - p->x); // tangential

            if (locallyInside(p, hole) &&
                (tanCur < tanMin || (tanCur == tanMin && (p->x > m->x || sectorContainsSector(m, p))))) {
                m = p;
                tanMin = tanCur;
            }
        }

        p = p->next;
    } while (p != stop);

    return m;
}

// whether sector in vertex m contains sector in vertex p in the same coordinates
template <typename N>
bool Earcut<N>::sectorContainsSector(const Node* m, const Node* p) {
    return area(m->prev, m, p->prev) < 0 && area(p->next, m, m->next) < 0;
}

// interlink polygon nodes in z-order
template <typename N>
void Earcut<N>::indexCurve(Node* start) {
    assert(start);
    Node* p = start;

    do {
        p->z = p->z ? p->z : zOrder(p->x, p->y);
        p->prevZ = p->prev;
        p->nextZ = p->next;
        p = p->next;
    } while (p != start);

    p->prevZ->nextZ = nullptr;
    p->prevZ = nullptr;

    sortLinked(p);
}

// Simon Tatham's linked list merge sort algorithm
// http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::sortLinked(Node* list) {
    assert(list);
    Node* p;
    Node* q;
    Node* e;
    Node* tail;
    int i, numMerges, pSize, qSize;
    int inSize = 1;

    for (;;) {
        p = list;
        list = nullptr;
        tail = nullptr;
        numMerges = 0;

        while (p) {
            numMerges++;
            q = p;
            pSize = 0;
            for (i = 0; i < inSize; i++) {
                pSize++;
                q = q->nextZ;
                if (!q) break;
            }

            qSize = inSize;

            while (pSize > 0 || (qSize > 0 && q)) {

                if (pSize == 0) {
                    e = q;
                    q = q->nextZ;
                    qSize--;
                } else if (qSize == 0 || !q) {
                    e = p;
                    p = p->nextZ;
                    pSize--;
                } else if (p->z <= q->z) {
                    e = p;
                    p = p->nextZ;
                    pSize--;
                } else {
                    e = q;
                    q = q->nextZ;
                    qSize--;
                }

                if (tail) tail->nextZ = e;
                else list = e;

                e->prevZ = tail;
                tail = e;
            }

            p = q;
        }

        tail->nextZ = nullptr;

        if (numMerges <= 1) return list;

        inSize *= 2;
    }
}

// z-order of a Vertex given coords and size of the data bounding box
template <typename N>
int32_t Earcut<N>::zOrder(const double x_, const double y_) {
    // coords are transformed into non-negative 15-bit integer range
    int32_t x = static_cast<int32_t>((x_ - minX) * inv_size);
    int32_t y = static_cast<int32_t>((y_ - minY) * inv_size);

    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

// find the leftmost node of a polygon ring
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::getLeftmost(Node* start) {
    Node* p = start;
    Node* leftmost = start;
    do {
        if (p->x < leftmost->x || (p->x == leftmost->x && p->y < leftmost->y))
            leftmost = p;
        p = p->next;
    } while (p != start);

    return leftmost;
}

// check if a point lies within a convex triangle
template <typename N>
bool Earcut<N>::pointInTriangle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) const {
    return (cx - px) * (ay - py) >= (ax - px) * (cy - py) &&
           (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
           (bx - px) * (cy - py) >= (cx - px) * (by - py);
}

// check if a diagonal between two polygon nodes is valid (lies in polygon interior)
template <typename N>
bool Earcut<N>::isValidDiagonal(Node* a, Node* b) {
    return a->next->i != b->i && a->prev->i != b->i && !intersectsPolygon(a, b) && // dones't intersect other edges
           ((locallyInside(a, b) && locallyInside(b, a) && middleInside(a, b) && // locally visible
            (area(a->prev, a, b->prev) != 0.0 || area(a, b->prev, b) != 0.0)) || // does not create opposite-facing sectors
            (equals(a, b) && area(a->prev, a, a->next) > 0 && area(b->prev, b, b->next) > 0)); // special zero-length case
}

// signed area of a triangle
template <typename N>
double Earcut<N>::area(const Node* p, const Node* q, const Node* r) const {
    return (q->y - p->y) * (r->x - q->x) - (q->x - p->x) * (r->y - q->y);
}

// check if two points are equal
template <typename N>
bool Earcut<N>::equals(const Node* p1, const Node* p2) {
    return p1->x == p2->x && p1->y == p2->y;
}

// check if two segments intersect
template <typename N>
bool Earcut<N>::intersects(const Node* p1, const Node* q1, const Node* p2, const Node* q2) {
    int o1 = sign(area(p1, q1, p2));
    int o2 = sign(area(p1, q1, q2));
    int o3 = sign(area(p2, q2, p1));
    int o4 = sign(area(p2, q2, q1));

    if (o1 != o2 && o3 != o4) return true; // general case

    if (o1 == 0 && onSegment(p1, p2, q1)) return true; // p1, q1 and p2 are collinear and p2 lies on p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true; // p1, q1 and q2 are collinear and q2 lies on p1q1
    if (o3 == 0 && onSegment(p2, p1, q2)) return true; // p2, q2 and p1 are collinear and p1 lies on p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true; // p2, q2 and q1 are collinear and q1 lies on p2q2

    return false;
}

// for collinear points p, q, r, check if point q lies on segment pr
template <typename N>
bool Earcut<N>::onSegment(const Node* p, const Node* q, const Node* r) {
    return q->x <= std::max<double>(p->x, r->x) &&
        q->x >= std::min<double>(p->x, r->x) &&
        q->y <= std::max<double>(p->y, r->y) &&
        q->y >= std::min<double>(p->y, r->y);
}

template <typename N>
int Earcut<N>::sign(double val) {
    return (0.0 < val) - (val < 0.0);
}

// check if a polygon diagonal intersects any polygon segments
template <typename N>
bool Earcut<N>::intersectsPolygon(const Node* a, const Node* b) {
    const Node* p = a;
    do {
        if (p->i != a->i && p->next->i != a->i && p->i != b->i && p->next->i != b->i &&
                intersects(p, p->next, a, b)) return true;
        p = p->next;
    } while (p != a);

    return false;
}

// check if a polygon diagonal is locally inside the polygon
template <typename N>
bool Earcut<N>::locallyInside(const Node* a, const Node* b) {
    return area(a->prev, a, a->next) < 0 ?
        area(a, b, a->next) >= 0 && area(a, a->prev, b) >= 0 :
        area(a, b, a->prev) < 0 || area(a, a->next, b) < 0;
}

// check if the middle Vertex of a polygon diagonal is inside the polygon
template <typename N>
bool Earcut<N>::middleInside(const Node* a, const Node* b) {
    const Node* p = a;
    bool inside = false;
    double px = (a->x + b->x) / 2;
    double py = (a->y + b->y) / 2;
    do {
        if (((p->y > py) != (p->next->y > py)) && p->next->y != p->y &&
                (px < (p->next->x - p->x) * (py - p->y) / (p->next->y - p->y) + p->x))
            inside = !inside;
        p = p->next;
    } while (p != a);

    return inside;
}

// link two polygon vertices with a bridge; if the vertices belong to the same ring, it splits
// polygon into two; if one belongs to the outer ring and another to a hole, it merges it into a
// single ring
template <typename N>
typename Earcut<N>::Node*
Earcut<N>::splitPolygon(Node* a, Node* b) {
    Node* a2 = nodes.construct(a->i, a->x, a->y);
    Node* b2 = nodes.construct(b->i, b->x, b->y);
    Node* an = a->next;
    Node* bp = b->prev;

    a->next = b;
    b->prev = a;

    a2->next = an;
    an->prev = a2;

    b2->next = a2;
    a2->prev = b2;

    bp->next = b2;
    b2->prev = bp;

    return b2;
}

// create a node and util::optionally link it with previous one (in a circular doubly linked list)
template <typename N> template <typename Point>
typename Earcut<N>::Node*
Earcut<N>::insertNode(std::size_t i, const Point& pt, Node* last) {
    Node* p = nodes.construct(static_cast<N>(i), util::nth<0, Point>::get(pt), util::nth<1, Point>::get(pt));

    if (!last) {
        p->prev = p;
        p->next = p;

    } else {
        assert(last);
        p->next = last->next;
        p->prev = last;
        last->next->prev = p;
        last->next = p;
    }
    return p;
}

template <typename N>
void Earcut<N>::removeNode(Node* p) {
    p->next->prev = p->prev;
    p->prev->next = p->next;

    if (p->prevZ) p->prevZ->nextZ = p->nextZ;
    if (p->nextZ) p->nextZ->prevZ = p->prevZ;
}
}

template <typename N = uint32_t, typename Polygon>
std::vector<N> earcut(const Polygon& poly) {
    mapbox::detail::Earcut<N> earcut;
    earcut(poly);
    return std::move(earcut.indices);
}
}

//
// Code below is from: https://github.com/fastfloat/fast_float/releases/download/v3.5.1/fast_float.h
//

// fast_float by Daniel Lemire
// fast_float by João Paulo Magalhaes
//
// with contributions from Eugene Golushkov
// with contributions from Maksim Kita
// with contributions from Marcin Wojdyr
// with contributions from Neal Richardson
// with contributions from Tim Paine
// with contributions from Fabio Pellacini
//
// Licensed under the Apache License, Version 2.0, or the
// MIT License at your option. This file may not be copied,
// modified, or distributed except according to those terms.
//
// MIT License Notice
//
//    MIT License
//    
//    Copyright (c) 2021 The fast_float authors
//    
//    Permission is hereby granted, free of charge, to any
//    person obtaining a copy of this software and associated
//    documentation files (the "Software"), to deal in the
//    Software without restriction, including without
//    limitation the rights to use, copy, modify, merge,
//    publish, distribute, sublicense, and/or sell copies of
//    the Software, and to permit persons to whom the Software
//    is furnished to do so, subject to the following
//    conditions:
//    
//    The above copyright notice and this permission notice
//    shall be included in all copies or substantial portions
//    of the Software.
//    
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
//    ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
//    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
//    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
//    SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
//    OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
//    IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//    DEALINGS IN THE SOFTWARE.
//
// Apache License (Version 2.0) Notice
//
//    Copyright 2021 The fast_float authors
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//    
//    http://www.apache.org/licenses/LICENSE-2.0
//    
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//

#ifndef FASTFLOAT_FAST_FLOAT_H
#define FASTFLOAT_FAST_FLOAT_H

namespace fast_float {
enum chars_format {
    scientific = 1<<0,
    fixed = 1<<2,
    hex = 1<<3,
    general = fixed | scientific
};


struct from_chars_result {
  const char *ptr;
  std::errc ec;
};

struct parse_options {
  constexpr explicit parse_options(chars_format fmt = chars_format::general,
                         char dot = '.')
    : format(fmt), decimal_point(dot) {}

  /** Which number formats are accepted */
  chars_format format;
  /** The character used as decimal point */
  char decimal_point;
};

/**
 * This function parses the character sequence [first,last) for a number. It parses floating-point numbers expecting
 * a locale-indepent format equivalent to what is used by std::strtod in the default ("C") locale.
 * The resulting floating-point value is the closest floating-point values (using either float or double),
 * using the "round to even" convention for values that would otherwise fall right in-between two values.
 * That is, we provide exact parsing according to the IEEE standard.
 *
 * Given a successful parse, the pointer (`ptr`) in the returned value is set to point right after the
 * parsed number, and the `value` referenced is set to the parsed value. In case of error, the returned
 * `ec` contains a representative error, otherwise the default (`std::errc()`) value is stored.
 *
 * The implementation does not throw and does not allocate memory (e.g., with `new` or `malloc`).
 *
 * Like the C++17 standard, the `fast_float::from_chars` functions take an optional last argument of
 * the type `fast_float::chars_format`. It is a bitset value: we check whether
 * `fmt & fast_float::chars_format::fixed` and `fmt & fast_float::chars_format::scientific` are set
 * to determine whether we allowe the fixed point and scientific notation respectively.
 * The default is  `fast_float::chars_format::general` which allows both `fixed` and `scientific`.
 */
template<typename T>
from_chars_result from_chars(const char *first, const char *last,
                             T &value, chars_format fmt = chars_format::general)  noexcept;

/**
 * Like from_chars, but accepts an `options` argument to govern number parsing.
 */
template<typename T>
from_chars_result from_chars_advanced(const char *first, const char *last,
                                      T &value, parse_options options)  noexcept;

}
#endif // FASTFLOAT_FAST_FLOAT_H

#ifndef FASTFLOAT_FLOAT_COMMON_H
#define FASTFLOAT_FLOAT_COMMON_H

#if (defined(__x86_64) || defined(__x86_64__) || defined(_M_X64)   \
       || defined(__amd64) || defined(__aarch64__) || defined(_M_ARM64) \
       || defined(__MINGW64__)                                          \
       || defined(__s390x__)                                            \
       || (defined(__ppc64__) || defined(__PPC64__) || defined(__ppc64le__) || defined(__PPC64LE__)) )
#define FASTFLOAT_64BIT
#elif (defined(__i386) || defined(__i386__) || defined(_M_IX86)   \
     || defined(__arm__) || defined(_M_ARM)                   \
     || defined(__MINGW32__) || defined(__EMSCRIPTEN__))
#define FASTFLOAT_32BIT
#else
  // Need to check incrementally, since SIZE_MAX is a size_t, avoid overflow.
  // We can never tell the register width, but the SIZE_MAX is a good approximation.
  // UINTPTR_MAX and INTPTR_MAX are optional, so avoid them for max portability.
  #if SIZE_MAX == 0xffff
    #error Unknown platform (16-bit, unsupported)
  #elif SIZE_MAX == 0xffffffff
    #define FASTFLOAT_32BIT
  #elif SIZE_MAX == 0xffffffffffffffff
    #define FASTFLOAT_64BIT
  #else
    #error Unknown platform (not 32-bit, not 64-bit?)
  #endif
#endif

#if ((defined(_WIN32) || defined(_WIN64)) && !defined(__clang__))
#include <intrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define FASTFLOAT_VISUAL_STUDIO 1
#endif

#if defined __BYTE_ORDER__ && defined __ORDER_BIG_ENDIAN__
#define FASTFLOAT_IS_BIG_ENDIAN (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#elif defined _WIN32
#define FASTFLOAT_IS_BIG_ENDIAN 0
#else
#if defined(__APPLE__) || defined(__FreeBSD__)
#include <machine/endian.h>
#elif defined(sun) || defined(__sun)
#include <sys/byteorder.h>
#else
#include <endian.h>
#endif
#
#ifndef __BYTE_ORDER__
// safe choice
#define FASTFLOAT_IS_BIG_ENDIAN 0
#endif
#
#ifndef __ORDER_LITTLE_ENDIAN__
// safe choice
#define FASTFLOAT_IS_BIG_ENDIAN 0
#endif
#
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define FASTFLOAT_IS_BIG_ENDIAN 0
#else
#define FASTFLOAT_IS_BIG_ENDIAN 1
#endif
#endif

#ifdef FASTFLOAT_VISUAL_STUDIO
#define fastfloat_really_inline __forceinline
#else
#define fastfloat_really_inline inline __attribute__((always_inline))
#endif

#ifndef FASTFLOAT_ASSERT
#define FASTFLOAT_ASSERT(x)  { if (!(x)) abort(); }
#endif

#ifndef FASTFLOAT_DEBUG_ASSERT
#define FASTFLOAT_DEBUG_ASSERT(x) assert(x)
#endif

// rust style `try!()` macro, or `?` operator
#define FASTFLOAT_TRY(x) { if (!(x)) return false; }

namespace fast_float {

// Compares two ASCII strings in a case insensitive manner.
inline bool fastfloat_strncasecmp(const char *input1, const char *input2,
                                  size_t length) {
  char running_diff{0};
  for (size_t i = 0; i < length; i++) {
    running_diff |= (input1[i] ^ input2[i]);
  }
  return (running_diff == 0) || (running_diff == 32);
}

#ifndef FLT_EVAL_METHOD
#error "FLT_EVAL_METHOD should be defined, please include cfloat."
#endif

// a pointer and a length to a contiguous block of memory
template <typename T>
struct span {
  const T* ptr;
  size_t length;
  span(const T* _ptr, size_t _length) : ptr(_ptr), length(_length) {}
  span() : ptr(nullptr), length(0) {}

  constexpr size_t len() const noexcept {
    return length;
  }

  const T& operator[](size_t index) const noexcept {
    FASTFLOAT_DEBUG_ASSERT(index < length);
    return ptr[index];
  }
};

struct value128 {
  uint64_t low;
  uint64_t high;
  value128(uint64_t _low, uint64_t _high) : low(_low), high(_high) {}
  value128() : low(0), high(0) {}
};

/* result might be undefined when input_num is zero */
fastfloat_really_inline int leading_zeroes(uint64_t input_num) {
  assert(input_num > 0);
#ifdef FASTFLOAT_VISUAL_STUDIO
  #if defined(_M_X64) || defined(_M_ARM64)
  unsigned long leading_zero = 0;
  // Search the mask data from most significant bit (MSB)
  // to least significant bit (LSB) for a set bit (1).
  _BitScanReverse64(&leading_zero, input_num);
  return (int)(63 - leading_zero);
  #else
  int last_bit = 0;
  if(input_num & uint64_t(0xffffffff00000000)) input_num >>= 32, last_bit |= 32;
  if(input_num & uint64_t(        0xffff0000)) input_num >>= 16, last_bit |= 16;
  if(input_num & uint64_t(            0xff00)) input_num >>=  8, last_bit |=  8;
  if(input_num & uint64_t(              0xf0)) input_num >>=  4, last_bit |=  4;
  if(input_num & uint64_t(               0xc)) input_num >>=  2, last_bit |=  2;
  if(input_num & uint64_t(               0x2)) input_num >>=  1, last_bit |=  1;
  return 63 - last_bit;
  #endif
#else
  return __builtin_clzll(input_num);
#endif
}

#ifdef FASTFLOAT_32BIT

// slow emulation routine for 32-bit
fastfloat_really_inline uint64_t emulu(uint32_t x, uint32_t y) {
    return x * (uint64_t)y;
}

// slow emulation routine for 32-bit
#if !defined(__MINGW64__)
fastfloat_really_inline uint64_t _umul128(uint64_t ab, uint64_t cd,
                                          uint64_t *hi) {
  uint64_t ad = emulu((uint32_t)(ab >> 32), (uint32_t)cd);
  uint64_t bd = emulu((uint32_t)ab, (uint32_t)cd);
  uint64_t adbc = ad + emulu((uint32_t)ab, (uint32_t)(cd >> 32));
  uint64_t adbc_carry = !!(adbc < ad);
  uint64_t lo = bd + (adbc << 32);
  *hi = emulu((uint32_t)(ab >> 32), (uint32_t)(cd >> 32)) + (adbc >> 32) +
        (adbc_carry << 32) + !!(lo < bd);
  return lo;
}
#endif // !__MINGW64__

#endif // FASTFLOAT_32BIT


// compute 64-bit a*b
fastfloat_really_inline value128 full_multiplication(uint64_t a,
                                                     uint64_t b) {
  value128 answer;
#ifdef _M_ARM64
  // ARM64 has native support for 64-bit multiplications, no need to emulate
  answer.high = __umulh(a, b);
  answer.low = a * b;
#elif defined(FASTFLOAT_32BIT) || (defined(_WIN64) && !defined(__clang__))
  answer.low = _umul128(a, b, &answer.high); // _umul128 not available on ARM64
#elif defined(FASTFLOAT_64BIT)
  __uint128_t r = ((__uint128_t)a) * b;
  answer.low = uint64_t(r);
  answer.high = uint64_t(r >> 64);
#else
  #error Not implemented
#endif
  return answer;
}

struct adjusted_mantissa {
  uint64_t mantissa{0};
  int32_t power2{0}; // a negative value indicates an invalid result
  adjusted_mantissa() = default;
  bool operator==(const adjusted_mantissa &o) const {
    return mantissa == o.mantissa && power2 == o.power2;
  }
  bool operator!=(const adjusted_mantissa &o) const {
    return mantissa != o.mantissa || power2 != o.power2;
  }
};

// Bias so we can get the real exponent with an invalid adjusted_mantissa.
constexpr static int32_t invalid_am_bias = -0x8000;

constexpr static double powers_of_ten_double[] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};
constexpr static float powers_of_ten_float[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5,
                                                1e6, 1e7, 1e8, 1e9, 1e10};

template <typename T> struct binary_format {
  using equiv_uint = typename std::conditional<sizeof(T) == 4, uint32_t, uint64_t>::type;

  static inline constexpr int mantissa_explicit_bits();
  static inline constexpr int minimum_exponent();
  static inline constexpr int infinite_power();
  static inline constexpr int sign_index();
  static inline constexpr int min_exponent_fast_path();
  static inline constexpr int max_exponent_fast_path();
  static inline constexpr int max_exponent_round_to_even();
  static inline constexpr int min_exponent_round_to_even();
  static inline constexpr uint64_t max_mantissa_fast_path();
  static inline constexpr int largest_power_of_ten();
  static inline constexpr int smallest_power_of_ten();
  static inline constexpr T exact_power_of_ten(int64_t power);
  static inline constexpr size_t max_digits();
  static inline constexpr equiv_uint exponent_mask();
  static inline constexpr equiv_uint mantissa_mask();
  static inline constexpr equiv_uint hidden_bit_mask();
};

template <> inline constexpr int binary_format<double>::mantissa_explicit_bits() {
  return 52;
}
template <> inline constexpr int binary_format<float>::mantissa_explicit_bits() {
  return 23;
}

template <> inline constexpr int binary_format<double>::max_exponent_round_to_even() {
  return 23;
}

template <> inline constexpr int binary_format<float>::max_exponent_round_to_even() {
  return 10;
}

template <> inline constexpr int binary_format<double>::min_exponent_round_to_even() {
  return -4;
}

template <> inline constexpr int binary_format<float>::min_exponent_round_to_even() {
  return -17;
}

template <> inline constexpr int binary_format<double>::minimum_exponent() {
  return -1023;
}
template <> inline constexpr int binary_format<float>::minimum_exponent() {
  return -127;
}

template <> inline constexpr int binary_format<double>::infinite_power() {
  return 0x7FF;
}
template <> inline constexpr int binary_format<float>::infinite_power() {
  return 0xFF;
}

template <> inline constexpr int binary_format<double>::sign_index() { return 63; }
template <> inline constexpr int binary_format<float>::sign_index() { return 31; }

template <> inline constexpr int binary_format<double>::min_exponent_fast_path() {
#if (FLT_EVAL_METHOD != 1) && (FLT_EVAL_METHOD != 0)
  return 0;
#else
  return -22;
#endif
}
template <> inline constexpr int binary_format<float>::min_exponent_fast_path() {
#if (FLT_EVAL_METHOD != 1) && (FLT_EVAL_METHOD != 0)
  return 0;
#else
  return -10;
#endif
}

template <> inline constexpr int binary_format<double>::max_exponent_fast_path() {
  return 22;
}
template <> inline constexpr int binary_format<float>::max_exponent_fast_path() {
  return 10;
}

template <> inline constexpr uint64_t binary_format<double>::max_mantissa_fast_path() {
  return uint64_t(2) << mantissa_explicit_bits();
}
template <> inline constexpr uint64_t binary_format<float>::max_mantissa_fast_path() {
  return uint64_t(2) << mantissa_explicit_bits();
}

template <>
inline constexpr double binary_format<double>::exact_power_of_ten(int64_t power) {
  return powers_of_ten_double[power];
}
template <>
inline constexpr float binary_format<float>::exact_power_of_ten(int64_t power) {

  return powers_of_ten_float[power];
}


template <>
inline constexpr int binary_format<double>::largest_power_of_ten() {
  return 308;
}
template <>
inline constexpr int binary_format<float>::largest_power_of_ten() {
  return 38;
}

template <>
inline constexpr int binary_format<double>::smallest_power_of_ten() {
  return -342;
}
template <>
inline constexpr int binary_format<float>::smallest_power_of_ten() {
  return -65;
}

template <> inline constexpr size_t binary_format<double>::max_digits() {
  return 769;
}
template <> inline constexpr size_t binary_format<float>::max_digits() {
  return 114;
}

template <> inline constexpr binary_format<float>::equiv_uint
    binary_format<float>::exponent_mask() {
  return 0x7F800000;
}
template <> inline constexpr binary_format<double>::equiv_uint
    binary_format<double>::exponent_mask() {
  return 0x7FF0000000000000;
}

template <> inline constexpr binary_format<float>::equiv_uint
    binary_format<float>::mantissa_mask() {
  return 0x007FFFFF;
}
template <> inline constexpr binary_format<double>::equiv_uint
    binary_format<double>::mantissa_mask() {
  return 0x000FFFFFFFFFFFFF;
}

template <> inline constexpr binary_format<float>::equiv_uint
    binary_format<float>::hidden_bit_mask() {
  return 0x00800000;
}
template <> inline constexpr binary_format<double>::equiv_uint
    binary_format<double>::hidden_bit_mask() {
  return 0x0010000000000000;
}

template<typename T>
fastfloat_really_inline void to_float(bool negative, adjusted_mantissa am, T &value) {
  uint64_t word = am.mantissa;
  word |= uint64_t(am.power2) << binary_format<T>::mantissa_explicit_bits();
  word = negative
  ? word | (uint64_t(1) << binary_format<T>::sign_index()) : word;
#if FASTFLOAT_IS_BIG_ENDIAN == 1
   if (std::is_same<T, float>::value) {
     ::memcpy(&value, (char *)&word + 4, sizeof(T)); // extract value at offset 4-7 if float on big-endian
   } else {
     ::memcpy(&value, &word, sizeof(T));
   }
#else
   // For little-endian systems:
   ::memcpy(&value, &word, sizeof(T));
#endif
}

} // namespace fast_float

#endif

#ifndef FASTFLOAT_ASCII_NUMBER_H
#define FASTFLOAT_ASCII_NUMBER_H

namespace fast_float {

// Next function can be micro-optimized, but compilers are entirely
// able to optimize it well.
fastfloat_really_inline bool is_integer(char c)  noexcept  { return c >= '0' && c <= '9'; }

fastfloat_really_inline uint64_t byteswap(uint64_t val) {
  return (val & 0xFF00000000000000) >> 56
    | (val & 0x00FF000000000000) >> 40
    | (val & 0x0000FF0000000000) >> 24
    | (val & 0x000000FF00000000) >> 8
    | (val & 0x00000000FF000000) << 8
    | (val & 0x0000000000FF0000) << 24
    | (val & 0x000000000000FF00) << 40
    | (val & 0x00000000000000FF) << 56;
}

fastfloat_really_inline uint64_t read_u64(const char *chars) {
  uint64_t val;
  ::memcpy(&val, chars, sizeof(uint64_t));
#if FASTFLOAT_IS_BIG_ENDIAN == 1
  // Need to read as-if the number was in little-endian order.
  val = byteswap(val);
#endif
  return val;
}

fastfloat_really_inline void write_u64(uint8_t *chars, uint64_t val) {
#if FASTFLOAT_IS_BIG_ENDIAN == 1
  // Need to read as-if the number was in little-endian order.
  val = byteswap(val);
#endif
  ::memcpy(chars, &val, sizeof(uint64_t));
}

// credit  @aqrit
fastfloat_really_inline uint32_t  parse_eight_digits_unrolled(uint64_t val) {
  const uint64_t mask = 0x000000FF000000FF;
  const uint64_t mul1 = 0x000F424000000064; // 100 + (1000000ULL << 32)
  const uint64_t mul2 = 0x0000271000000001; // 1 + (10000ULL << 32)
  val -= 0x3030303030303030;
  val = (val * 10) + (val >> 8); // val = (val * 2561) >> 8;
  val = (((val & mask) * mul1) + (((val >> 16) & mask) * mul2)) >> 32;
  return uint32_t(val);
}

fastfloat_really_inline uint32_t parse_eight_digits_unrolled(const char *chars)  noexcept  {
  return parse_eight_digits_unrolled(read_u64(chars));
}

// credit @aqrit
fastfloat_really_inline bool is_made_of_eight_digits_fast(uint64_t val)  noexcept  {
  return !((((val + 0x4646464646464646) | (val - 0x3030303030303030)) &
     0x8080808080808080));
}

fastfloat_really_inline bool is_made_of_eight_digits_fast(const char *chars)  noexcept  {
  return is_made_of_eight_digits_fast(read_u64(chars));
}

typedef span<const char> byte_span;

struct parsed_number_string {
  int64_t exponent{0};
  uint64_t mantissa{0};
  const char *lastmatch{nullptr};
  bool negative{false};
  bool valid{false};
  bool too_many_digits{false};
  // contains the range of the significant digits
  byte_span integer{};  // non-nullable
  byte_span fraction{}; // nullable
};

// Assuming that you use no more than 19 digits, this will
// parse an ASCII string.
fastfloat_really_inline
parsed_number_string parse_number_string(const char *p, const char *pend, parse_options options) noexcept {
  const chars_format fmt = options.format;
  const char decimal_point = options.decimal_point;

  parsed_number_string answer;
  answer.valid = false;
  answer.too_many_digits = false;
  answer.negative = (*p == '-');
  if (*p == '-') { // C++17 20.19.3.(7.1) explicitly forbids '+' sign here
    ++p;
    if (p == pend) {
      return answer;
    }
    if (!is_integer(*p) && (*p != decimal_point)) { // a sign must be followed by an integer or the dot
      return answer;
    }
  }
  const char *const start_digits = p;

  uint64_t i = 0; // an unsigned int avoids signed overflows (which are bad)

  while ((std::distance(p, pend) >= 8) && is_made_of_eight_digits_fast(p)) {
    i = i * 100000000 + parse_eight_digits_unrolled(p); // in rare cases, this will overflow, but that's ok
    p += 8;
  }
  while ((p != pend) && is_integer(*p)) {
    // a multiplication by 10 is cheaper than an arbitrary integer
    // multiplication
    i = 10 * i +
        uint64_t(*p - '0'); // might overflow, we will handle the overflow later
    ++p;
  }
  const char *const end_of_integer_part = p;
  int64_t digit_count = int64_t(end_of_integer_part - start_digits);
  answer.integer = byte_span(start_digits, size_t(digit_count));
  int64_t exponent = 0;
  if ((p != pend) && (*p == decimal_point)) {
    ++p;
    const char* before = p;
    // can occur at most twice without overflowing, but let it occur more, since
    // for integers with many digits, digit parsing is the primary bottleneck.
    while ((std::distance(p, pend) >= 8) && is_made_of_eight_digits_fast(p)) {
      i = i * 100000000 + parse_eight_digits_unrolled(p); // in rare cases, this will overflow, but that's ok
      p += 8;
    }
    while ((p != pend) && is_integer(*p)) {
      uint8_t digit = uint8_t(*p - '0');
      ++p;
      i = i * 10 + digit; // in rare cases, this will overflow, but that's ok
    }
    exponent = before - p;
    answer.fraction = byte_span(before, size_t(p - before));
    digit_count -= exponent;
  }
  // we must have encountered at least one integer!
  if (digit_count == 0) {
    return answer;
  }
  int64_t exp_number = 0;            // explicit exponential part
  if ((fmt & chars_format::scientific) && (p != pend) && (('e' == *p) || ('E' == *p))) {
    const char * location_of_e = p;
    ++p;
    bool neg_exp = false;
    if ((p != pend) && ('-' == *p)) {
      neg_exp = true;
      ++p;
    } else if ((p != pend) && ('+' == *p)) { // '+' on exponent is allowed by C++17 20.19.3.(7.1)
      ++p;
    }
    if ((p == pend) || !is_integer(*p)) {
      if(!(fmt & chars_format::fixed)) {
        // We are in error.
        return answer;
      }
      // Otherwise, we will be ignoring the 'e'.
      p = location_of_e;
    } else {
      while ((p != pend) && is_integer(*p)) {
        uint8_t digit = uint8_t(*p - '0');
        if (exp_number < 0x10000000) {
          exp_number = 10 * exp_number + digit;
        }
        ++p;
      }
      if(neg_exp) { exp_number = - exp_number; }
      exponent += exp_number;
    }
  } else {
    // If it scientific and not fixed, we have to bail out.
    if((fmt & chars_format::scientific) && !(fmt & chars_format::fixed)) { return answer; }
  }
  answer.lastmatch = p;
  answer.valid = true;

  // If we frequently had to deal with long strings of digits,
  // we could extend our code by using a 128-bit integer instead
  // of a 64-bit integer. However, this is uncommon.
  //
  // We can deal with up to 19 digits.
  if (digit_count > 19) { // this is uncommon
    // It is possible that the integer had an overflow.
    // We have to handle the case where we have 0.0000somenumber.
    // We need to be mindful of the case where we only have zeroes...
    // E.g., 0.000000000...000.
    const char *start = start_digits;
    while ((start != pend) && (*start == '0' || *start == decimal_point)) {
      if(*start == '0') { digit_count --; }
      start++;
    }
    if (digit_count > 19) {
      answer.too_many_digits = true;
      // Let us start again, this time, avoiding overflows.
      // We don't need to check if is_integer, since we use the
      // pre-tokenized spans from above.
      i = 0;
      p = answer.integer.ptr;
      const char* int_end = p + answer.integer.len();
      const uint64_t minimal_nineteen_digit_integer{1000000000000000000};
      while((i < minimal_nineteen_digit_integer) && (p != int_end)) {
        i = i * 10 + uint64_t(*p - '0');
        ++p;
      }
      if (i >= minimal_nineteen_digit_integer) { // We have a big integers
        exponent = end_of_integer_part - p + exp_number;
      } else { // We have a value with a fractional component.
          p = answer.fraction.ptr;
          const char* frac_end = p + answer.fraction.len();
          while((i < minimal_nineteen_digit_integer) && (p != frac_end)) {
            i = i * 10 + uint64_t(*p - '0');
            ++p;
          }
          exponent = answer.fraction.ptr - p + exp_number;
      }
      // We have now corrected both exponent and i, to a truncated value
    }
  }
  answer.exponent = exponent;
  answer.mantissa = i;
  return answer;
}

} // namespace fast_float

#endif

#ifndef FASTFLOAT_FAST_TABLE_H
#define FASTFLOAT_FAST_TABLE_H

namespace fast_float {

/**
 * When mapping numbers from decimal to binary,
 * we go from w * 10^q to m * 2^p but we have
 * 10^q = 5^q * 2^q, so effectively
 * we are trying to match
 * w * 2^q * 5^q to m * 2^p. Thus the powers of two
 * are not a concern since they can be represented
 * exactly using the binary notation, only the powers of five
 * affect the binary significand.
 */

/**
 * The smallest non-zero float (binary64) is 2^−1074.
 * We take as input numbers of the form w x 10^q where w < 2^64.
 * We have that w * 10^-343  <  2^(64-344) 5^-343 < 2^-1076.
 * However, we have that
 * (2^64-1) * 10^-342 =  (2^64-1) * 2^-342 * 5^-342 > 2^−1074.
 * Thus it is possible for a number of the form w * 10^-342 where
 * w is a 64-bit value to be a non-zero floating-point number.
 *********
 * Any number of form w * 10^309 where w>= 1 is going to be
 * infinite in binary64 so we never need to worry about powers
 * of 5 greater than 308.
 */
template <class unused = void>
struct powers_template {

constexpr static int smallest_power_of_five = binary_format<double>::smallest_power_of_ten();
constexpr static int largest_power_of_five = binary_format<double>::largest_power_of_ten();
constexpr static int number_of_entries = 2 * (largest_power_of_five - smallest_power_of_five + 1);
// Powers of five from 5^-342 all the way to 5^308 rounded toward one.
static const uint64_t power_of_five_128[number_of_entries];
};

template <class unused>
const uint64_t powers_template<unused>::power_of_five_128[number_of_entries] = {
        0xeef453d6923bd65a,0x113faa2906a13b3f,
        0x9558b4661b6565f8,0x4ac7ca59a424c507,
        0xbaaee17fa23ebf76,0x5d79bcf00d2df649,
        0xe95a99df8ace6f53,0xf4d82c2c107973dc,
        0x91d8a02bb6c10594,0x79071b9b8a4be869,
        0xb64ec836a47146f9,0x9748e2826cdee284,
        0xe3e27a444d8d98b7,0xfd1b1b2308169b25,
        0x8e6d8c6ab0787f72,0xfe30f0f5e50e20f7,
        0xb208ef855c969f4f,0xbdbd2d335e51a935,
        0xde8b2b66b3bc4723,0xad2c788035e61382,
        0x8b16fb203055ac76,0x4c3bcb5021afcc31,
        0xaddcb9e83c6b1793,0xdf4abe242a1bbf3d,
        0xd953e8624b85dd78,0xd71d6dad34a2af0d,
        0x87d4713d6f33aa6b,0x8672648c40e5ad68,
        0xa9c98d8ccb009506,0x680efdaf511f18c2,
        0xd43bf0effdc0ba48,0x212bd1b2566def2,
        0x84a57695fe98746d,0x14bb630f7604b57,
        0xa5ced43b7e3e9188,0x419ea3bd35385e2d,
        0xcf42894a5dce35ea,0x52064cac828675b9,
        0x818995ce7aa0e1b2,0x7343efebd1940993,
        0xa1ebfb4219491a1f,0x1014ebe6c5f90bf8,
        0xca66fa129f9b60a6,0xd41a26e077774ef6,
        0xfd00b897478238d0,0x8920b098955522b4,
        0x9e20735e8cb16382,0x55b46e5f5d5535b0,
        0xc5a890362fddbc62,0xeb2189f734aa831d,
        0xf712b443bbd52b7b,0xa5e9ec7501d523e4,
        0x9a6bb0aa55653b2d,0x47b233c92125366e,
        0xc1069cd4eabe89f8,0x999ec0bb696e840a,
        0xf148440a256e2c76,0xc00670ea43ca250d,
        0x96cd2a865764dbca,0x380406926a5e5728,
        0xbc807527ed3e12bc,0xc605083704f5ecf2,
        0xeba09271e88d976b,0xf7864a44c633682e,
        0x93445b8731587ea3,0x7ab3ee6afbe0211d,
        0xb8157268fdae9e4c,0x5960ea05bad82964,
        0xe61acf033d1a45df,0x6fb92487298e33bd,
        0x8fd0c16206306bab,0xa5d3b6d479f8e056,
        0xb3c4f1ba87bc8696,0x8f48a4899877186c,
        0xe0b62e2929aba83c,0x331acdabfe94de87,
        0x8c71dcd9ba0b4925,0x9ff0c08b7f1d0b14,
        0xaf8e5410288e1b6f,0x7ecf0ae5ee44dd9,
        0xdb71e91432b1a24a,0xc9e82cd9f69d6150,
        0x892731ac9faf056e,0xbe311c083a225cd2,
        0xab70fe17c79ac6ca,0x6dbd630a48aaf406,
        0xd64d3d9db981787d,0x92cbbccdad5b108,
        0x85f0468293f0eb4e,0x25bbf56008c58ea5,
        0xa76c582338ed2621,0xaf2af2b80af6f24e,
        0xd1476e2c07286faa,0x1af5af660db4aee1,
        0x82cca4db847945ca,0x50d98d9fc890ed4d,
        0xa37fce126597973c,0xe50ff107bab528a0,
        0xcc5fc196fefd7d0c,0x1e53ed49a96272c8,
        0xff77b1fcbebcdc4f,0x25e8e89c13bb0f7a,
        0x9faacf3df73609b1,0x77b191618c54e9ac,
        0xc795830d75038c1d,0xd59df5b9ef6a2417,
        0xf97ae3d0d2446f25,0x4b0573286b44ad1d,
        0x9becce62836ac577,0x4ee367f9430aec32,
        0xc2e801fb244576d5,0x229c41f793cda73f,
        0xf3a20279ed56d48a,0x6b43527578c1110f,
        0x9845418c345644d6,0x830a13896b78aaa9,
        0xbe5691ef416bd60c,0x23cc986bc656d553,
        0xedec366b11c6cb8f,0x2cbfbe86b7ec8aa8,
        0x94b3a202eb1c3f39,0x7bf7d71432f3d6a9,
        0xb9e08a83a5e34f07,0xdaf5ccd93fb0cc53,
        0xe858ad248f5c22c9,0xd1b3400f8f9cff68,
        0x91376c36d99995be,0x23100809b9c21fa1,
        0xb58547448ffffb2d,0xabd40a0c2832a78a,
        0xe2e69915b3fff9f9,0x16c90c8f323f516c,
        0x8dd01fad907ffc3b,0xae3da7d97f6792e3,
        0xb1442798f49ffb4a,0x99cd11cfdf41779c,
        0xdd95317f31c7fa1d,0x40405643d711d583,
        0x8a7d3eef7f1cfc52,0x482835ea666b2572,
        0xad1c8eab5ee43b66,0xda3243650005eecf,
        0xd863b256369d4a40,0x90bed43e40076a82,
        0x873e4f75e2224e68,0x5a7744a6e804a291,
        0xa90de3535aaae202,0x711515d0a205cb36,
        0xd3515c2831559a83,0xd5a5b44ca873e03,
        0x8412d9991ed58091,0xe858790afe9486c2,
        0xa5178fff668ae0b6,0x626e974dbe39a872,
        0xce5d73ff402d98e3,0xfb0a3d212dc8128f,
        0x80fa687f881c7f8e,0x7ce66634bc9d0b99,
        0xa139029f6a239f72,0x1c1fffc1ebc44e80,
        0xc987434744ac874e,0xa327ffb266b56220,
        0xfbe9141915d7a922,0x4bf1ff9f0062baa8,
        0x9d71ac8fada6c9b5,0x6f773fc3603db4a9,
        0xc4ce17b399107c22,0xcb550fb4384d21d3,
        0xf6019da07f549b2b,0x7e2a53a146606a48,
        0x99c102844f94e0fb,0x2eda7444cbfc426d,
        0xc0314325637a1939,0xfa911155fefb5308,
        0xf03d93eebc589f88,0x793555ab7eba27ca,
        0x96267c7535b763b5,0x4bc1558b2f3458de,
        0xbbb01b9283253ca2,0x9eb1aaedfb016f16,
        0xea9c227723ee8bcb,0x465e15a979c1cadc,
        0x92a1958a7675175f,0xbfacd89ec191ec9,
        0xb749faed14125d36,0xcef980ec671f667b,
        0xe51c79a85916f484,0x82b7e12780e7401a,
        0x8f31cc0937ae58d2,0xd1b2ecb8b0908810,
        0xb2fe3f0b8599ef07,0x861fa7e6dcb4aa15,
        0xdfbdcece67006ac9,0x67a791e093e1d49a,
        0x8bd6a141006042bd,0xe0c8bb2c5c6d24e0,
        0xaecc49914078536d,0x58fae9f773886e18,
        0xda7f5bf590966848,0xaf39a475506a899e,
        0x888f99797a5e012d,0x6d8406c952429603,
        0xaab37fd7d8f58178,0xc8e5087ba6d33b83,
        0xd5605fcdcf32e1d6,0xfb1e4a9a90880a64,
        0x855c3be0a17fcd26,0x5cf2eea09a55067f,
        0xa6b34ad8c9dfc06f,0xf42faa48c0ea481e,
        0xd0601d8efc57b08b,0xf13b94daf124da26,
        0x823c12795db6ce57,0x76c53d08d6b70858,
        0xa2cb1717b52481ed,0x54768c4b0c64ca6e,
        0xcb7ddcdda26da268,0xa9942f5dcf7dfd09,
        0xfe5d54150b090b02,0xd3f93b35435d7c4c,
        0x9efa548d26e5a6e1,0xc47bc5014a1a6daf,
        0xc6b8e9b0709f109a,0x359ab6419ca1091b,
        0xf867241c8cc6d4c0,0xc30163d203c94b62,
        0x9b407691d7fc44f8,0x79e0de63425dcf1d,
        0xc21094364dfb5636,0x985915fc12f542e4,
        0xf294b943e17a2bc4,0x3e6f5b7b17b2939d,
        0x979cf3ca6cec5b5a,0xa705992ceecf9c42,
        0xbd8430bd08277231,0x50c6ff782a838353,
        0xece53cec4a314ebd,0xa4f8bf5635246428,
        0x940f4613ae5ed136,0x871b7795e136be99,
        0xb913179899f68584,0x28e2557b59846e3f,
        0xe757dd7ec07426e5,0x331aeada2fe589cf,
        0x9096ea6f3848984f,0x3ff0d2c85def7621,
        0xb4bca50b065abe63,0xfed077a756b53a9,
        0xe1ebce4dc7f16dfb,0xd3e8495912c62894,
        0x8d3360f09cf6e4bd,0x64712dd7abbbd95c,
        0xb080392cc4349dec,0xbd8d794d96aacfb3,
        0xdca04777f541c567,0xecf0d7a0fc5583a0,
        0x89e42caaf9491b60,0xf41686c49db57244,
        0xac5d37d5b79b6239,0x311c2875c522ced5,
        0xd77485cb25823ac7,0x7d633293366b828b,
        0x86a8d39ef77164bc,0xae5dff9c02033197,
        0xa8530886b54dbdeb,0xd9f57f830283fdfc,
        0xd267caa862a12d66,0xd072df63c324fd7b,
        0x8380dea93da4bc60,0x4247cb9e59f71e6d,
        0xa46116538d0deb78,0x52d9be85f074e608,
        0xcd795be870516656,0x67902e276c921f8b,
        0x806bd9714632dff6,0xba1cd8a3db53b6,
        0xa086cfcd97bf97f3,0x80e8a40eccd228a4,
        0xc8a883c0fdaf7df0,0x6122cd128006b2cd,
        0xfad2a4b13d1b5d6c,0x796b805720085f81,
        0x9cc3a6eec6311a63,0xcbe3303674053bb0,
        0xc3f490aa77bd60fc,0xbedbfc4411068a9c,
        0xf4f1b4d515acb93b,0xee92fb5515482d44,
        0x991711052d8bf3c5,0x751bdd152d4d1c4a,
        0xbf5cd54678eef0b6,0xd262d45a78a0635d,
        0xef340a98172aace4,0x86fb897116c87c34,
        0x9580869f0e7aac0e,0xd45d35e6ae3d4da0,
        0xbae0a846d2195712,0x8974836059cca109,
        0xe998d258869facd7,0x2bd1a438703fc94b,
        0x91ff83775423cc06,0x7b6306a34627ddcf,
        0xb67f6455292cbf08,0x1a3bc84c17b1d542,
        0xe41f3d6a7377eeca,0x20caba5f1d9e4a93,
        0x8e938662882af53e,0x547eb47b7282ee9c,
        0xb23867fb2a35b28d,0xe99e619a4f23aa43,
        0xdec681f9f4c31f31,0x6405fa00e2ec94d4,
        0x8b3c113c38f9f37e,0xde83bc408dd3dd04,
        0xae0b158b4738705e,0x9624ab50b148d445,
        0xd98ddaee19068c76,0x3badd624dd9b0957,
        0x87f8a8d4cfa417c9,0xe54ca5d70a80e5d6,
        0xa9f6d30a038d1dbc,0x5e9fcf4ccd211f4c,
        0xd47487cc8470652b,0x7647c3200069671f,
        0x84c8d4dfd2c63f3b,0x29ecd9f40041e073,
        0xa5fb0a17c777cf09,0xf468107100525890,
        0xcf79cc9db955c2cc,0x7182148d4066eeb4,
        0x81ac1fe293d599bf,0xc6f14cd848405530,
        0xa21727db38cb002f,0xb8ada00e5a506a7c,
        0xca9cf1d206fdc03b,0xa6d90811f0e4851c,
        0xfd442e4688bd304a,0x908f4a166d1da663,
        0x9e4a9cec15763e2e,0x9a598e4e043287fe,
        0xc5dd44271ad3cdba,0x40eff1e1853f29fd,
        0xf7549530e188c128,0xd12bee59e68ef47c,
        0x9a94dd3e8cf578b9,0x82bb74f8301958ce,
        0xc13a148e3032d6e7,0xe36a52363c1faf01,
        0xf18899b1bc3f8ca1,0xdc44e6c3cb279ac1,
        0x96f5600f15a7b7e5,0x29ab103a5ef8c0b9,
        0xbcb2b812db11a5de,0x7415d448f6b6f0e7,
        0xebdf661791d60f56,0x111b495b3464ad21,
        0x936b9fcebb25c995,0xcab10dd900beec34,
        0xb84687c269ef3bfb,0x3d5d514f40eea742,
        0xe65829b3046b0afa,0xcb4a5a3112a5112,
        0x8ff71a0fe2c2e6dc,0x47f0e785eaba72ab,
        0xb3f4e093db73a093,0x59ed216765690f56,
        0xe0f218b8d25088b8,0x306869c13ec3532c,
        0x8c974f7383725573,0x1e414218c73a13fb,
        0xafbd2350644eeacf,0xe5d1929ef90898fa,
        0xdbac6c247d62a583,0xdf45f746b74abf39,
        0x894bc396ce5da772,0x6b8bba8c328eb783,
        0xab9eb47c81f5114f,0x66ea92f3f326564,
        0xd686619ba27255a2,0xc80a537b0efefebd,
        0x8613fd0145877585,0xbd06742ce95f5f36,
        0xa798fc4196e952e7,0x2c48113823b73704,
        0xd17f3b51fca3a7a0,0xf75a15862ca504c5,
        0x82ef85133de648c4,0x9a984d73dbe722fb,
        0xa3ab66580d5fdaf5,0xc13e60d0d2e0ebba,
        0xcc963fee10b7d1b3,0x318df905079926a8,
        0xffbbcfe994e5c61f,0xfdf17746497f7052,
        0x9fd561f1fd0f9bd3,0xfeb6ea8bedefa633,
        0xc7caba6e7c5382c8,0xfe64a52ee96b8fc0,
        0xf9bd690a1b68637b,0x3dfdce7aa3c673b0,
        0x9c1661a651213e2d,0x6bea10ca65c084e,
        0xc31bfa0fe5698db8,0x486e494fcff30a62,
        0xf3e2f893dec3f126,0x5a89dba3c3efccfa,
        0x986ddb5c6b3a76b7,0xf89629465a75e01c,
        0xbe89523386091465,0xf6bbb397f1135823,
        0xee2ba6c0678b597f,0x746aa07ded582e2c,
        0x94db483840b717ef,0xa8c2a44eb4571cdc,
        0xba121a4650e4ddeb,0x92f34d62616ce413,
        0xe896a0d7e51e1566,0x77b020baf9c81d17,
        0x915e2486ef32cd60,0xace1474dc1d122e,
        0xb5b5ada8aaff80b8,0xd819992132456ba,
        0xe3231912d5bf60e6,0x10e1fff697ed6c69,
        0x8df5efabc5979c8f,0xca8d3ffa1ef463c1,
        0xb1736b96b6fd83b3,0xbd308ff8a6b17cb2,
        0xddd0467c64bce4a0,0xac7cb3f6d05ddbde,
        0x8aa22c0dbef60ee4,0x6bcdf07a423aa96b,
        0xad4ab7112eb3929d,0x86c16c98d2c953c6,
        0xd89d64d57a607744,0xe871c7bf077ba8b7,
        0x87625f056c7c4a8b,0x11471cd764ad4972,
        0xa93af6c6c79b5d2d,0xd598e40d3dd89bcf,
        0xd389b47879823479,0x4aff1d108d4ec2c3,
        0x843610cb4bf160cb,0xcedf722a585139ba,
        0xa54394fe1eedb8fe,0xc2974eb4ee658828,
        0xce947a3da6a9273e,0x733d226229feea32,
        0x811ccc668829b887,0x806357d5a3f525f,
        0xa163ff802a3426a8,0xca07c2dcb0cf26f7,
        0xc9bcff6034c13052,0xfc89b393dd02f0b5,
        0xfc2c3f3841f17c67,0xbbac2078d443ace2,
        0x9d9ba7832936edc0,0xd54b944b84aa4c0d,
        0xc5029163f384a931,0xa9e795e65d4df11,
        0xf64335bcf065d37d,0x4d4617b5ff4a16d5,
        0x99ea0196163fa42e,0x504bced1bf8e4e45,
        0xc06481fb9bcf8d39,0xe45ec2862f71e1d6,
        0xf07da27a82c37088,0x5d767327bb4e5a4c,
        0x964e858c91ba2655,0x3a6a07f8d510f86f,
        0xbbe226efb628afea,0x890489f70a55368b,
        0xeadab0aba3b2dbe5,0x2b45ac74ccea842e,
        0x92c8ae6b464fc96f,0x3b0b8bc90012929d,
        0xb77ada0617e3bbcb,0x9ce6ebb40173744,
        0xe55990879ddcaabd,0xcc420a6a101d0515,
        0x8f57fa54c2a9eab6,0x9fa946824a12232d,
        0xb32df8e9f3546564,0x47939822dc96abf9,
        0xdff9772470297ebd,0x59787e2b93bc56f7,
        0x8bfbea76c619ef36,0x57eb4edb3c55b65a,
        0xaefae51477a06b03,0xede622920b6b23f1,
        0xdab99e59958885c4,0xe95fab368e45eced,
        0x88b402f7fd75539b,0x11dbcb0218ebb414,
        0xaae103b5fcd2a881,0xd652bdc29f26a119,
        0xd59944a37c0752a2,0x4be76d3346f0495f,
        0x857fcae62d8493a5,0x6f70a4400c562ddb,
        0xa6dfbd9fb8e5b88e,0xcb4ccd500f6bb952,
        0xd097ad07a71f26b2,0x7e2000a41346a7a7,
        0x825ecc24c873782f,0x8ed400668c0c28c8,
        0xa2f67f2dfa90563b,0x728900802f0f32fa,
        0xcbb41ef979346bca,0x4f2b40a03ad2ffb9,
        0xfea126b7d78186bc,0xe2f610c84987bfa8,
        0x9f24b832e6b0f436,0xdd9ca7d2df4d7c9,
        0xc6ede63fa05d3143,0x91503d1c79720dbb,
        0xf8a95fcf88747d94,0x75a44c6397ce912a,
        0x9b69dbe1b548ce7c,0xc986afbe3ee11aba,
        0xc24452da229b021b,0xfbe85badce996168,
        0xf2d56790ab41c2a2,0xfae27299423fb9c3,
        0x97c560ba6b0919a5,0xdccd879fc967d41a,
        0xbdb6b8e905cb600f,0x5400e987bbc1c920,
        0xed246723473e3813,0x290123e9aab23b68,
        0x9436c0760c86e30b,0xf9a0b6720aaf6521,
        0xb94470938fa89bce,0xf808e40e8d5b3e69,
        0xe7958cb87392c2c2,0xb60b1d1230b20e04,
        0x90bd77f3483bb9b9,0xb1c6f22b5e6f48c2,
        0xb4ecd5f01a4aa828,0x1e38aeb6360b1af3,
        0xe2280b6c20dd5232,0x25c6da63c38de1b0,
        0x8d590723948a535f,0x579c487e5a38ad0e,
        0xb0af48ec79ace837,0x2d835a9df0c6d851,
        0xdcdb1b2798182244,0xf8e431456cf88e65,
        0x8a08f0f8bf0f156b,0x1b8e9ecb641b58ff,
        0xac8b2d36eed2dac5,0xe272467e3d222f3f,
        0xd7adf884aa879177,0x5b0ed81dcc6abb0f,
        0x86ccbb52ea94baea,0x98e947129fc2b4e9,
        0xa87fea27a539e9a5,0x3f2398d747b36224,
        0xd29fe4b18e88640e,0x8eec7f0d19a03aad,
        0x83a3eeeef9153e89,0x1953cf68300424ac,
        0xa48ceaaab75a8e2b,0x5fa8c3423c052dd7,
        0xcdb02555653131b6,0x3792f412cb06794d,
        0x808e17555f3ebf11,0xe2bbd88bbee40bd0,
        0xa0b19d2ab70e6ed6,0x5b6aceaeae9d0ec4,
        0xc8de047564d20a8b,0xf245825a5a445275,
        0xfb158592be068d2e,0xeed6e2f0f0d56712,
        0x9ced737bb6c4183d,0x55464dd69685606b,
        0xc428d05aa4751e4c,0xaa97e14c3c26b886,
        0xf53304714d9265df,0xd53dd99f4b3066a8,
        0x993fe2c6d07b7fab,0xe546a8038efe4029,
        0xbf8fdb78849a5f96,0xde98520472bdd033,
        0xef73d256a5c0f77c,0x963e66858f6d4440,
        0x95a8637627989aad,0xdde7001379a44aa8,
        0xbb127c53b17ec159,0x5560c018580d5d52,
        0xe9d71b689dde71af,0xaab8f01e6e10b4a6,
        0x9226712162ab070d,0xcab3961304ca70e8,
        0xb6b00d69bb55c8d1,0x3d607b97c5fd0d22,
        0xe45c10c42a2b3b05,0x8cb89a7db77c506a,
        0x8eb98a7a9a5b04e3,0x77f3608e92adb242,
        0xb267ed1940f1c61c,0x55f038b237591ed3,
        0xdf01e85f912e37a3,0x6b6c46dec52f6688,
        0x8b61313bbabce2c6,0x2323ac4b3b3da015,
        0xae397d8aa96c1b77,0xabec975e0a0d081a,
        0xd9c7dced53c72255,0x96e7bd358c904a21,
        0x881cea14545c7575,0x7e50d64177da2e54,
        0xaa242499697392d2,0xdde50bd1d5d0b9e9,
        0xd4ad2dbfc3d07787,0x955e4ec64b44e864,
        0x84ec3c97da624ab4,0xbd5af13bef0b113e,
        0xa6274bbdd0fadd61,0xecb1ad8aeacdd58e,
        0xcfb11ead453994ba,0x67de18eda5814af2,
        0x81ceb32c4b43fcf4,0x80eacf948770ced7,
        0xa2425ff75e14fc31,0xa1258379a94d028d,
        0xcad2f7f5359a3b3e,0x96ee45813a04330,
        0xfd87b5f28300ca0d,0x8bca9d6e188853fc,
        0x9e74d1b791e07e48,0x775ea264cf55347e,
        0xc612062576589dda,0x95364afe032a819e,
        0xf79687aed3eec551,0x3a83ddbd83f52205,
        0x9abe14cd44753b52,0xc4926a9672793543,
        0xc16d9a0095928a27,0x75b7053c0f178294,
        0xf1c90080baf72cb1,0x5324c68b12dd6339,
        0x971da05074da7bee,0xd3f6fc16ebca5e04,
        0xbce5086492111aea,0x88f4bb1ca6bcf585,
        0xec1e4a7db69561a5,0x2b31e9e3d06c32e6,
        0x9392ee8e921d5d07,0x3aff322e62439fd0,
        0xb877aa3236a4b449,0x9befeb9fad487c3,
        0xe69594bec44de15b,0x4c2ebe687989a9b4,
        0x901d7cf73ab0acd9,0xf9d37014bf60a11,
        0xb424dc35095cd80f,0x538484c19ef38c95,
        0xe12e13424bb40e13,0x2865a5f206b06fba,
        0x8cbccc096f5088cb,0xf93f87b7442e45d4,
        0xafebff0bcb24aafe,0xf78f69a51539d749,
        0xdbe6fecebdedd5be,0xb573440e5a884d1c,
        0x89705f4136b4a597,0x31680a88f8953031,
        0xabcc77118461cefc,0xfdc20d2b36ba7c3e,
        0xd6bf94d5e57a42bc,0x3d32907604691b4d,
        0x8637bd05af6c69b5,0xa63f9a49c2c1b110,
        0xa7c5ac471b478423,0xfcf80dc33721d54,
        0xd1b71758e219652b,0xd3c36113404ea4a9,
        0x83126e978d4fdf3b,0x645a1cac083126ea,
        0xa3d70a3d70a3d70a,0x3d70a3d70a3d70a4,
        0xcccccccccccccccc,0xcccccccccccccccd,
        0x8000000000000000,0x0,
        0xa000000000000000,0x0,
        0xc800000000000000,0x0,
        0xfa00000000000000,0x0,
        0x9c40000000000000,0x0,
        0xc350000000000000,0x0,
        0xf424000000000000,0x0,
        0x9896800000000000,0x0,
        0xbebc200000000000,0x0,
        0xee6b280000000000,0x0,
        0x9502f90000000000,0x0,
        0xba43b74000000000,0x0,
        0xe8d4a51000000000,0x0,
        0x9184e72a00000000,0x0,
        0xb5e620f480000000,0x0,
        0xe35fa931a0000000,0x0,
        0x8e1bc9bf04000000,0x0,
        0xb1a2bc2ec5000000,0x0,
        0xde0b6b3a76400000,0x0,
        0x8ac7230489e80000,0x0,
        0xad78ebc5ac620000,0x0,
        0xd8d726b7177a8000,0x0,
        0x878678326eac9000,0x0,
        0xa968163f0a57b400,0x0,
        0xd3c21bcecceda100,0x0,
        0x84595161401484a0,0x0,
        0xa56fa5b99019a5c8,0x0,
        0xcecb8f27f4200f3a,0x0,
        0x813f3978f8940984,0x4000000000000000,
        0xa18f07d736b90be5,0x5000000000000000,
        0xc9f2c9cd04674ede,0xa400000000000000,
        0xfc6f7c4045812296,0x4d00000000000000,
        0x9dc5ada82b70b59d,0xf020000000000000,
        0xc5371912364ce305,0x6c28000000000000,
        0xf684df56c3e01bc6,0xc732000000000000,
        0x9a130b963a6c115c,0x3c7f400000000000,
        0xc097ce7bc90715b3,0x4b9f100000000000,
        0xf0bdc21abb48db20,0x1e86d40000000000,
        0x96769950b50d88f4,0x1314448000000000,
        0xbc143fa4e250eb31,0x17d955a000000000,
        0xeb194f8e1ae525fd,0x5dcfab0800000000,
        0x92efd1b8d0cf37be,0x5aa1cae500000000,
        0xb7abc627050305ad,0xf14a3d9e40000000,
        0xe596b7b0c643c719,0x6d9ccd05d0000000,
        0x8f7e32ce7bea5c6f,0xe4820023a2000000,
        0xb35dbf821ae4f38b,0xdda2802c8a800000,
        0xe0352f62a19e306e,0xd50b2037ad200000,
        0x8c213d9da502de45,0x4526f422cc340000,
        0xaf298d050e4395d6,0x9670b12b7f410000,
        0xdaf3f04651d47b4c,0x3c0cdd765f114000,
        0x88d8762bf324cd0f,0xa5880a69fb6ac800,
        0xab0e93b6efee0053,0x8eea0d047a457a00,
        0xd5d238a4abe98068,0x72a4904598d6d880,
        0x85a36366eb71f041,0x47a6da2b7f864750,
        0xa70c3c40a64e6c51,0x999090b65f67d924,
        0xd0cf4b50cfe20765,0xfff4b4e3f741cf6d,
        0x82818f1281ed449f,0xbff8f10e7a8921a4,
        0xa321f2d7226895c7,0xaff72d52192b6a0d,
        0xcbea6f8ceb02bb39,0x9bf4f8a69f764490,
        0xfee50b7025c36a08,0x2f236d04753d5b4,
        0x9f4f2726179a2245,0x1d762422c946590,
        0xc722f0ef9d80aad6,0x424d3ad2b7b97ef5,
        0xf8ebad2b84e0d58b,0xd2e0898765a7deb2,
        0x9b934c3b330c8577,0x63cc55f49f88eb2f,
        0xc2781f49ffcfa6d5,0x3cbf6b71c76b25fb,
        0xf316271c7fc3908a,0x8bef464e3945ef7a,
        0x97edd871cfda3a56,0x97758bf0e3cbb5ac,
        0xbde94e8e43d0c8ec,0x3d52eeed1cbea317,
        0xed63a231d4c4fb27,0x4ca7aaa863ee4bdd,
        0x945e455f24fb1cf8,0x8fe8caa93e74ef6a,
        0xb975d6b6ee39e436,0xb3e2fd538e122b44,
        0xe7d34c64a9c85d44,0x60dbbca87196b616,
        0x90e40fbeea1d3a4a,0xbc8955e946fe31cd,
        0xb51d13aea4a488dd,0x6babab6398bdbe41,
        0xe264589a4dcdab14,0xc696963c7eed2dd1,
        0x8d7eb76070a08aec,0xfc1e1de5cf543ca2,
        0xb0de65388cc8ada8,0x3b25a55f43294bcb,
        0xdd15fe86affad912,0x49ef0eb713f39ebe,
        0x8a2dbf142dfcc7ab,0x6e3569326c784337,
        0xacb92ed9397bf996,0x49c2c37f07965404,
        0xd7e77a8f87daf7fb,0xdc33745ec97be906,
        0x86f0ac99b4e8dafd,0x69a028bb3ded71a3,
        0xa8acd7c0222311bc,0xc40832ea0d68ce0c,
        0xd2d80db02aabd62b,0xf50a3fa490c30190,
        0x83c7088e1aab65db,0x792667c6da79e0fa,
        0xa4b8cab1a1563f52,0x577001b891185938,
        0xcde6fd5e09abcf26,0xed4c0226b55e6f86,
        0x80b05e5ac60b6178,0x544f8158315b05b4,
        0xa0dc75f1778e39d6,0x696361ae3db1c721,
        0xc913936dd571c84c,0x3bc3a19cd1e38e9,
        0xfb5878494ace3a5f,0x4ab48a04065c723,
        0x9d174b2dcec0e47b,0x62eb0d64283f9c76,
        0xc45d1df942711d9a,0x3ba5d0bd324f8394,
        0xf5746577930d6500,0xca8f44ec7ee36479,
        0x9968bf6abbe85f20,0x7e998b13cf4e1ecb,
        0xbfc2ef456ae276e8,0x9e3fedd8c321a67e,
        0xefb3ab16c59b14a2,0xc5cfe94ef3ea101e,
        0x95d04aee3b80ece5,0xbba1f1d158724a12,
        0xbb445da9ca61281f,0x2a8a6e45ae8edc97,
        0xea1575143cf97226,0xf52d09d71a3293bd,
        0x924d692ca61be758,0x593c2626705f9c56,
        0xb6e0c377cfa2e12e,0x6f8b2fb00c77836c,
        0xe498f455c38b997a,0xb6dfb9c0f956447,
        0x8edf98b59a373fec,0x4724bd4189bd5eac,
        0xb2977ee300c50fe7,0x58edec91ec2cb657,
        0xdf3d5e9bc0f653e1,0x2f2967b66737e3ed,
        0x8b865b215899f46c,0xbd79e0d20082ee74,
        0xae67f1e9aec07187,0xecd8590680a3aa11,
        0xda01ee641a708de9,0xe80e6f4820cc9495,
        0x884134fe908658b2,0x3109058d147fdcdd,
        0xaa51823e34a7eede,0xbd4b46f0599fd415,
        0xd4e5e2cdc1d1ea96,0x6c9e18ac7007c91a,
        0x850fadc09923329e,0x3e2cf6bc604ddb0,
        0xa6539930bf6bff45,0x84db8346b786151c,
        0xcfe87f7cef46ff16,0xe612641865679a63,
        0x81f14fae158c5f6e,0x4fcb7e8f3f60c07e,
        0xa26da3999aef7749,0xe3be5e330f38f09d,
        0xcb090c8001ab551c,0x5cadf5bfd3072cc5,
        0xfdcb4fa002162a63,0x73d9732fc7c8f7f6,
        0x9e9f11c4014dda7e,0x2867e7fddcdd9afa,
        0xc646d63501a1511d,0xb281e1fd541501b8,
        0xf7d88bc24209a565,0x1f225a7ca91a4226,
        0x9ae757596946075f,0x3375788de9b06958,
        0xc1a12d2fc3978937,0x52d6b1641c83ae,
        0xf209787bb47d6b84,0xc0678c5dbd23a49a,
        0x9745eb4d50ce6332,0xf840b7ba963646e0,
        0xbd176620a501fbff,0xb650e5a93bc3d898,
        0xec5d3fa8ce427aff,0xa3e51f138ab4cebe,
        0x93ba47c980e98cdf,0xc66f336c36b10137,
        0xb8a8d9bbe123f017,0xb80b0047445d4184,
        0xe6d3102ad96cec1d,0xa60dc059157491e5,
        0x9043ea1ac7e41392,0x87c89837ad68db2f,
        0xb454e4a179dd1877,0x29babe4598c311fb,
        0xe16a1dc9d8545e94,0xf4296dd6fef3d67a,
        0x8ce2529e2734bb1d,0x1899e4a65f58660c,
        0xb01ae745b101e9e4,0x5ec05dcff72e7f8f,
        0xdc21a1171d42645d,0x76707543f4fa1f73,
        0x899504ae72497eba,0x6a06494a791c53a8,
        0xabfa45da0edbde69,0x487db9d17636892,
        0xd6f8d7509292d603,0x45a9d2845d3c42b6,
        0x865b86925b9bc5c2,0xb8a2392ba45a9b2,
        0xa7f26836f282b732,0x8e6cac7768d7141e,
        0xd1ef0244af2364ff,0x3207d795430cd926,
        0x8335616aed761f1f,0x7f44e6bd49e807b8,
        0xa402b9c5a8d3a6e7,0x5f16206c9c6209a6,
        0xcd036837130890a1,0x36dba887c37a8c0f,
        0x802221226be55a64,0xc2494954da2c9789,
        0xa02aa96b06deb0fd,0xf2db9baa10b7bd6c,
        0xc83553c5c8965d3d,0x6f92829494e5acc7,
        0xfa42a8b73abbf48c,0xcb772339ba1f17f9,
        0x9c69a97284b578d7,0xff2a760414536efb,
        0xc38413cf25e2d70d,0xfef5138519684aba,
        0xf46518c2ef5b8cd1,0x7eb258665fc25d69,
        0x98bf2f79d5993802,0xef2f773ffbd97a61,
        0xbeeefb584aff8603,0xaafb550ffacfd8fa,
        0xeeaaba2e5dbf6784,0x95ba2a53f983cf38,
        0x952ab45cfa97a0b2,0xdd945a747bf26183,
        0xba756174393d88df,0x94f971119aeef9e4,
        0xe912b9d1478ceb17,0x7a37cd5601aab85d,
        0x91abb422ccb812ee,0xac62e055c10ab33a,
        0xb616a12b7fe617aa,0x577b986b314d6009,
        0xe39c49765fdf9d94,0xed5a7e85fda0b80b,
        0x8e41ade9fbebc27d,0x14588f13be847307,
        0xb1d219647ae6b31c,0x596eb2d8ae258fc8,
        0xde469fbd99a05fe3,0x6fca5f8ed9aef3bb,
        0x8aec23d680043bee,0x25de7bb9480d5854,
        0xada72ccc20054ae9,0xaf561aa79a10ae6a,
        0xd910f7ff28069da4,0x1b2ba1518094da04,
        0x87aa9aff79042286,0x90fb44d2f05d0842,
        0xa99541bf57452b28,0x353a1607ac744a53,
        0xd3fa922f2d1675f2,0x42889b8997915ce8,
        0x847c9b5d7c2e09b7,0x69956135febada11,
        0xa59bc234db398c25,0x43fab9837e699095,
        0xcf02b2c21207ef2e,0x94f967e45e03f4bb,
        0x8161afb94b44f57d,0x1d1be0eebac278f5,
        0xa1ba1ba79e1632dc,0x6462d92a69731732,
        0xca28a291859bbf93,0x7d7b8f7503cfdcfe,
        0xfcb2cb35e702af78,0x5cda735244c3d43e,
        0x9defbf01b061adab,0x3a0888136afa64a7,
        0xc56baec21c7a1916,0x88aaa1845b8fdd0,
        0xf6c69a72a3989f5b,0x8aad549e57273d45,
        0x9a3c2087a63f6399,0x36ac54e2f678864b,
        0xc0cb28a98fcf3c7f,0x84576a1bb416a7dd,
        0xf0fdf2d3f3c30b9f,0x656d44a2a11c51d5,
        0x969eb7c47859e743,0x9f644ae5a4b1b325,
        0xbc4665b596706114,0x873d5d9f0dde1fee,
        0xeb57ff22fc0c7959,0xa90cb506d155a7ea,
        0x9316ff75dd87cbd8,0x9a7f12442d588f2,
        0xb7dcbf5354e9bece,0xc11ed6d538aeb2f,
        0xe5d3ef282a242e81,0x8f1668c8a86da5fa,
        0x8fa475791a569d10,0xf96e017d694487bc,
        0xb38d92d760ec4455,0x37c981dcc395a9ac,
        0xe070f78d3927556a,0x85bbe253f47b1417,
        0x8c469ab843b89562,0x93956d7478ccec8e,
        0xaf58416654a6babb,0x387ac8d1970027b2,
        0xdb2e51bfe9d0696a,0x6997b05fcc0319e,
        0x88fcf317f22241e2,0x441fece3bdf81f03,
        0xab3c2fddeeaad25a,0xd527e81cad7626c3,
        0xd60b3bd56a5586f1,0x8a71e223d8d3b074,
        0x85c7056562757456,0xf6872d5667844e49,
        0xa738c6bebb12d16c,0xb428f8ac016561db,
        0xd106f86e69d785c7,0xe13336d701beba52,
        0x82a45b450226b39c,0xecc0024661173473,
        0xa34d721642b06084,0x27f002d7f95d0190,
        0xcc20ce9bd35c78a5,0x31ec038df7b441f4,
        0xff290242c83396ce,0x7e67047175a15271,
        0x9f79a169bd203e41,0xf0062c6e984d386,
        0xc75809c42c684dd1,0x52c07b78a3e60868,
        0xf92e0c3537826145,0xa7709a56ccdf8a82,
        0x9bbcc7a142b17ccb,0x88a66076400bb691,
        0xc2abf989935ddbfe,0x6acff893d00ea435,
        0xf356f7ebf83552fe,0x583f6b8c4124d43,
        0x98165af37b2153de,0xc3727a337a8b704a,
        0xbe1bf1b059e9a8d6,0x744f18c0592e4c5c,
        0xeda2ee1c7064130c,0x1162def06f79df73,
        0x9485d4d1c63e8be7,0x8addcb5645ac2ba8,
        0xb9a74a0637ce2ee1,0x6d953e2bd7173692,
        0xe8111c87c5c1ba99,0xc8fa8db6ccdd0437,
        0x910ab1d4db9914a0,0x1d9c9892400a22a2,
        0xb54d5e4a127f59c8,0x2503beb6d00cab4b,
        0xe2a0b5dc971f303a,0x2e44ae64840fd61d,
        0x8da471a9de737e24,0x5ceaecfed289e5d2,
        0xb10d8e1456105dad,0x7425a83e872c5f47,
        0xdd50f1996b947518,0xd12f124e28f77719,
        0x8a5296ffe33cc92f,0x82bd6b70d99aaa6f,
        0xace73cbfdc0bfb7b,0x636cc64d1001550b,
        0xd8210befd30efa5a,0x3c47f7e05401aa4e,
        0x8714a775e3e95c78,0x65acfaec34810a71,
        0xa8d9d1535ce3b396,0x7f1839a741a14d0d,
        0xd31045a8341ca07c,0x1ede48111209a050,
        0x83ea2b892091e44d,0x934aed0aab460432,
        0xa4e4b66b68b65d60,0xf81da84d5617853f,
        0xce1de40642e3f4b9,0x36251260ab9d668e,
        0x80d2ae83e9ce78f3,0xc1d72b7c6b426019,
        0xa1075a24e4421730,0xb24cf65b8612f81f,
        0xc94930ae1d529cfc,0xdee033f26797b627,
        0xfb9b7cd9a4a7443c,0x169840ef017da3b1,
        0x9d412e0806e88aa5,0x8e1f289560ee864e,
        0xc491798a08a2ad4e,0xf1a6f2bab92a27e2,
        0xf5b5d7ec8acb58a2,0xae10af696774b1db,
        0x9991a6f3d6bf1765,0xacca6da1e0a8ef29,
        0xbff610b0cc6edd3f,0x17fd090a58d32af3,
        0xeff394dcff8a948e,0xddfc4b4cef07f5b0,
        0x95f83d0a1fb69cd9,0x4abdaf101564f98e,
        0xbb764c4ca7a4440f,0x9d6d1ad41abe37f1,
        0xea53df5fd18d5513,0x84c86189216dc5ed,
        0x92746b9be2f8552c,0x32fd3cf5b4e49bb4,
        0xb7118682dbb66a77,0x3fbc8c33221dc2a1,
        0xe4d5e82392a40515,0xfabaf3feaa5334a,
        0x8f05b1163ba6832d,0x29cb4d87f2a7400e,
        0xb2c71d5bca9023f8,0x743e20e9ef511012,
        0xdf78e4b2bd342cf6,0x914da9246b255416,
        0x8bab8eefb6409c1a,0x1ad089b6c2f7548e,
        0xae9672aba3d0c320,0xa184ac2473b529b1,
        0xda3c0f568cc4f3e8,0xc9e5d72d90a2741e,
        0x8865899617fb1871,0x7e2fa67c7a658892,
        0xaa7eebfb9df9de8d,0xddbb901b98feeab7,
        0xd51ea6fa85785631,0x552a74227f3ea565,
        0x8533285c936b35de,0xd53a88958f87275f,
        0xa67ff273b8460356,0x8a892abaf368f137,
        0xd01fef10a657842c,0x2d2b7569b0432d85,
        0x8213f56a67f6b29b,0x9c3b29620e29fc73,
        0xa298f2c501f45f42,0x8349f3ba91b47b8f,
        0xcb3f2f7642717713,0x241c70a936219a73,
        0xfe0efb53d30dd4d7,0xed238cd383aa0110,
        0x9ec95d1463e8a506,0xf4363804324a40aa,
        0xc67bb4597ce2ce48,0xb143c6053edcd0d5,
        0xf81aa16fdc1b81da,0xdd94b7868e94050a,
        0x9b10a4e5e9913128,0xca7cf2b4191c8326,
        0xc1d4ce1f63f57d72,0xfd1c2f611f63a3f0,
        0xf24a01a73cf2dccf,0xbc633b39673c8cec,
        0x976e41088617ca01,0xd5be0503e085d813,
        0xbd49d14aa79dbc82,0x4b2d8644d8a74e18,
        0xec9c459d51852ba2,0xddf8e7d60ed1219e,
        0x93e1ab8252f33b45,0xcabb90e5c942b503,
        0xb8da1662e7b00a17,0x3d6a751f3b936243,
        0xe7109bfba19c0c9d,0xcc512670a783ad4,
        0x906a617d450187e2,0x27fb2b80668b24c5,
        0xb484f9dc9641e9da,0xb1f9f660802dedf6,
        0xe1a63853bbd26451,0x5e7873f8a0396973,
        0x8d07e33455637eb2,0xdb0b487b6423e1e8,
        0xb049dc016abc5e5f,0x91ce1a9a3d2cda62,
        0xdc5c5301c56b75f7,0x7641a140cc7810fb,
        0x89b9b3e11b6329ba,0xa9e904c87fcb0a9d,
        0xac2820d9623bf429,0x546345fa9fbdcd44,
        0xd732290fbacaf133,0xa97c177947ad4095,
        0x867f59a9d4bed6c0,0x49ed8eabcccc485d,
        0xa81f301449ee8c70,0x5c68f256bfff5a74,
        0xd226fc195c6a2f8c,0x73832eec6fff3111,
        0x83585d8fd9c25db7,0xc831fd53c5ff7eab,
        0xa42e74f3d032f525,0xba3e7ca8b77f5e55,
        0xcd3a1230c43fb26f,0x28ce1bd2e55f35eb,
        0x80444b5e7aa7cf85,0x7980d163cf5b81b3,
        0xa0555e361951c366,0xd7e105bcc332621f,
        0xc86ab5c39fa63440,0x8dd9472bf3fefaa7,
        0xfa856334878fc150,0xb14f98f6f0feb951,
        0x9c935e00d4b9d8d2,0x6ed1bf9a569f33d3,
        0xc3b8358109e84f07,0xa862f80ec4700c8,
        0xf4a642e14c6262c8,0xcd27bb612758c0fa,
        0x98e7e9cccfbd7dbd,0x8038d51cb897789c,
        0xbf21e44003acdd2c,0xe0470a63e6bd56c3,
        0xeeea5d5004981478,0x1858ccfce06cac74,
        0x95527a5202df0ccb,0xf37801e0c43ebc8,
        0xbaa718e68396cffd,0xd30560258f54e6ba,
        0xe950df20247c83fd,0x47c6b82ef32a2069,
        0x91d28b7416cdd27e,0x4cdc331d57fa5441,
        0xb6472e511c81471d,0xe0133fe4adf8e952,
        0xe3d8f9e563a198e5,0x58180fddd97723a6,
        0x8e679c2f5e44ff8f,0x570f09eaa7ea7648,};
using powers = powers_template<>;

}

#endif

#ifndef FASTFLOAT_DECIMAL_TO_BINARY_H
#define FASTFLOAT_DECIMAL_TO_BINARY_H

namespace fast_float {

// This will compute or rather approximate w * 5**q and return a pair of 64-bit words approximating
// the result, with the "high" part corresponding to the most significant bits and the
// low part corresponding to the least significant bits.
//
template <int bit_precision>
fastfloat_really_inline
value128 compute_product_approximation(int64_t q, uint64_t w) {
  const int index = 2 * int(q - powers::smallest_power_of_five);
  // For small values of q, e.g., q in [0,27], the answer is always exact because
  // The line value128 firstproduct = full_multiplication(w, power_of_five_128[index]);
  // gives the exact answer.
  value128 firstproduct = full_multiplication(w, powers::power_of_five_128[index]);
  static_assert((bit_precision >= 0) && (bit_precision <= 64), " precision should  be in (0,64]");
  constexpr uint64_t precision_mask = (bit_precision < 64) ?
               (uint64_t(0xFFFFFFFFFFFFFFFF) >> bit_precision)
               : uint64_t(0xFFFFFFFFFFFFFFFF);
  if((firstproduct.high & precision_mask) == precision_mask) { // could further guard with  (lower + w < lower)
    // regarding the second product, we only need secondproduct.high, but our expectation is that the compiler will optimize this extra work away if needed.
    value128 secondproduct = full_multiplication(w, powers::power_of_five_128[index + 1]);
    firstproduct.low += secondproduct.high;
    if(secondproduct.high > firstproduct.low) {
      firstproduct.high++;
    }
  }
  return firstproduct;
}

namespace detail {
/**
 * For q in (0,350), we have that
 *  f = (((152170 + 65536) * q ) >> 16);
 * is equal to
 *   floor(p) + q
 * where
 *   p = log(5**q)/log(2) = q * log(5)/log(2)
 *
 * For negative values of q in (-400,0), we have that 
 *  f = (((152170 + 65536) * q ) >> 16);
 * is equal to 
 *   -ceil(p) + q
 * where
 *   p = log(5**-q)/log(2) = -q * log(5)/log(2)
 */
  constexpr fastfloat_really_inline int32_t power(int32_t q)  noexcept  {
    return (((152170 + 65536) * q) >> 16) + 63;
  }
} // namespace detail

// create an adjusted mantissa, biased by the invalid power2
// for significant digits already multiplied by 10 ** q.
template <typename binary>
fastfloat_really_inline
adjusted_mantissa compute_error_scaled(int64_t q, uint64_t w, int lz) noexcept  {
  int hilz = int(w >> 63) ^ 1;
  adjusted_mantissa answer;
  answer.mantissa = w << hilz;
  int bias = binary::mantissa_explicit_bits() - binary::minimum_exponent();
  answer.power2 = int32_t(detail::power(int32_t(q)) + bias - hilz - lz - 62 + invalid_am_bias);
  return answer;
}

// w * 10 ** q, without rounding the representation up.
// the power2 in the exponent will be adjusted by invalid_am_bias.
template <typename binary>
fastfloat_really_inline
adjusted_mantissa compute_error(int64_t q, uint64_t w)  noexcept  {
  int lz = leading_zeroes(w);
  w <<= lz;
  value128 product = compute_product_approximation<binary::mantissa_explicit_bits() + 3>(q, w);
  return compute_error_scaled<binary>(q, product.high, lz);
}

// w * 10 ** q
// The returned value should be a valid ieee64 number that simply need to be packed.
// However, in some very rare cases, the computation will fail. In such cases, we
// return an adjusted_mantissa with a negative power of 2: the caller should recompute
// in such cases.
template <typename binary>
fastfloat_really_inline
adjusted_mantissa compute_float(int64_t q, uint64_t w)  noexcept  {
  adjusted_mantissa answer;
  if ((w == 0) || (q < binary::smallest_power_of_ten())) {
    answer.power2 = 0;
    answer.mantissa = 0;
    // result should be zero
    return answer;
  }
  if (q > binary::largest_power_of_ten()) {
    // we want to get infinity:
    answer.power2 = binary::infinite_power();
    answer.mantissa = 0;
    return answer;
  }
  // At this point in time q is in [powers::smallest_power_of_five, powers::largest_power_of_five].

  // We want the most significant bit of i to be 1. Shift if needed.
  int lz = leading_zeroes(w);
  w <<= lz;

  // The required precision is binary::mantissa_explicit_bits() + 3 because
  // 1. We need the implicit bit
  // 2. We need an extra bit for rounding purposes
  // 3. We might lose a bit due to the "upperbit" routine (result too small, requiring a shift)

  value128 product = compute_product_approximation<binary::mantissa_explicit_bits() + 3>(q, w);
  if(product.low == 0xFFFFFFFFFFFFFFFF) { //  could guard it further
    // In some very rare cases, this could happen, in which case we might need a more accurate
    // computation that what we can provide cheaply. This is very, very unlikely.
    //
    const bool inside_safe_exponent = (q >= -27) && (q <= 55); // always good because 5**q <2**128 when q>=0, 
    // and otherwise, for q<0, we have 5**-q<2**64 and the 128-bit reciprocal allows for exact computation.
    if(!inside_safe_exponent) {
      return compute_error_scaled<binary>(q, product.high, lz);
    }
  }
  // The "compute_product_approximation" function can be slightly slower than a branchless approach:
  // value128 product = compute_product(q, w);
  // but in practice, we can win big with the compute_product_approximation if its additional branch
  // is easily predicted. Which is best is data specific.
  int upperbit = int(product.high >> 63);

  answer.mantissa = product.high >> (upperbit + 64 - binary::mantissa_explicit_bits() - 3);

  answer.power2 = int32_t(detail::power(int32_t(q)) + upperbit - lz - binary::minimum_exponent());
  if (answer.power2 <= 0) { // we have a subnormal?
    // Here have that answer.power2 <= 0 so -answer.power2 >= 0
    if(-answer.power2 + 1 >= 64) { // if we have more than 64 bits below the minimum exponent, you have a zero for sure.
      answer.power2 = 0;
      answer.mantissa = 0;
      // result should be zero
      return answer;
    }
    // next line is safe because -answer.power2 + 1 < 64
    answer.mantissa >>= -answer.power2 + 1;
    // Thankfully, we can't have both "round-to-even" and subnormals because
    // "round-to-even" only occurs for powers close to 0.
    answer.mantissa += (answer.mantissa & 1); // round up
    answer.mantissa >>= 1;
    // There is a weird scenario where we don't have a subnormal but just.
    // Suppose we start with 2.2250738585072013e-308, we end up
    // with 0x3fffffffffffff x 2^-1023-53 which is technically subnormal
    // whereas 0x40000000000000 x 2^-1023-53  is normal. Now, we need to round
    // up 0x3fffffffffffff x 2^-1023-53  and once we do, we are no longer
    // subnormal, but we can only know this after rounding.
    // So we only declare a subnormal if we are smaller than the threshold.
    answer.power2 = (answer.mantissa < (uint64_t(1) << binary::mantissa_explicit_bits())) ? 0 : 1;
    return answer;
  }

  // usually, we round *up*, but if we fall right in between and and we have an
  // even basis, we need to round down
  // We are only concerned with the cases where 5**q fits in single 64-bit word.
  if ((product.low <= 1) &&  (q >= binary::min_exponent_round_to_even()) && (q <= binary::max_exponent_round_to_even()) &&
      ((answer.mantissa & 3) == 1) ) { // we may fall between two floats!
    // To be in-between two floats we need that in doing
    //   answer.mantissa = product.high >> (upperbit + 64 - binary::mantissa_explicit_bits() - 3);
    // ... we dropped out only zeroes. But if this happened, then we can go back!!!
    if((answer.mantissa  << (upperbit + 64 - binary::mantissa_explicit_bits() - 3)) ==  product.high) {
      answer.mantissa &= ~uint64_t(1);          // flip it so that we do not round up
    }
  }

  answer.mantissa += (answer.mantissa & 1); // round up
  answer.mantissa >>= 1;
  if (answer.mantissa >= (uint64_t(2) << binary::mantissa_explicit_bits())) {
    answer.mantissa = (uint64_t(1) << binary::mantissa_explicit_bits());
    answer.power2++; // undo previous addition
  }

  answer.mantissa &= ~(uint64_t(1) << binary::mantissa_explicit_bits());
  if (answer.power2 >= binary::infinite_power()) { // infinity
    answer.power2 = binary::infinite_power();
    answer.mantissa = 0;
  }
  return answer;
}

} // namespace fast_float

#endif

#ifndef FASTFLOAT_BIGINT_H
#define FASTFLOAT_BIGINT_H

namespace fast_float {

// the limb width: we want efficient multiplication of double the bits in
// limb, or for 64-bit limbs, at least 64-bit multiplication where we can
// extract the high and low parts efficiently. this is every 64-bit
// architecture except for sparc, which emulates 128-bit multiplication.
// we might have platforms where `CHAR_BIT` is not 8, so let's avoid
// doing `8 * sizeof(limb)`.
#if defined(FASTFLOAT_64BIT) && !defined(__sparc)
#define FASTFLOAT_64BIT_LIMB
typedef uint64_t limb;
constexpr size_t limb_bits = 64;
#else
#define FASTFLOAT_32BIT_LIMB
typedef uint32_t limb;
constexpr size_t limb_bits = 32;
#endif

typedef span<limb> limb_span;

// number of bits in a bigint. this needs to be at least the number
// of bits required to store the largest bigint, which is
// `log2(10**(digits + max_exp))`, or `log2(10**(767 + 342))`, or
// ~3600 bits, so we round to 4000.
constexpr size_t bigint_bits = 4000;
constexpr size_t bigint_limbs = bigint_bits / limb_bits;

// vector-like type that is allocated on the stack. the entire
// buffer is pre-allocated, and only the length changes.
template <uint16_t size>
struct stackvec {
  limb data[size];
  // we never need more than 150 limbs
  uint16_t length{0};

  stackvec() = default;
  stackvec(const stackvec &) = delete;
  stackvec &operator=(const stackvec &) = delete;
  stackvec(stackvec &&) = delete;
  stackvec &operator=(stackvec &&other) = delete;

  // create stack vector from existing limb span.
  stackvec(limb_span s) {
    FASTFLOAT_ASSERT(try_extend(s));
  }

  limb& operator[](size_t index) noexcept {
    FASTFLOAT_DEBUG_ASSERT(index < length);
    return data[index];
  }
  const limb& operator[](size_t index) const noexcept {
    FASTFLOAT_DEBUG_ASSERT(index < length);
    return data[index];
  }
  // index from the end of the container
  const limb& rindex(size_t index) const noexcept {
    FASTFLOAT_DEBUG_ASSERT(index < length);
    size_t rindex = length - index - 1;
    return data[rindex];
  }

  // set the length, without bounds checking.
  void set_len(size_t len) noexcept {
    length = uint16_t(len);
  }
  constexpr size_t len() const noexcept {
    return length;
  }
  constexpr bool is_empty() const noexcept {
    return length == 0;
  }
  constexpr size_t capacity() const noexcept {
    return size;
  }
  // append item to vector, without bounds checking
  void push_unchecked(limb value) noexcept {
    data[length] = value;
    length++;
  }
  // append item to vector, returning if item was added
  bool try_push(limb value) noexcept {
    if (len() < capacity()) {
      push_unchecked(value);
      return true;
    } else {
      return false;
    }
  }
  // add items to the vector, from a span, without bounds checking
  void extend_unchecked(limb_span s) noexcept {
    limb* ptr = data + length;
    ::memcpy((void*)ptr, (const void*)s.ptr, sizeof(limb) * s.len());
    set_len(len() + s.len());
  }
  // try to add items to the vector, returning if items were added
  bool try_extend(limb_span s) noexcept {
    if (len() + s.len() <= capacity()) {
      extend_unchecked(s);
      return true;
    } else {
      return false;
    }
  }
  // resize the vector, without bounds checking
  // if the new size is longer than the vector, assign value to each
  // appended item.
  void resize_unchecked(size_t new_len, limb value) noexcept {
    if (new_len > len()) {
      size_t count = new_len - len();
      limb* first = data + len();
      limb* last = first + count;
      ::std::fill(first, last, value);
      set_len(new_len);
    } else {
      set_len(new_len);
    }
  }
  // try to resize the vector, returning if the vector was resized.
  bool try_resize(size_t new_len, limb value) noexcept {
    if (new_len > capacity()) {
      return false;
    } else {
      resize_unchecked(new_len, value);
      return true;
    }
  }
  // check if any limbs are non-zero after the given index.
  // this needs to be done in reverse order, since the index
  // is relative to the most significant limbs.
  bool nonzero(size_t index) const noexcept {
    while (index < len()) {
      if (rindex(index) != 0) {
        return true;
      }
      index++;
    }
    return false;
  }
  // normalize the big integer, so most-significant zero limbs are removed.
  void normalize() noexcept {
    while (len() > 0 && rindex(0) == 0) {
      length--;
    }
  }
};

fastfloat_really_inline
uint64_t empty_hi64(bool& truncated) noexcept {
  truncated = false;
  return 0;
}

fastfloat_really_inline
uint64_t uint64_hi64(uint64_t r0, bool& truncated) noexcept {
  truncated = false;
  int shl = leading_zeroes(r0);
  return r0 << shl;
}

fastfloat_really_inline
uint64_t uint64_hi64(uint64_t r0, uint64_t r1, bool& truncated) noexcept {
  int shl = leading_zeroes(r0);
  if (shl == 0) {
    truncated = r1 != 0;
    return r0;
  } else {
    int shr = 64 - shl;
    truncated = (r1 << shl) != 0;
    return (r0 << shl) | (r1 >> shr);
  }
}

fastfloat_really_inline
uint64_t uint32_hi64(uint32_t r0, bool& truncated) noexcept {
  return uint64_hi64(r0, truncated);
}

fastfloat_really_inline
uint64_t uint32_hi64(uint32_t r0, uint32_t r1, bool& truncated) noexcept {
  uint64_t x0 = r0;
  uint64_t x1 = r1;
  return uint64_hi64((x0 << 32) | x1, truncated);
}

fastfloat_really_inline
uint64_t uint32_hi64(uint32_t r0, uint32_t r1, uint32_t r2, bool& truncated) noexcept {
  uint64_t x0 = r0;
  uint64_t x1 = r1;
  uint64_t x2 = r2;
  return uint64_hi64(x0, (x1 << 32) | x2, truncated);
}

// add two small integers, checking for overflow.
// we want an efficient operation. for msvc, where
// we don't have built-in intrinsics, this is still
// pretty fast.
fastfloat_really_inline
limb scalar_add(limb x, limb y, bool& overflow) noexcept {
  limb z;

// gcc and clang
#if defined(__has_builtin)
  #if __has_builtin(__builtin_add_overflow)
    overflow = __builtin_add_overflow(x, y, &z);
    return z;
  #endif
#endif

  // generic, this still optimizes correctly on MSVC.
  z = x + y;
  overflow = z < x;
  return z;
}

// multiply two small integers, getting both the high and low bits.
fastfloat_really_inline
limb scalar_mul(limb x, limb y, limb& carry) noexcept {
#ifdef FASTFLOAT_64BIT_LIMB
  #if defined(__SIZEOF_INT128__)
  // GCC and clang both define it as an extension.
  __uint128_t z = __uint128_t(x) * __uint128_t(y) + __uint128_t(carry);
  carry = limb(z >> limb_bits);
  return limb(z);
  #else
  // fallback, no native 128-bit integer multiplication with carry.
  // on msvc, this optimizes identically, somehow.
  value128 z = full_multiplication(x, y);
  bool overflow;
  z.low = scalar_add(z.low, carry, overflow);
  z.high += uint64_t(overflow);  // cannot overflow
  carry = z.high;
  return z.low;
  #endif
#else
  uint64_t z = uint64_t(x) * uint64_t(y) + uint64_t(carry);
  carry = limb(z >> limb_bits);
  return limb(z);
#endif
}

// add scalar value to bigint starting from offset.
// used in grade school multiplication
template <uint16_t size>
inline bool small_add_from(stackvec<size>& vec, limb y, size_t start) noexcept {
  size_t index = start;
  limb carry = y;
  bool overflow;
  while (carry != 0 && index < vec.len()) {
    vec[index] = scalar_add(vec[index], carry, overflow);
    carry = limb(overflow);
    index += 1;
  }
  if (carry != 0) {
    FASTFLOAT_TRY(vec.try_push(carry));
  }
  return true;
}

// add scalar value to bigint.
template <uint16_t size>
fastfloat_really_inline bool small_add(stackvec<size>& vec, limb y) noexcept {
  return small_add_from(vec, y, 0);
}

// multiply bigint by scalar value.
template <uint16_t size>
inline bool small_mul(stackvec<size>& vec, limb y) noexcept {
  limb carry = 0;
  for (size_t index = 0; index < vec.len(); index++) {
    vec[index] = scalar_mul(vec[index], y, carry);
  }
  if (carry != 0) {
    FASTFLOAT_TRY(vec.try_push(carry));
  }
  return true;
}

// add bigint to bigint starting from index.
// used in grade school multiplication
template <uint16_t size>
bool large_add_from(stackvec<size>& x, limb_span y, size_t start) noexcept {
  // the effective x buffer is from `xstart..x.len()`, so exit early
  // if we can't get that current range.
  if (x.len() < start || y.len() > x.len() - start) {
      FASTFLOAT_TRY(x.try_resize(y.len() + start, 0));
  }

  bool carry = false;
  for (size_t index = 0; index < y.len(); index++) {
    limb xi = x[index + start];
    limb yi = y[index];
    bool c1 = false;
    bool c2 = false;
    xi = scalar_add(xi, yi, c1);
    if (carry) {
      xi = scalar_add(xi, 1, c2);
    }
    x[index + start] = xi;
    carry = c1 | c2;
  }

  // handle overflow
  if (carry) {
    FASTFLOAT_TRY(small_add_from(x, 1, y.len() + start));
  }
  return true;
}

// add bigint to bigint.
template <uint16_t size>
fastfloat_really_inline bool large_add_from(stackvec<size>& x, limb_span y) noexcept {
  return large_add_from(x, y, 0);
}

// grade-school multiplication algorithm
template <uint16_t size>
bool long_mul(stackvec<size>& x, limb_span y) noexcept {
  limb_span xs = limb_span(x.data, x.len());
  stackvec<size> z(xs);
  limb_span zs = limb_span(z.data, z.len());

  if (y.len() != 0) {
    limb y0 = y[0];
    FASTFLOAT_TRY(small_mul(x, y0));
    for (size_t index = 1; index < y.len(); index++) {
      limb yi = y[index];
      stackvec<size> zi;
      if (yi != 0) {
        // re-use the same buffer throughout
        zi.set_len(0);
        FASTFLOAT_TRY(zi.try_extend(zs));
        FASTFLOAT_TRY(small_mul(zi, yi));
        limb_span zis = limb_span(zi.data, zi.len());
        FASTFLOAT_TRY(large_add_from(x, zis, index));
      }
    }
  }

  x.normalize();
  return true;
}

// grade-school multiplication algorithm
template <uint16_t size>
bool large_mul(stackvec<size>& x, limb_span y) noexcept {
  if (y.len() == 1) {
    FASTFLOAT_TRY(small_mul(x, y[0]));
  } else {
    FASTFLOAT_TRY(long_mul(x, y));
  }
  return true;
}

// big integer type. implements a small subset of big integer
// arithmetic, using simple algorithms since asymptotically
// faster algorithms are slower for a small number of limbs.
// all operations assume the big-integer is normalized.
struct bigint {
  // storage of the limbs, in little-endian order.
  stackvec<bigint_limbs> vec;

  bigint(): vec() {}
  bigint(const bigint &) = delete;
  bigint &operator=(const bigint &) = delete;
  bigint(bigint &&) = delete;
  bigint &operator=(bigint &&other) = delete;

  bigint(uint64_t value): vec() {
#ifdef FASTFLOAT_64BIT_LIMB
    vec.push_unchecked(value);
#else
    vec.push_unchecked(uint32_t(value));
    vec.push_unchecked(uint32_t(value >> 32));
#endif
    vec.normalize();
  }

  // get the high 64 bits from the vector, and if bits were truncated.
  // this is to get the significant digits for the float.
  uint64_t hi64(bool& truncated) const noexcept {
#ifdef FASTFLOAT_64BIT_LIMB
    if (vec.len() == 0) {
      return empty_hi64(truncated);
    } else if (vec.len() == 1) {
      return uint64_hi64(vec.rindex(0), truncated);
    } else {
      uint64_t result = uint64_hi64(vec.rindex(0), vec.rindex(1), truncated);
      truncated |= vec.nonzero(2);
      return result;
    }
#else
    if (vec.len() == 0) {
      return empty_hi64(truncated);
    } else if (vec.len() == 1) {
      return uint32_hi64(vec.rindex(0), truncated);
    } else if (vec.len() == 2) {
      return uint32_hi64(vec.rindex(0), vec.rindex(1), truncated);
    } else {
      uint64_t result = uint32_hi64(vec.rindex(0), vec.rindex(1), vec.rindex(2), truncated);
      truncated |= vec.nonzero(3);
      return result;
    }
#endif
  }

  // compare two big integers, returning the large value.
  // assumes both are normalized. if the return value is
  // negative, other is larger, if the return value is
  // positive, this is larger, otherwise they are equal.
  // the limbs are stored in little-endian order, so we
  // must compare the limbs in ever order.
  int compare(const bigint& other) const noexcept {
    if (vec.len() > other.vec.len()) {
      return 1;
    } else if (vec.len() < other.vec.len()) {
      return -1;
    } else {
      for (size_t index = vec.len(); index > 0; index--) {
        limb xi = vec[index - 1];
        limb yi = other.vec[index - 1];
        if (xi > yi) {
          return 1;
        } else if (xi < yi) {
          return -1;
        }
      }
      return 0;
    }
  }

  // shift left each limb n bits, carrying over to the new limb
  // returns true if we were able to shift all the digits.
  bool shl_bits(size_t n) noexcept {
    // Internally, for each item, we shift left by n, and add the previous
    // right shifted limb-bits.
    // For example, we transform (for u8) shifted left 2, to:
    //      b10100100 b01000010
    //      b10 b10010001 b00001000
    FASTFLOAT_DEBUG_ASSERT(n != 0);
    FASTFLOAT_DEBUG_ASSERT(n < sizeof(limb) * 8);

    size_t shl = n;
    size_t shr = limb_bits - shl;
    limb prev = 0;
    for (size_t index = 0; index < vec.len(); index++) {
      limb xi = vec[index];
      vec[index] = (xi << shl) | (prev >> shr);
      prev = xi;
    }

    limb carry = prev >> shr;
    if (carry != 0) {
      return vec.try_push(carry);
    }
    return true;
  }

  // move the limbs left by `n` limbs.
  bool shl_limbs(size_t n) noexcept {
    FASTFLOAT_DEBUG_ASSERT(n != 0);
    if (n + vec.len() > vec.capacity()) {
      return false;
    } else if (!vec.is_empty()) {
      // move limbs
      limb* dst = vec.data + n;
      const limb* src = vec.data;
      ::memmove(dst, src, sizeof(limb) * vec.len());
      // fill in empty limbs
      limb* first = vec.data;
      limb* last = first + n;
      ::std::fill(first, last, 0);
      vec.set_len(n + vec.len());
      return true;
    } else {
      return true;
    }
  }

  // move the limbs left by `n` bits.
  bool shl(size_t n) noexcept {
    size_t rem = n % limb_bits;
    size_t div = n / limb_bits;
    if (rem != 0) {
      FASTFLOAT_TRY(shl_bits(rem));
    }
    if (div != 0) {
      FASTFLOAT_TRY(shl_limbs(div));
    }
    return true;
  }

  // get the number of leading zeros in the bigint.
  int ctlz() const noexcept {
    if (vec.is_empty()) {
      return 0;
    } else {
#ifdef FASTFLOAT_64BIT_LIMB
      return leading_zeroes(vec.rindex(0));
#else
      // no use defining a specialized leading_zeroes for a 32-bit type.
      uint64_t r0 = vec.rindex(0);
      return leading_zeroes(r0 << 32);
#endif
    }
  }

  // get the number of bits in the bigint.
  int bit_length() const noexcept {
    int lz = ctlz();
    return int(limb_bits * vec.len()) - lz;
  }

  bool mul(limb y) noexcept {
    return small_mul(vec, y);
  }

  bool add(limb y) noexcept {
    return small_add(vec, y);
  }

  // multiply as if by 2 raised to a power.
  bool pow2(uint32_t exp) noexcept {
    return shl(exp);
  }

  // multiply as if by 5 raised to a power.
  bool pow5(uint32_t exp) noexcept {
    // multiply by a power of 5
    static constexpr uint32_t large_step = 135;
    static constexpr uint64_t small_power_of_5[] = {
      1UL, 5UL, 25UL, 125UL, 625UL, 3125UL, 15625UL, 78125UL, 390625UL,
      1953125UL, 9765625UL, 48828125UL, 244140625UL, 1220703125UL,
      6103515625UL, 30517578125UL, 152587890625UL, 762939453125UL,
      3814697265625UL, 19073486328125UL, 95367431640625UL, 476837158203125UL,
      2384185791015625UL, 11920928955078125UL, 59604644775390625UL,
      298023223876953125UL, 1490116119384765625UL, 7450580596923828125UL,
    };
#ifdef FASTFLOAT_64BIT_LIMB
    constexpr static limb large_power_of_5[] = {
      1414648277510068013UL, 9180637584431281687UL, 4539964771860779200UL,
      10482974169319127550UL, 198276706040285095UL};
#else
    constexpr static limb large_power_of_5[] = {
      4279965485U, 329373468U, 4020270615U, 2137533757U, 4287402176U,
      1057042919U, 1071430142U, 2440757623U, 381945767U, 46164893U};
#endif
    size_t large_length = sizeof(large_power_of_5) / sizeof(limb);
    limb_span large = limb_span(large_power_of_5, large_length);
    while (exp >= large_step) {
      FASTFLOAT_TRY(large_mul(vec, large));
      exp -= large_step;
    }
#ifdef FASTFLOAT_64BIT_LIMB
    uint32_t small_step = 27;
    limb max_native = 7450580596923828125UL;
#else
    uint32_t small_step = 13;
    limb max_native = 1220703125U;
#endif
    while (exp >= small_step) {
      FASTFLOAT_TRY(small_mul(vec, max_native));
      exp -= small_step;
    }
    if (exp != 0) {
      FASTFLOAT_TRY(small_mul(vec, limb(small_power_of_5[exp])));
    }

    return true;
  }

  // multiply as if by 10 raised to a power.
  bool pow10(uint32_t exp) noexcept {
    FASTFLOAT_TRY(pow5(exp));
    return pow2(exp);
  }
};

} // namespace fast_float

#endif

#ifndef FASTFLOAT_ASCII_NUMBER_H
#define FASTFLOAT_ASCII_NUMBER_H

namespace fast_float {

// Next function can be micro-optimized, but compilers are entirely
// able to optimize it well.
fastfloat_really_inline bool is_integer(char c)  noexcept  { return c >= '0' && c <= '9'; }

fastfloat_really_inline uint64_t byteswap(uint64_t val) {
  return (val & 0xFF00000000000000) >> 56
    | (val & 0x00FF000000000000) >> 40
    | (val & 0x0000FF0000000000) >> 24
    | (val & 0x000000FF00000000) >> 8
    | (val & 0x00000000FF000000) << 8
    | (val & 0x0000000000FF0000) << 24
    | (val & 0x000000000000FF00) << 40
    | (val & 0x00000000000000FF) << 56;
}

fastfloat_really_inline uint64_t read_u64(const char *chars) {
  uint64_t val;
  ::memcpy(&val, chars, sizeof(uint64_t));
#if FASTFLOAT_IS_BIG_ENDIAN == 1
  // Need to read as-if the number was in little-endian order.
  val = byteswap(val);
#endif
  return val;
}

fastfloat_really_inline void write_u64(uint8_t *chars, uint64_t val) {
#if FASTFLOAT_IS_BIG_ENDIAN == 1
  // Need to read as-if the number was in little-endian order.
  val = byteswap(val);
#endif
  ::memcpy(chars, &val, sizeof(uint64_t));
}

// credit  @aqrit
fastfloat_really_inline uint32_t  parse_eight_digits_unrolled(uint64_t val) {
  const uint64_t mask = 0x000000FF000000FF;
  const uint64_t mul1 = 0x000F424000000064; // 100 + (1000000ULL << 32)
  const uint64_t mul2 = 0x0000271000000001; // 1 + (10000ULL << 32)
  val -= 0x3030303030303030;
  val = (val * 10) + (val >> 8); // val = (val * 2561) >> 8;
  val = (((val & mask) * mul1) + (((val >> 16) & mask) * mul2)) >> 32;
  return uint32_t(val);
}

fastfloat_really_inline uint32_t parse_eight_digits_unrolled(const char *chars)  noexcept  {
  return parse_eight_digits_unrolled(read_u64(chars));
}

// credit @aqrit
fastfloat_really_inline bool is_made_of_eight_digits_fast(uint64_t val)  noexcept  {
  return !((((val + 0x4646464646464646) | (val - 0x3030303030303030)) &
     0x8080808080808080));
}

fastfloat_really_inline bool is_made_of_eight_digits_fast(const char *chars)  noexcept  {
  return is_made_of_eight_digits_fast(read_u64(chars));
}

typedef span<const char> byte_span;

struct parsed_number_string {
  int64_t exponent{0};
  uint64_t mantissa{0};
  const char *lastmatch{nullptr};
  bool negative{false};
  bool valid{false};
  bool too_many_digits{false};
  // contains the range of the significant digits
  byte_span integer{};  // non-nullable
  byte_span fraction{}; // nullable
};

// Assuming that you use no more than 19 digits, this will
// parse an ASCII string.
fastfloat_really_inline
parsed_number_string parse_number_string(const char *p, const char *pend, parse_options options) noexcept {
  const chars_format fmt = options.format;
  const char decimal_point = options.decimal_point;

  parsed_number_string answer;
  answer.valid = false;
  answer.too_many_digits = false;
  answer.negative = (*p == '-');
  if (*p == '-') { // C++17 20.19.3.(7.1) explicitly forbids '+' sign here
    ++p;
    if (p == pend) {
      return answer;
    }
    if (!is_integer(*p) && (*p != decimal_point)) { // a sign must be followed by an integer or the dot
      return answer;
    }
  }
  const char *const start_digits = p;

  uint64_t i = 0; // an unsigned int avoids signed overflows (which are bad)

  while ((std::distance(p, pend) >= 8) && is_made_of_eight_digits_fast(p)) {
    i = i * 100000000 + parse_eight_digits_unrolled(p); // in rare cases, this will overflow, but that's ok
    p += 8;
  }
  while ((p != pend) && is_integer(*p)) {
    // a multiplication by 10 is cheaper than an arbitrary integer
    // multiplication
    i = 10 * i +
        uint64_t(*p - '0'); // might overflow, we will handle the overflow later
    ++p;
  }
  const char *const end_of_integer_part = p;
  int64_t digit_count = int64_t(end_of_integer_part - start_digits);
  answer.integer = byte_span(start_digits, size_t(digit_count));
  int64_t exponent = 0;
  if ((p != pend) && (*p == decimal_point)) {
    ++p;
    const char* before = p;
    // can occur at most twice without overflowing, but let it occur more, since
    // for integers with many digits, digit parsing is the primary bottleneck.
    while ((std::distance(p, pend) >= 8) && is_made_of_eight_digits_fast(p)) {
      i = i * 100000000 + parse_eight_digits_unrolled(p); // in rare cases, this will overflow, but that's ok
      p += 8;
    }
    while ((p != pend) && is_integer(*p)) {
      uint8_t digit = uint8_t(*p - '0');
      ++p;
      i = i * 10 + digit; // in rare cases, this will overflow, but that's ok
    }
    exponent = before - p;
    answer.fraction = byte_span(before, size_t(p - before));
    digit_count -= exponent;
  }
  // we must have encountered at least one integer!
  if (digit_count == 0) {
    return answer;
  }
  int64_t exp_number = 0;            // explicit exponential part
  if ((fmt & chars_format::scientific) && (p != pend) && (('e' == *p) || ('E' == *p))) {
    const char * location_of_e = p;
    ++p;
    bool neg_exp = false;
    if ((p != pend) && ('-' == *p)) {
      neg_exp = true;
      ++p;
    } else if ((p != pend) && ('+' == *p)) { // '+' on exponent is allowed by C++17 20.19.3.(7.1)
      ++p;
    }
    if ((p == pend) || !is_integer(*p)) {
      if(!(fmt & chars_format::fixed)) {
        // We are in error.
        return answer;
      }
      // Otherwise, we will be ignoring the 'e'.
      p = location_of_e;
    } else {
      while ((p != pend) && is_integer(*p)) {
        uint8_t digit = uint8_t(*p - '0');
        if (exp_number < 0x10000000) {
          exp_number = 10 * exp_number + digit;
        }
        ++p;
      }
      if(neg_exp) { exp_number = - exp_number; }
      exponent += exp_number;
    }
  } else {
    // If it scientific and not fixed, we have to bail out.
    if((fmt & chars_format::scientific) && !(fmt & chars_format::fixed)) { return answer; }
  }
  answer.lastmatch = p;
  answer.valid = true;

  // If we frequently had to deal with long strings of digits,
  // we could extend our code by using a 128-bit integer instead
  // of a 64-bit integer. However, this is uncommon.
  //
  // We can deal with up to 19 digits.
  if (digit_count > 19) { // this is uncommon
    // It is possible that the integer had an overflow.
    // We have to handle the case where we have 0.0000somenumber.
    // We need to be mindful of the case where we only have zeroes...
    // E.g., 0.000000000...000.
    const char *start = start_digits;
    while ((start != pend) && (*start == '0' || *start == decimal_point)) {
      if(*start == '0') { digit_count --; }
      start++;
    }
    if (digit_count > 19) {
      answer.too_many_digits = true;
      // Let us start again, this time, avoiding overflows.
      // We don't need to check if is_integer, since we use the
      // pre-tokenized spans from above.
      i = 0;
      p = answer.integer.ptr;
      const char* int_end = p + answer.integer.len();
      const uint64_t minimal_nineteen_digit_integer{1000000000000000000};
      while((i < minimal_nineteen_digit_integer) && (p != int_end)) {
        i = i * 10 + uint64_t(*p - '0');
        ++p;
      }
      if (i >= minimal_nineteen_digit_integer) { // We have a big integers
        exponent = end_of_integer_part - p + exp_number;
      } else { // We have a value with a fractional component.
          p = answer.fraction.ptr;
          const char* frac_end = p + answer.fraction.len();
          while((i < minimal_nineteen_digit_integer) && (p != frac_end)) {
            i = i * 10 + uint64_t(*p - '0');
            ++p;
          }
          exponent = answer.fraction.ptr - p + exp_number;
      }
      // We have now corrected both exponent and i, to a truncated value
    }
  }
  answer.exponent = exponent;
  answer.mantissa = i;
  return answer;
}

} // namespace fast_float

#endif

#ifndef FASTFLOAT_DIGIT_COMPARISON_H
#define FASTFLOAT_DIGIT_COMPARISON_H

namespace fast_float {

// 1e0 to 1e19
constexpr static uint64_t powers_of_ten_uint64[] = {
    1UL, 10UL, 100UL, 1000UL, 10000UL, 100000UL, 1000000UL, 10000000UL, 100000000UL,
    1000000000UL, 10000000000UL, 100000000000UL, 1000000000000UL, 10000000000000UL,
    100000000000000UL, 1000000000000000UL, 10000000000000000UL, 100000000000000000UL,
    1000000000000000000UL, 10000000000000000000UL};

// calculate the exponent, in scientific notation, of the number.
// this algorithm is not even close to optimized, but it has no practical
// effect on performance: in order to have a faster algorithm, we'd need
// to slow down performance for faster algorithms, and this is still fast.
fastfloat_really_inline int32_t scientific_exponent(parsed_number_string& num) noexcept {
  uint64_t mantissa = num.mantissa;
  int32_t exponent = int32_t(num.exponent);
  while (mantissa >= 10000) {
    mantissa /= 10000;
    exponent += 4;
  }
  while (mantissa >= 100) {
    mantissa /= 100;
    exponent += 2;
  }
  while (mantissa >= 10) {
    mantissa /= 10;
    exponent += 1;
  }
  return exponent;
}

// this converts a native floating-point number to an extended-precision float.
template <typename T>
fastfloat_really_inline adjusted_mantissa to_extended(T value) noexcept {
  using equiv_uint = typename binary_format<T>::equiv_uint;
  constexpr equiv_uint exponent_mask = binary_format<T>::exponent_mask();
  constexpr equiv_uint mantissa_mask = binary_format<T>::mantissa_mask();
  constexpr equiv_uint hidden_bit_mask = binary_format<T>::hidden_bit_mask();

  adjusted_mantissa am;
  int32_t bias = binary_format<T>::mantissa_explicit_bits() - binary_format<T>::minimum_exponent();
  equiv_uint bits;
  ::memcpy(&bits, &value, sizeof(T));
  if ((bits & exponent_mask) == 0) {
    // denormal
    am.power2 = 1 - bias;
    am.mantissa = bits & mantissa_mask;
  } else {
    // normal
    am.power2 = int32_t((bits & exponent_mask) >> binary_format<T>::mantissa_explicit_bits());
    am.power2 -= bias;
    am.mantissa = (bits & mantissa_mask) | hidden_bit_mask;
  }

  return am;
}

// get the extended precision value of the halfway point between b and b+u.
// we are given a native float that represents b, so we need to adjust it
// halfway between b and b+u.
template <typename T>
fastfloat_really_inline adjusted_mantissa to_extended_halfway(T value) noexcept {
  adjusted_mantissa am = to_extended(value);
  am.mantissa <<= 1;
  am.mantissa += 1;
  am.power2 -= 1;
  return am;
}

// round an extended-precision float to the nearest machine float.
template <typename T, typename callback>
fastfloat_really_inline void round(adjusted_mantissa& am, callback cb) noexcept {
  int32_t mantissa_shift = 64 - binary_format<T>::mantissa_explicit_bits() - 1;
  if (-am.power2 >= mantissa_shift) {
    // have a denormal float
    int32_t shift = -am.power2 + 1;
    cb(am, std::min<int32_t>(shift, 64));
    // check for round-up: if rounding-nearest carried us to the hidden bit.
    am.power2 = (am.mantissa < (uint64_t(1) << binary_format<T>::mantissa_explicit_bits())) ? 0 : 1;
    return;
  }

  // have a normal float, use the default shift.
  cb(am, mantissa_shift);

  // check for carry
  if (am.mantissa >= (uint64_t(2) << binary_format<T>::mantissa_explicit_bits())) {
    am.mantissa = (uint64_t(1) << binary_format<T>::mantissa_explicit_bits());
    am.power2++;
  }

  // check for infinite: we could have carried to an infinite power
  am.mantissa &= ~(uint64_t(1) << binary_format<T>::mantissa_explicit_bits());
  if (am.power2 >= binary_format<T>::infinite_power()) {
    am.power2 = binary_format<T>::infinite_power();
    am.mantissa = 0;
  }
}

template <typename callback>
fastfloat_really_inline
void round_nearest_tie_even(adjusted_mantissa& am, int32_t shift, callback cb) noexcept {
  uint64_t mask;
  uint64_t halfway;
  if (shift == 64) {
    mask = UINT64_MAX;
  } else {
    mask = (uint64_t(1) << shift) - 1;
  }
  if (shift == 0) {
    halfway = 0;
  } else {
    halfway = uint64_t(1) << (shift - 1);
  }
  uint64_t truncated_bits = am.mantissa & mask;
  uint64_t is_above = truncated_bits > halfway;
  uint64_t is_halfway = truncated_bits == halfway;

  // shift digits into position
  if (shift == 64) {
    am.mantissa = 0;
  } else {
    am.mantissa >>= shift;
  }
  am.power2 += shift;

  bool is_odd = (am.mantissa & 1) == 1;
  am.mantissa += uint64_t(cb(is_odd, is_halfway, is_above));
}

fastfloat_really_inline void round_down(adjusted_mantissa& am, int32_t shift) noexcept {
  if (shift == 64) {
    am.mantissa = 0;
  } else {
    am.mantissa >>= shift;
  }
  am.power2 += shift;
}

fastfloat_really_inline void skip_zeros(const char*& first, const char* last) noexcept {
  uint64_t val;
  while (std::distance(first, last) >= 8) {
    ::memcpy(&val, first, sizeof(uint64_t));
    if (val != 0x3030303030303030) {
      break;
    }
    first += 8;
  }
  while (first != last) {
    if (*first != '0') {
      break;
    }
    first++;
  }
}

// determine if any non-zero digits were truncated.
// all characters must be valid digits.
fastfloat_really_inline bool is_truncated(const char* first, const char* last) noexcept {
  // do 8-bit optimizations, can just compare to 8 literal 0s.
  uint64_t val;
  while (std::distance(first, last) >= 8) {
    ::memcpy(&val, first, sizeof(uint64_t));
    if (val != 0x3030303030303030) {
      return true;
    }
    first += 8;
  }
  while (first != last) {
    if (*first != '0') {
      return true;
    }
    first++;
  }
  return false;
}

fastfloat_really_inline bool is_truncated(byte_span s) noexcept {
  return is_truncated(s.ptr, s.ptr + s.len());
}

fastfloat_really_inline
void parse_eight_digits(const char*& p, limb& value, size_t& counter, size_t& count) noexcept {
  value = value * 100000000 + parse_eight_digits_unrolled(p);
  p += 8;
  counter += 8;
  count += 8;
}

fastfloat_really_inline
void parse_one_digit(const char*& p, limb& value, size_t& counter, size_t& count) noexcept {
  value = value * 10 + limb(*p - '0');
  p++;
  counter++;
  count++;
}

fastfloat_really_inline
void add_native(bigint& big, limb power, limb value) noexcept {
  big.mul(power);
  big.add(value);
}

fastfloat_really_inline void round_up_bigint(bigint& big, size_t& count) noexcept {
  // need to round-up the digits, but need to avoid rounding
  // ....9999 to ...10000, which could cause a false halfway point.
  add_native(big, 10, 1);
  count++;
}

// parse the significant digits into a big integer
inline void parse_mantissa(bigint& result, parsed_number_string& num, size_t max_digits, size_t& digits) noexcept {
  // try to minimize the number of big integer and scalar multiplication.
  // therefore, try to parse 8 digits at a time, and multiply by the largest
  // scalar value (9 or 19 digits) for each step.
  size_t counter = 0;
  digits = 0;
  limb value = 0;
#ifdef FASTFLOAT_64BIT_LIMB
  size_t step = 19;
#else
  size_t step = 9;
#endif

  // process all integer digits.
  const char* p = num.integer.ptr;
  const char* pend = p + num.integer.len();
  skip_zeros(p, pend);
  // process all digits, in increments of step per loop
  while (p != pend) {
    while ((std::distance(p, pend) >= 8) && (step - counter >= 8) && (max_digits - digits >= 8)) {
      parse_eight_digits(p, value, counter, digits);
    }
    while (counter < step && p != pend && digits < max_digits) {
      parse_one_digit(p, value, counter, digits);
    }
    if (digits == max_digits) {
      // add the temporary value, then check if we've truncated any digits
      add_native(result, limb(powers_of_ten_uint64[counter]), value);
      bool truncated = is_truncated(p, pend);
      if (num.fraction.ptr != nullptr) {
        truncated |= is_truncated(num.fraction);
      }
      if (truncated) {
        round_up_bigint(result, digits);
      }
      return;
    } else {
      add_native(result, limb(powers_of_ten_uint64[counter]), value);
      counter = 0;
      value = 0;
    }
  }

  // add our fraction digits, if they're available.
  if (num.fraction.ptr != nullptr) {
    p = num.fraction.ptr;
    pend = p + num.fraction.len();
    if (digits == 0) {
      skip_zeros(p, pend);
    }
    // process all digits, in increments of step per loop
    while (p != pend) {
      while ((std::distance(p, pend) >= 8) && (step - counter >= 8) && (max_digits - digits >= 8)) {
        parse_eight_digits(p, value, counter, digits);
      }
      while (counter < step && p != pend && digits < max_digits) {
        parse_one_digit(p, value, counter, digits);
      }
      if (digits == max_digits) {
        // add the temporary value, then check if we've truncated any digits
        add_native(result, limb(powers_of_ten_uint64[counter]), value);
        bool truncated = is_truncated(p, pend);
        if (truncated) {
          round_up_bigint(result, digits);
        }
        return;
      } else {
        add_native(result, limb(powers_of_ten_uint64[counter]), value);
        counter = 0;
        value = 0;
      }
    }
  }

  if (counter != 0) {
    add_native(result, limb(powers_of_ten_uint64[counter]), value);
  }
}

template <typename T>
inline adjusted_mantissa positive_digit_comp(bigint& bigmant, int32_t exponent) noexcept {
  FASTFLOAT_ASSERT(bigmant.pow10(uint32_t(exponent)));
  adjusted_mantissa answer;
  bool truncated;
  answer.mantissa = bigmant.hi64(truncated);
  int bias = binary_format<T>::mantissa_explicit_bits() - binary_format<T>::minimum_exponent();
  answer.power2 = bigmant.bit_length() - 64 + bias;

  round<T>(answer, [truncated](adjusted_mantissa& a, int32_t shift) {
    round_nearest_tie_even(a, shift, [truncated](bool is_odd, bool is_halfway, bool is_above) -> bool {
      return is_above || (is_halfway && truncated) || (is_odd && is_halfway);
    });
  });

  return answer;
}

// the scaling here is quite simple: we have, for the real digits `m * 10^e`,
// and for the theoretical digits `n * 2^f`. Since `e` is always negative,
// to scale them identically, we do `n * 2^f * 5^-f`, so we now have `m * 2^e`.
// we then need to scale by `2^(f- e)`, and then the two significant digits
// are of the same magnitude.
template <typename T>
inline adjusted_mantissa negative_digit_comp(bigint& bigmant, adjusted_mantissa am, int32_t exponent) noexcept {
  bigint& real_digits = bigmant;
  int32_t real_exp = exponent;

  // get the value of `b`, rounded down, and get a bigint representation of b+h
  adjusted_mantissa am_b = am;
  // gcc7 buf: use a lambda to remove the noexcept qualifier bug with -Wnoexcept-type.
  round<T>(am_b, [](adjusted_mantissa&a, int32_t shift) { round_down(a, shift); });
  T b;
  to_float(false, am_b, b);
  adjusted_mantissa theor = to_extended_halfway(b);
  bigint theor_digits(theor.mantissa);
  int32_t theor_exp = theor.power2;

  // scale real digits and theor digits to be same power.
  int32_t pow2_exp = theor_exp - real_exp;
  uint32_t pow5_exp = uint32_t(-real_exp);
  if (pow5_exp != 0) {
    FASTFLOAT_ASSERT(theor_digits.pow5(pow5_exp));
  }
  if (pow2_exp > 0) {
    FASTFLOAT_ASSERT(theor_digits.pow2(uint32_t(pow2_exp)));
  } else if (pow2_exp < 0) {
    FASTFLOAT_ASSERT(real_digits.pow2(uint32_t(-pow2_exp)));
  }

  // compare digits, and use it to director rounding
  int ord = real_digits.compare(theor_digits);
  adjusted_mantissa answer = am;
  round<T>(answer, [ord](adjusted_mantissa& a, int32_t shift) {
    round_nearest_tie_even(a, shift, [ord](bool is_odd, bool _, bool __) -> bool {
      (void)_;  // not needed, since we've done our comparison
      (void)__; // not needed, since we've done our comparison
      if (ord > 0) {
        return true;
      } else if (ord < 0) {
        return false;
      } else {
        return is_odd;
      }
    });
  });

  return answer;
}

// parse the significant digits as a big integer to unambiguously round the
// the significant digits. here, we are trying to determine how to round
// an extended float representation close to `b+h`, halfway between `b`
// (the float rounded-down) and `b+u`, the next positive float. this
// algorithm is always correct, and uses one of two approaches. when
// the exponent is positive relative to the significant digits (such as
// 1234), we create a big-integer representation, get the high 64-bits,
// determine if any lower bits are truncated, and use that to direct
// rounding. in case of a negative exponent relative to the significant
// digits (such as 1.2345), we create a theoretical representation of
// `b` as a big-integer type, scaled to the same binary exponent as
// the actual digits. we then compare the big integer representations
// of both, and use that to direct rounding.
template <typename T>
inline adjusted_mantissa digit_comp(parsed_number_string& num, adjusted_mantissa am) noexcept {
  // remove the invalid exponent bias
  am.power2 -= invalid_am_bias;

  int32_t sci_exp = scientific_exponent(num);
  size_t max_digits = binary_format<T>::max_digits();
  size_t digits = 0;
  bigint bigmant;
  parse_mantissa(bigmant, num, max_digits, digits);
  // can't underflow, since digits is at most max_digits.
  int32_t exponent = sci_exp + 1 - int32_t(digits);
  if (exponent >= 0) {
    return positive_digit_comp<T>(bigmant, exponent);
  } else {
    return negative_digit_comp<T>(bigmant, am, exponent);
  }
}

} // namespace fast_float

#endif

#ifndef FASTFLOAT_PARSE_NUMBER_H
#define FASTFLOAT_PARSE_NUMBER_H

namespace fast_float {


namespace detail {
/**
 * Special case +inf, -inf, nan, infinity, -infinity.
 * The case comparisons could be made much faster given that we know that the
 * strings a null-free and fixed.
 **/
template <typename T>
from_chars_result parse_infnan(const char *first, const char *last, T &value)  noexcept  {
  from_chars_result answer;
  answer.ptr = first;
  answer.ec = std::errc(); // be optimistic
  bool minusSign = false;
  if (*first == '-') { // assume first < last, so dereference without checks; C++17 20.19.3.(7.1) explicitly forbids '+' here
      minusSign = true;
      ++first;
  }
  if (last - first >= 3) {
    if (fastfloat_strncasecmp(first, "nan", 3)) {
      answer.ptr = (first += 3);
      value = minusSign ? -std::numeric_limits<T>::quiet_NaN() : std::numeric_limits<T>::quiet_NaN();
      // Check for possible nan(n-char-seq-opt), C++17 20.19.3.7, C11 7.20.1.3.3. At least MSVC produces nan(ind) and nan(snan).
      if(first != last && *first == '(') {
        for(const char* ptr = first + 1; ptr != last; ++ptr) {
          if (*ptr == ')') {
            answer.ptr = ptr + 1; // valid nan(n-char-seq-opt)
            break;
          }
          else if(!(('a' <= *ptr && *ptr <= 'z') || ('A' <= *ptr && *ptr <= 'Z') || ('0' <= *ptr && *ptr <= '9') || *ptr == '_'))
            break; // forbidden char, not nan(n-char-seq-opt)
        }
      }
      return answer;
    }
    if (fastfloat_strncasecmp(first, "inf", 3)) {
      if ((last - first >= 8) && fastfloat_strncasecmp(first + 3, "inity", 5)) {
        answer.ptr = first + 8;
      } else {
        answer.ptr = first + 3;
      }
      value = minusSign ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
      return answer;
    }
  }
  answer.ec = std::errc::invalid_argument;
  return answer;
}

} // namespace detail

template<typename T>
from_chars_result from_chars(const char *first, const char *last,
                             T &value, chars_format fmt /*= chars_format::general*/)  noexcept  {
  return from_chars_advanced(first, last, value, parse_options{fmt});
}

template<typename T>
from_chars_result from_chars_advanced(const char *first, const char *last,
                                      T &value, parse_options options)  noexcept  {

  static_assert (std::is_same<T, double>::value || std::is_same<T, float>::value, "only float and double are supported");


  from_chars_result answer;
  if (first == last) {
    answer.ec = std::errc::invalid_argument;
    answer.ptr = first;
    return answer;
  }
  parsed_number_string pns = parse_number_string(first, last, options);
  if (!pns.valid) {
    return detail::parse_infnan(first, last, value);
  }
  answer.ec = std::errc(); // be optimistic
  answer.ptr = pns.lastmatch;
  // Next is Clinger's fast path.
  if (binary_format<T>::min_exponent_fast_path() <= pns.exponent && pns.exponent <= binary_format<T>::max_exponent_fast_path() && pns.mantissa <=binary_format<T>::max_mantissa_fast_path() && !pns.too_many_digits) {
    value = T(pns.mantissa);
    if (pns.exponent < 0) { value = value / binary_format<T>::exact_power_of_ten(-pns.exponent); }
    else { value = value * binary_format<T>::exact_power_of_ten(pns.exponent); }
    if (pns.negative) { value = -value; }
    return answer;
  }
  adjusted_mantissa am = compute_float<binary_format<T>>(pns.exponent, pns.mantissa);
  if(pns.too_many_digits && am.power2 >= 0) {
    if(am != compute_float<binary_format<T>>(pns.exponent, pns.mantissa + 1)) {
      am = compute_error<binary_format<T>>(pns.exponent, pns.mantissa);
    }
  }
  // If we called compute_float<binary_format<T>>(pns.exponent, pns.mantissa) and we have an invalid power (am.power2 < 0),
  // then we need to go the long way around again. This is very uncommon.
  if(am.power2 < 0) { am = digit_comp<T>(pns, am); }
  to_float(pns.negative, am, value);
  return answer;
}

} // namespace fast_float

#endif

// clang-format on

//
// Third-party includes End
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace rapidobj {
enum class rapidobj_errc {
    Success,
    ParseError,
    MaterialFileError,
    MaterialParseError,
    MaterialNotFoundError,
    MaterialRelativePathError,
    AmbiguousMaterialLibraryError,
    LineTooLongError,
    IndexOutOfBoundsError,
    InvalidArgumentsError,
    TooFewIndicesError,
    TooManyIndicesError,
    TriangulationError,
    InternalError
};
} // namespace rapidobj

namespace std {
template <>
struct is_error_code_enum<rapidobj::rapidobj_errc> : true_type {};
} // namespace std

namespace rapidobj {

struct rapidobj_error_category : std::error_category {
    const char* name() const noexcept override { return "rapidobj"; }
    std::string message(int err) const override
    {
        switch (static_cast<rapidobj_errc>(err)) {
        case rapidobj_errc::Success: return "No error.";
        case rapidobj_errc::ParseError: return "Object file parse error.";
        case rapidobj_errc::MaterialFileError: return "Material file not found.";
        case rapidobj_errc::MaterialParseError: return "Material file parse error.";
        case rapidobj_errc::MaterialNotFoundError: return "Material not found.";
        case rapidobj_errc::MaterialRelativePathError: return "Material file path is relative.";
        case rapidobj_errc::AmbiguousMaterialLibraryError: return "Ambiguous material library.";
        case rapidobj_errc::LineTooLongError: return "Line too long.";
        case rapidobj_errc::IndexOutOfBoundsError: return "Index out of bounds.";
        case rapidobj_errc::InvalidArgumentsError: return "Invalid Arguments.";
        case rapidobj_errc::TooFewIndicesError: return "Polygon has too few indices.";
        case rapidobj_errc::TooManyIndicesError: return "Polygon has too many indices.";
        case rapidobj_errc::TriangulationError: return "Triangulation errror.";
        case rapidobj_errc::InternalError: return "Internal error.";
        }
        return "Unrecognised error.";
    }
};

inline std::error_code make_error_code(rapidobj_errc err)
{
    static const rapidobj_error_category category{};
    return { static_cast<int>(err), category };
}

namespace detail {

constexpr std::size_t operator""_GiB(unsigned long long int gibibytes) noexcept
{
    return static_cast<std::size_t>(1024 * 1024 * 1024 * gibibytes);
}

constexpr std::size_t operator""_MiB(unsigned long long int mebibytes) noexcept
{
    return static_cast<std::size_t>(1024 * 1024 * mebibytes);
}

constexpr std::size_t operator""_KiB(unsigned long long int kibibytes) noexcept
{
    return static_cast<std::size_t>(1024 * kibibytes);
}

static constexpr auto kMaxLineLength      = 4_KiB;
static constexpr auto kBlockSize          = 256_KiB;
static constexpr auto kMinVerticesInFace  = 3;
static constexpr auto kMaxVerticesInFace  = 255;
static constexpr auto kMinVerticesInLine  = 2;
static constexpr auto kMaxVerticesInLine  = 1000;
static constexpr auto kMinVerticesInPoint = 1;
static constexpr auto kMaxVerticesInPoint = 1000;
static constexpr auto kSingleThreadCutoff = 1_MiB;

static constexpr auto kMergeCopyByteCost  = 12;
static constexpr auto kMergeCopyIntCost   = 31;
static constexpr auto kMergeFillIdCost    = 32;
static constexpr auto kMergeCopyIndexCost = 186;
static constexpr auto kMergeSubdivideCost = 50000000;

static constexpr auto kTriangulateTriangleCost  = 55;
static constexpr auto kTriangulateQuadCost      = 140;
static constexpr auto kTriangulatePerIndexCost  = 46;
static constexpr auto kTriangulateSubdivideCost = 5000000;

static constexpr auto kMemoryRecyclingSize = 25_MiB;

static_assert(kMaxLineLength < kBlockSize);
static_assert(kBlockSize % 4_KiB == 0);

static constexpr auto kSuccess = std::errc();

inline constexpr bool StartsWith(std::string_view text, std::string_view sv) noexcept
{
    if (text.length() < sv.length()) {
        return false;
    }
    for (size_t i = 0; i != sv.length(); ++i) {
        if (text[i] != sv[i]) {
            return false;
        }
    }
    return true;
}

inline constexpr bool EndsWith(std::string_view text, char c) noexcept
{
    return text.length() > 0 && text.back() == c;
}

inline void TrimLeft(std::string& text) noexcept
{
    size_t index = 0;
    while (index < text.size() && (text[index] == ' ' || text[index] == '\t')) {
        ++index;
    }
    text.erase(0, index);
}

inline void TrimRight(std::string& text) noexcept
{
    auto index = text.size();
    while (index && (text[index - 1] == ' ' || text[index - 1] == '\t')) {
        --index;
    }
    text.erase(index);
}

inline void Trim(std::string& text) noexcept
{
    TrimLeft(text);
    TrimRight(text);
}

inline constexpr void TrimLeft(std::string_view& text) noexcept
{
    size_t index = 0;
    while (index < text.size() && (text[index] == ' ' || text[index] == '\t')) {
        ++index;
    }
    text.remove_prefix(index);
}

inline constexpr void TrimRight(std::string_view& text) noexcept
{
    auto index = text.size();
    while (index && (text[index - 1] == ' ' || text[index - 1] == '\t')) {
        --index;
    }
    text = text.substr(0, index);
}

inline constexpr void Trim(std::string_view& text) noexcept
{
    TrimLeft(text);
    TrimRight(text);
}

enum class ApplyOffset : uint8_t { None = 0, Position = 1, Texcoord = 2, Normal = 4, All = 7 };

using OffsetFlags = std::underlying_type_t<ApplyOffset>;

inline bool operator&(OffsetFlags lhs, ApplyOffset rhs) noexcept
{
    return lhs & static_cast<OffsetFlags>(rhs);
}

inline OffsetFlags operator|(OffsetFlags lhs, ApplyOffset rhs) noexcept
{
    return lhs | static_cast<OffsetFlags>(rhs);
}

inline OffsetFlags operator|(ApplyOffset lhs, ApplyOffset rhs) noexcept
{
    return static_cast<OffsetFlags>(lhs) | static_cast<OffsetFlags>(rhs);
}

template <typename T>
struct Buffer final {
    Buffer() noexcept = default;
    Buffer(size_t size) noexcept : m_size(size), m_room(0), m_data(new T[size]) {}

    size_t   size() const noexcept { return m_size; }
    const T* data() const noexcept { return m_data.get(); }
    T*       data() noexcept { return m_data.get(); }
    T&       back() noexcept { return m_data[m_size - 1]; }

    void push_back(T value) noexcept
    {
        m_data[m_size] = value;
        ++m_size, --m_room;
    }

    T pop_back() noexcept
    {
        --m_size, ++m_room;
        return m_data.get()[m_size];
    }

    void fill_n(size_t n, T value)
    {
        std::fill_n(m_data.get() + m_size, n, value);
        m_size += n;
        m_room -= n;
    }

    void ensure_enough_room_for(size_t size)
    {
        if (size > m_room) {
            auto cap = std::max(kInitialSize, 2 * (m_size + size));
            auto src = std::unique_ptr<T[]>(std::move(m_data));
            m_data.reset(new T[cap]);
            if (src) {
                memcpy(m_data.get(), src.get(), m_size * sizeof(T));
            }
            m_room = cap - m_size;
        }
    }

  private:
    static constexpr size_t kInitialSize = 4096;

    size_t               m_size{};
    size_t               m_room{};
    std::unique_ptr<T[]> m_data{};

    static_assert(std::is_trivially_copyable_v<T>);
};

struct ShapeRecord final {
    struct Mesh final {
        size_t index_buffer_start{};
        size_t face_buffer_start{};
    };
    struct Lines final {
        size_t index_buffer_start{};
        size_t segment_buffer_start{};
    };
    struct Points final {
        size_t index_buffer_start{};
    };
    std::string name{};
    Mesh        mesh{};
    Lines       lines{};
    Points      points{};
    size_t      chunk_index{};
};

struct MaterialRecord final {
    std::string name;
    std::string line{};
    size_t      line_num{};
    size_t      face_buffer_start{};
};

using MaterialMap = std::map<std::string, int>;

struct ParseMaterialsResult final {
    MaterialMap material_map;
    Materials   materials;
    Error       error;
};

struct SmoothingRecord final {
    unsigned int group_id{};
    size_t       face_buffer_start{};
};

struct SharedContext final {
    struct Thread final {
        size_t concurrency{};
    } thread;

    struct Stats final {
        size_t num_positions{};
        size_t num_texcoords{};
        size_t num_normals{};
    } stats;

    struct Material final {
        const MaterialLibrary*            library{};
        std::filesystem::path             basepath{};
        std::string                       library_name{};
        std::future<ParseMaterialsResult> parse_result{};
        std::mutex                        mutex{}; // protects library_name and parse_result
    } material;

    struct Parsing final {
        std::atomic_size_t thread_count{};
        std::promise<void> completed{};
    } parsing;

    struct Merging final {
        std::atomic_size_t task_index{};
        std::atomic_size_t thread_count{};
        std::promise<void> completed{};
        rapidobj_errc      error{};
        std::mutex         mutex{}; // protects error
    } merging;

    struct Debug final {
        struct IO final {
            std::vector<int>                      num_requests{};
            std::vector<size_t>                   num_bytes_read{};
            std::vector<std::chrono::nanoseconds> submit_time;
            std::vector<std::chrono::nanoseconds> wait_time;
        } io;
        struct Parse final {
            std::vector<std::chrono::nanoseconds> time;
            std::chrono::nanoseconds              total_time;
        } parse;
        struct Merge final {
            std::chrono::nanoseconds total_time;
        } merge;
    } debug;
};

struct Chunk final {
    struct Text final {
        size_t line_count{};
    };
    struct Positions final {
        Buffer<float> buffer{};
        size_t        count{};
    };
    struct Texcoords final {
        Buffer<float> buffer{};
        size_t        count{};
    };
    struct Normals final {
        Buffer<float> buffer{};
        size_t        count{};
    };
    struct Colors final {
        Buffer<float> buffer{};
        size_t        count{};
    };
    struct Mesh final {
        struct Indices final {
            Buffer<Index>       buffer;
            Buffer<OffsetFlags> flags;
        } indices{};
        struct Faces final {
            Buffer<unsigned char> buffer;
            size_t                count;
        } faces{};
    };
    struct Lines final {
        struct Indices final {
            Buffer<Index>       buffer;
            Buffer<OffsetFlags> flags;
        } indices{};
        struct Segments final {
            Buffer<int> buffer;
            size_t      count;
        } segments{};
    };
    struct Points final {
        struct Indices final {
            Buffer<Index>       buffer;
            Buffer<OffsetFlags> flags;
        } indices{};
    };
    struct Shapes final {
        std::vector<ShapeRecord> list;
    };
    struct Materials final {
        std::vector<MaterialRecord> list;
    };
    struct Smoothing final {
        std::vector<SmoothingRecord> list;
    };

    Text      text;
    Positions positions;
    Texcoords texcoords;
    Normals   normals;
    Colors    colors;
    Mesh      mesh;
    Lines     lines;
    Points    points;
    Shapes    shapes;
    Materials materials;
    Smoothing smoothing;
    Error     error;
};

inline size_t SizeInBytes(const Chunk& chunk) noexcept
{
    auto size = size_t{ 0 };

    size += chunk.positions.buffer.size() * sizeof(float);
    size += chunk.texcoords.buffer.size() * sizeof(float);
    size += chunk.normals.buffer.size() * sizeof(float);
    size += chunk.colors.buffer.size() * sizeof(float);
    size += chunk.mesh.indices.buffer.size() * sizeof(Index);
    size += chunk.mesh.indices.flags.size() * sizeof(OffsetFlags);
    size += chunk.mesh.faces.buffer.size() * sizeof(unsigned char);
    size += chunk.lines.indices.buffer.size() * sizeof(Index);
    size += chunk.lines.indices.flags.size() * sizeof(OffsetFlags);
    size += chunk.lines.segments.buffer.size() * sizeof(int);
    size += chunk.points.indices.buffer.size() * sizeof(Index);
    size += chunk.points.indices.flags.size() * sizeof(OffsetFlags);

    return size;
}

inline size_t SizeInBytes(const Mesh& mesh) noexcept
{
    auto size = size_t{ 0 };

    size += mesh.indices.size() * sizeof(Index);
    size += mesh.num_face_vertices.size() * sizeof(uint8_t);
    size += mesh.material_ids.size() * sizeof(int32_t);
    size += mesh.smoothing_group_ids.size() * sizeof(uint32_t);

    return size;
}

template <typename>
struct CopyElements;

template <typename>
struct FillElements;

template <typename>
struct FillIds;

template <typename T>
struct FillSrc final {
    T      id{};
    size_t offset{};
};

struct AttributeInfo final {
    size_t position{};
    size_t texcoord{};
    size_t normal{};

    AttributeInfo& operator+=(const AttributeInfo& rhs) noexcept
    {
        position += rhs.position;
        texcoord += rhs.texcoord;
        normal += rhs.normal;
        return *this;
    }
};

struct CopyIndices;

using CopyBytes  = CopyElements<uint8_t>;
using CopyInts   = CopyElements<int32_t>;
using CopyFloats = CopyElements<float>;

using FillFloats = FillElements<float>;

using FillMaterialIds       = FillIds<int32_t>;
using FillSmoothingGroupIds = FillIds<uint32_t>;

template <typename T>
struct CopyElements final {
    CopyElements(T* dst, const T* src, size_t size) noexcept : m_dst(dst), m_src(src), m_size(size) {}

    auto Cost() const noexcept
    {
        constexpr auto cost = sizeof(T) == 1 ? kMergeCopyByteCost : kMergeCopyIntCost;
        return cost * m_size;
    }

    auto Execute() const noexcept
    {
        memcpy(m_dst, m_src, m_size * sizeof(T));
        return rapidobj_errc::Success;
    }

    inline auto Subdivide(size_t num) const noexcept;

  private:
    T*       m_dst{};
    const T* m_src;
    size_t   m_size;
};

template <typename T>
struct FillElements final {
    FillElements(T* dst, T value, size_t size) noexcept : m_dst(dst), m_value(value), m_size(size) {}

    auto Cost() const noexcept
    {
        constexpr auto cost = sizeof(T) == 1 ? kMergeCopyByteCost : kMergeCopyIntCost;
        return cost * m_size;
    }

    auto Execute() const noexcept
    {
        std::fill_n(m_dst, m_size, m_value);
        return rapidobj_errc::Success;
    }

    inline auto Subdivide(size_t num) const noexcept;

  private:
    T*      m_dst{};
    const T m_value;
    size_t  m_size;
};

template <typename T>
struct FillIds final {
    FillIds(T* dst, const std::vector<FillSrc<T>>& src, size_t size, size_t start) noexcept
        : m_dst(dst), m_src(&src), m_size(size), m_start(start)
    {}

    auto Cost() const noexcept { return kMergeFillIdCost * m_size; }

    auto Execute() const noexcept
    {
        auto comp = [](auto src, size_t offset) { return src.offset < offset; };
        auto it   = std::lower_bound(m_src->begin(), m_src->end(), m_start, comp);
        if (it == m_src->end()) {
            return rapidobj_errc::InternalError;
        }

        auto index = static_cast<size_t>(it - m_src->begin());
        if (m_src->at(index).offset != m_start) {
            --index;
        }

        auto dst   = m_dst;
        auto size  = m_size;
        auto start = m_start;

        while (size > 0 && index < m_src->size() - 1) {
            auto count = std::min(size, m_src->at(index + 1).offset - start);
            std::fill_n(dst, count, m_src->at(index).id);
            start += count;
            dst += count;
            size -= count;
            ++index;
        }

        return rapidobj_errc::Success;
    }

    inline auto Subdivide(size_t num) const noexcept;

  private:
    T*                             m_dst{};
    const std::vector<FillSrc<T>>* m_src{};
    size_t                         m_size{};
    size_t                         m_start{};
};

struct CopyIndices final {
    CopyIndices(
        Index*             dst,
        const Index*       src,
        const OffsetFlags* offset_flags,
        size_t             size,
        AttributeInfo      offset,
        AttributeInfo      count) noexcept
        : m_dst(dst), m_src(src), m_offset_flags(offset_flags), m_size(size), m_offset(offset), m_count(count)
    {}

    auto Cost() const noexcept { return kMergeCopyIndexCost * m_size; }

    auto Execute() const noexcept
    {
        for (size_t i = 0; i != m_size; ++i) {
            auto position_index = m_src[i].position_index;
            auto texcoord_index = m_src[i].texcoord_index;
            auto normal_index   = m_src[i].normal_index;
            auto offset_flags   = m_offset_flags[i];

            bool is_out_of_bounds = false;

            if (offset_flags & ApplyOffset::Position) {
                position_index += static_cast<int>(m_offset.position);
            }
            is_out_of_bounds |= position_index < 0 || position_index >= static_cast<int>(m_count.position);

            if (offset_flags & ApplyOffset::Texcoord) {
                texcoord_index += static_cast<int>(m_offset.texcoord);
                is_out_of_bounds |= texcoord_index < 0;
            } else {
                is_out_of_bounds |= texcoord_index < -1;
            }
            is_out_of_bounds |= texcoord_index >= static_cast<int>(m_count.texcoord);

            if (offset_flags & ApplyOffset::Normal) {
                normal_index += static_cast<int>(m_offset.normal);
                is_out_of_bounds |= normal_index < 0;
            } else {
                is_out_of_bounds |= normal_index < -1;
            }
            is_out_of_bounds |= normal_index >= static_cast<int>(m_count.normal);

            if (is_out_of_bounds) {
                return rapidobj_errc::IndexOutOfBoundsError;
            }

            m_dst[i].position_index = position_index;
            m_dst[i].texcoord_index = texcoord_index;
            m_dst[i].normal_index   = normal_index;
        }

        return rapidobj_errc::Success;
    }

    inline auto Subdivide(size_t num) const noexcept;

  private:
    Index*             m_dst{};
    const Index*       m_src{};
    const OffsetFlags* m_offset_flags{};
    size_t             m_size{};
    AttributeInfo      m_offset{};
    AttributeInfo      m_count{};
};

using MergeTask =
    std::variant<CopyBytes, CopyInts, CopyFloats, CopyIndices, FillFloats, FillMaterialIds, FillSmoothingGroupIds>;
using MergeTasks = std::vector<MergeTask>;

template <typename T>
auto CopyElements<T>::Subdivide(size_t num) const noexcept
{
    auto begin = size_t{ 0 };
    auto tasks = MergeTasks();
    tasks.reserve(num);
    for (size_t i = 0; i != num; ++i) {
        auto end = (1 + i) * m_size / num;
        tasks.push_back(CopyElements(m_dst + begin, m_src + begin, end - begin));
        begin = end;
    }
    return tasks;
}

template <typename T>
auto FillElements<T>::Subdivide(size_t num) const noexcept
{
    auto begin = size_t{ 0 };
    auto tasks = MergeTasks();
    tasks.reserve(num);
    for (size_t i = 0; i != num; ++i) {
        auto end = (1 + i) * m_size / num;
        tasks.push_back(FillElements(m_dst + begin, m_value, end - begin));
        begin = end;
    }
    return tasks;
}

template <typename T>
auto FillIds<T>::Subdivide(size_t num) const noexcept
{
    auto begin = size_t{ 0 };
    auto tasks = MergeTasks();
    tasks.reserve(num);
    for (size_t i = 0; i != num; ++i) {
        auto end = (1 + i) * m_size / num;
        tasks.push_back(FillIds(m_dst + begin, *m_src, end - begin, m_start + begin));
        begin = end;
    }
    return tasks;
}

auto CopyIndices::Subdivide(size_t num) const noexcept
{
    auto begin = size_t{ 0 };
    auto tasks = MergeTasks();
    tasks.reserve(num);
    for (size_t i = 0; i != num; ++i) {
        auto end = (1 + i) * m_size / num;
        tasks.push_back(CopyIndices(m_dst + begin, m_src + begin, m_offset_flags, end - begin, m_offset, m_count));
        begin = end;
    }
    return tasks;
}

struct Reader {
    struct ReadResult final {
        std::size_t     bytes_read;
        std::error_code error_code;
    };

    virtual ~Reader() noexcept = default;

    virtual std::error_code ReadBlock(size_t offset, size_t size, char* buffer) = 0;
    virtual ReadResult      WaitForResult()                                     = 0;

    auto NumRequests() const noexcept { return m_num_requests; }
    auto SubmitTime() const noexcept { return m_submit_time; }
    auto WaitTime() const noexcept { return m_wait_time; }
    auto BytesRead() const noexcept { return m_bytes_read; }
    auto Error() const noexcept { return m_error; }

  protected:
    int                      m_num_requests{};
    std::chrono::nanoseconds m_submit_time{};
    std::chrono::nanoseconds m_wait_time{};
    size_t                   m_bytes_read{};
    std::error_code          m_error{};
};

namespace sys {

#ifdef __linux__

inline auto AlignedAllocate(size_t size, size_t alignment)
{
    return static_cast<char*>(aligned_alloc(alignment, size));
}

struct AlignedDeleter final {
    void operator()(void* ptr) const { free(ptr); }
};

class File final {
  public:
    File(const std::filesystem::path& filepath)
    {
        auto filepath_string = filepath.string();

        if (-1 == stat(filepath_string.c_str(), &m_info)) {
            m_error = std::error_code(errno, std::system_category());
            return;
        }

        m_fd = open(filepath_string.c_str(), O_RDONLY);

        if (-1 == m_fd) {
            m_error = std::error_code(errno, std::system_category());
        }
    }
    File(const File&)            = delete;
    File& operator=(const File&) = delete;
    File(File&&)                 = delete;
    File& operator=(File&&)      = delete;
    ~File() noexcept
    {
        if (m_fd != -1) {
            close(m_fd);
        }
    }

    explicit operator bool() const noexcept { return m_fd != -1; }
    auto     handle() const noexcept { return m_fd; }
    auto     size() const noexcept { return static_cast<size_t>(m_info.st_size); }
    auto     error() const noexcept { return m_error; }

  private:
    int             m_fd = -1;
    struct stat     m_info {};
    std::error_code m_error{};
};

struct FileReader : Reader {
    FileReader(const File& file) noexcept : m_fd{ file.handle() } {}

    std::error_code ReadBlock(size_t offset, size_t size, char* buffer) override
    {
        assert(buffer);
        assert(m_fd != -1);
        assert(std::uintptr_t(buffer) % 4096 == 0);

        ++m_num_requests;

        auto t1 = std::chrono::steady_clock::now();

        m_offset = offset;
        m_size   = size;
        m_buffer = buffer;

        readahead(m_fd, offset, size);

        auto t2 = std::chrono::steady_clock::now();

        m_submit_time += t2 - t1;

        return {};
    }

    ReadResult WaitForResult() override
    {
        auto t1 = std::chrono::steady_clock::now();

        auto result = pread64(m_fd, m_buffer, m_size, m_offset);

        auto t2 = std::chrono::steady_clock::now();

        m_wait_time += t2 - t1;

        if (result < 0) {
            auto error = std::error_code(errno, std::system_category());
            return ReadResult{ 0, error };
        }

        auto bytes_read = static_cast<size_t>(result);

        m_bytes_read += bytes_read;

        return ReadResult{ bytes_read, std::error_code() };
    }

  private:
    int    m_fd = -1;
    size_t m_offset{};
    size_t m_size{};
    char*  m_buffer{};
};

#elif _WIN32

inline auto AlignedAllocate(size_t size, size_t alignment)
{
    return static_cast<char*>(_aligned_malloc(size, alignment));
}

struct AlignedDeleter final {
    void operator()(void* ptr) const { _aligned_free(ptr); }
};

class File final {
  public:
    File(const std::filesystem::path& filepath)
    {
        auto filepath_string = filepath.string();

        m_handle = CreateFileA(
            filepath_string.c_str(),
            GENERIC_READ,
            0,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_READONLY | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr);

        if (m_handle == INVALID_HANDLE_VALUE) {
            m_error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
            return;
        }

        auto size = LARGE_INTEGER{};
        if (!GetFileSizeEx(m_handle, &size)) {
            m_error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
            Close();
            return;
        }

        m_size = static_cast<size_t>(size.QuadPart);
    }
    File(const File&)            = delete;
    File& operator=(const File&) = delete;
    File(File&&)                 = delete;
    File& operator=(File&&)      = delete;
    ~File() noexcept { Close(); }

    explicit operator bool() const noexcept { return m_handle != nullptr && m_handle != INVALID_HANDLE_VALUE; }
    auto     handle() const noexcept { return m_handle; }
    auto     size() const noexcept { return m_size; }
    auto     error() const noexcept { return m_error; }

  private:
    void Close() noexcept
    {
        if (m_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(m_handle);
            m_handle = INVALID_HANDLE_VALUE;
        }
    }

    HANDLE          m_handle{};
    size_t          m_size{};
    std::error_code m_error{};
};

struct FileReader : Reader {
    FileReader(const File& file)
    {
        m_handle = CreateEventA(nullptr, FALSE, FALSE, nullptr);

        if (m_handle == INVALID_HANDLE_VALUE) {
            m_error = std::error_code(static_cast<int>(GetLastError()), std::system_category());
        }

        m_file = file.handle();
    }

    ~FileReader() noexcept override
    {
        if (m_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(m_handle);
        }
    }

    std::error_code ReadBlock(size_t offset, size_t size, char* buffer) override
    {
        assert(m_handle && m_handle != INVALID_HANDLE_VALUE);
        assert(m_file && m_file != INVALID_HANDLE_VALUE);

        ++m_num_requests;

        auto t1 = std::chrono::steady_clock::now();

        m_overlapped.hEvent     = m_handle;
        m_overlapped.Offset     = static_cast<DWORD>(offset);
        m_overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);

        auto error   = std::error_code();
        bool success = ReadFile(m_file, buffer, static_cast<DWORD>(size), nullptr, &m_overlapped);

        if (!success) {
            if (auto ec = GetLastError(); ec != ERROR_IO_PENDING) {
                error = std::error_code(ec, std::system_category());
            }
        }

        auto t2 = std::chrono::steady_clock::now();

        m_submit_time += t2 - t1;

        return error;
    }

    ReadResult WaitForResult() override
    {
        assert(m_handle && m_handle != INVALID_HANDLE_VALUE);
        assert(m_file && m_file != INVALID_HANDLE_VALUE);

        auto t1 = std::chrono::steady_clock::now();

        auto bytes_read = DWORD{};
        bool success    = GetOverlappedResult(m_handle, &m_overlapped, &bytes_read, TRUE);

        auto t2 = std::chrono::steady_clock::now();

        m_wait_time += t2 - t1;

        if (!success) {
            if (auto ec = GetLastError(); ec != ERROR_HANDLE_EOF) {
                auto error = std::error_code(ec, std::system_category());
                return ReadResult{ 0, error };
            }
        }

        m_bytes_read += bytes_read;

        return { bytes_read, std::error_code() };
    }

  private:
    HANDLE     m_handle{};
    HANDLE     m_file{};
    OVERLAPPED m_overlapped{};
};

#elif __APPLE__

inline auto AlignedAllocate(size_t size, size_t alignment)
{
    return static_cast<char*>(aligned_alloc(alignment, size));
}

struct AlignedDeleter final {
    void operator()(void* ptr) const { free(ptr); }
};

class File final {
  public:
    File(const std::filesystem::path& filepath)
    {
        auto filepath_string = filepath.string();

        if (-1 == stat(filepath_string.c_str(), &m_info)) {
            m_error = std::error_code(errno, std::system_category());
            return;
        }

        m_fd = open(filepath_string.c_str(), O_RDONLY);

        if (-1 == m_fd) {
            m_error = std::error_code(errno, std::system_category());
        }
    }
    File(const File&)            = delete;
    File& operator=(const File&) = delete;
    File(File&&)                 = delete;
    File& operator=(File&&)      = delete;
    ~File() noexcept
    {
        if (m_fd != -1) {
            close(m_fd);
        }
    }

    explicit operator bool() const noexcept { return m_fd != -1; }
    auto     handle() const noexcept { return m_fd; }
    auto     size() const noexcept { return static_cast<size_t>(m_info.st_size); }
    auto     error() const noexcept { return m_error; }

  private:
    int             m_fd = -1;
    struct stat     m_info {};
    std::error_code m_error{};
};

struct FileReader : Reader {
    FileReader(const File& file) noexcept : m_fd{ file.handle() }
    {
        assert(m_fd != -1);

        fcntl(m_fd, F_NOCACHE, 1);
    }

    std::error_code ReadBlock(size_t offset, size_t size, char* buffer) override
    {
        assert(buffer);
        assert(std::uintptr_t(buffer) % 4096 == 0);

        ++m_num_requests;

        auto t1 = std::chrono::steady_clock::now();

        auto error = std::error_code();

        m_offset = offset;
        m_size   = size;
        m_buffer = buffer;

        radvisory args{};
        args.ra_offset = static_cast<off_t>(offset);
        args.ra_count  = static_cast<int>(size);

        auto result = fcntl(m_fd, F_RDADVISE, &args);

        if (result == -1) {
            error = std::error_code(errno, std::system_category());
        }

        auto t2 = std::chrono::steady_clock::now();

        m_submit_time += t2 - t1;

        return error;
    }

    ReadResult WaitForResult() override
    {
        auto t1 = std::chrono::steady_clock::now();

        auto result = pread(m_fd, m_buffer, m_size, m_offset);

        auto t2 = std::chrono::steady_clock::now();

        m_wait_time += t2 - t1;

        if (result < 0) {
            auto error = std::error_code(errno, std::system_category());
            return ReadResult{ 0, error };
        }

        auto bytes_read = static_cast<size_t>(result);

        m_bytes_read += bytes_read;

        return ReadResult{ bytes_read, std::error_code() };
    }

  private:
    int    m_fd = -1;
    off_t  m_offset{};
    size_t m_size{};
    char*  m_buffer{};
};

#endif

} // namespace sys

using DataSource = std::variant<sys::File*, std::istream*>;

struct StreamReader : Reader {
    StreamReader(std::istream* is) noexcept : m_stream(is)
    {
        assert(is);
        if (!is->good()) {
            m_error = std::make_error_code(std::io_errc::stream);
        }
    }

    std::error_code ReadBlock(size_t offset, size_t size, char* buffer) override
    {
        assert(buffer);

        ++m_num_requests;

        m_offset = offset;
        m_size   = size;
        m_buffer = buffer;

        return std::error_code();
    }

    ReadResult WaitForResult() override
    {
        ReadResult result{};

        auto t1 = std::chrono::steady_clock::now();

        if (m_stream->read(m_buffer, m_size) || m_stream->eof()) {
            result = { static_cast<size_t>(m_stream->gcount()), std::error_code() };
        } else {
            result = { size_t{}, std::make_error_code(std::io_errc::stream) };
        }

        auto t2 = std::chrono::steady_clock::now();

        m_wait_time += t2 - t1;

        m_bytes_read += result.bytes_read;

        return result;
    }

  private:
    std::istream* m_stream{};
    size_t        m_offset{};
    size_t        m_size{};
    char*         m_buffer{};
};

inline std::unique_ptr<Reader> CreateReader(sys::File* file)
{
    return std::make_unique<sys::FileReader>(*file);
}
inline std::unique_ptr<Reader> CreateReader(std::istream* is)
{
    return std::make_unique<StreamReader>(is);
}
inline std::unique_ptr<Reader> CreateReader(DataSource source)
{
    return std::visit([](auto arg) { return CreateReader(arg); }, source);
}

inline std::string ToString(std::chrono::nanoseconds time, int width = 0)
{
    auto text = std::string();
    if (time.count() > 19'000'000'000) {
        text = std::to_string(time.count() / 1'000'000'000);
        text.append(" s");
    } else if (time.count() > 19'000'000) {
        text = std::to_string(time.count() / 1'000'000);
        text.append(" ms");
    } else if (time.count() > 19'000) {
        text = std::to_string(time.count() / 1'000);
        text.append(" us");
    } else {
        text = std::to_string(time.count());
        text.append(" ns");
    }
    auto n = width - static_cast<int>(text.size());
    if (n > 0) {
        text.insert(0, n, ' ');
    }
    return text;
}

inline std::string ToString(size_t i, int width = 0)
{
    auto text = std::to_string(i);
    auto n    = width - static_cast<int>(text.size());
    if (n > 0) {
        text.insert(0, n, ' ');
    }
    return text;
}

inline std::string RateToString(double bytes_per_second, int width = 0)
{
    const char* unit = " B/s";

    if (bytes_per_second > 99_GiB) {
        bytes_per_second /= 1_GiB;
        unit = " GB/s";
    } else if (bytes_per_second > 99_MiB) {
        bytes_per_second /= 1_MiB;
        unit = " MB/s";
    } else if (bytes_per_second > 99_KiB) {
        bytes_per_second /= 1_KiB;
        unit = " KB/s";
    }

    auto text = std::to_string(static_cast<int>(bytes_per_second));
    text.append(unit);
    auto n = width - static_cast<int>(text.size());
    if (n > 0) {
        text.insert(0, n, ' ');
    }
    return text;
}

inline std::string ComputeDebugStats(const std::vector<std::chrono::nanoseconds>& population) noexcept
{
    auto text = std::string();

    auto min_time  = std::chrono::nanoseconds{ std::chrono::nanoseconds::max() };
    auto max_time  = std::chrono::nanoseconds{ std::chrono::nanoseconds::min() };
    auto min_index = size_t{};
    auto max_index = size_t{};
    auto avg_time  = std::chrono::nanoseconds{};

    for (size_t i = 0; i != population.size(); ++i) {
        auto sample = population[i];
        if (sample < min_time) {
            min_time  = sample;
            min_index = i;
        }
        if (sample > max_time) {
            max_time  = sample;
            max_index = i;
        }
        avg_time += sample;
    }

    avg_time /= population.size();

    int64_t sq_sum{};

    for (size_t i = 0; i != population.size(); ++i) {
        auto sample = population[i];
        auto diff   = (sample - avg_time).count();
        sq_sum += diff * diff;
    }

    auto std_dev = std::chrono::nanoseconds(static_cast<int64_t>(std::sqrt(sq_sum / population.size())));

    text.append("min: ").append(ToString(min_time, 10));
    text.append(" [#").append(std::to_string(min_index + 1)).append("]\n");
    text.append("max: ").append(ToString(max_time, 10));
    text.append(" [#").append(std::to_string(max_index + 1)).append("]\n");
    text.append("avg: ").append(ToString(avg_time, 10)).append("\n");
    text.append("std: ").append(ToString(std_dev, 10)).append("\n");

    return text;
}

inline std::string DumpDebug(const SharedContext& context)
{
    auto text = std::string();

    auto population = std::vector<std::chrono::nanoseconds>(context.debug.io.num_requests.size());

    auto num_bytes_read = size_t{};

    for (size_t i = 0; i != context.debug.io.num_requests.size(); ++i) {
        auto n          = context.debug.io.num_requests[i];
        auto submit_ns  = context.debug.io.submit_time[i];
        auto wait_ns    = context.debug.io.wait_time[i];
        auto total_ns   = submit_ns.count() + wait_ns.count();
        auto average_ns = total_ns / n;
        auto thread     = ToString(i + 1, 3);
        auto requests   = ToString(n, 5);
        auto submit     = ToString(std::chrono::nanoseconds(submit_ns), 9);
        auto wait       = ToString(std::chrono::nanoseconds(wait_ns), 9);
        auto total      = ToString(std::chrono::nanoseconds(total_ns), 9);
        auto average    = ToString(std::chrono::nanoseconds(average_ns), 9);

        population[i] = std::chrono::nanoseconds{ total_ns };

        num_bytes_read += context.debug.io.num_bytes_read[i];

        text.append(thread);
        text.append(": blk").append(requests);
        text.append("    read").append(submit);
        text.append("    wait").append(wait);
        text.append("    sum").append(total);
        text.append("    avg").append(average);
        text.push_back('\n');
    }

    text.push_back('\n');
    text.append(ComputeDebugStats(population));
    text.push_back('\n');

    for (size_t i = 0; i != context.debug.io.num_requests.size(); ++i) {
        auto parse_ns      = context.debug.parse.time[i];
        auto io_ns         = context.debug.io.submit_time[i] + context.debug.io.wait_time[i];
        auto io_percentage = static_cast<int>(0.5f + 100.0f * io_ns.count() / parse_ns.count());
        auto thread        = ToString(i + 1, 3);
        auto parse         = ToString(parse_ns, 9);

        population[i] = parse_ns;

        text.append(thread).append(": parse").append(parse);
        text.append("  (").append(std::to_string(io_percentage)).append("% io)\n");
    }

    text.push_back('\n');
    text.append(ComputeDebugStats(population));
    text.push_back('\n');

    auto parse_total      = context.debug.parse.total_time;
    auto merge_total      = context.debug.merge.total_time;
    auto total            = parse_total + merge_total;
    auto parse_percentage = static_cast<int>(0.5f + 100.0f * parse_total.count() / total.count());
    auto merge_percentage = 100 - parse_percentage;
    auto bytes_per_second = num_bytes_read / (parse_total.count() / 1000'000'000.0);

    text.append("Parse Time: ");
    text.append(ToString(parse_total, 10));
    text.append(" (").append(std::to_string(parse_percentage)).append("%)\n");
    text.append("Merge Time: ");
    text.append(ToString(merge_total, 10));
    text.append(" (").append(std::to_string(merge_percentage)).append("%)\n");
    text.append("Parse Rate: ").append(RateToString(bytes_per_second, 12)).append("\n");

    return text;
}

inline auto ParseXReals(std::string_view line, size_t max_count, float* out)
{
    size_t count = 0;
    while (!line.empty() && count < max_count) {
        TrimLeft(line);
        auto [ptr, rc] = fast_float::from_chars(line.data(), line.data() + line.size(), *out);
        if (rc != kSuccess) {
            return std::make_pair(count, line);
        }
        auto num_parsed = static_cast<size_t>(ptr - line.data());
        line.remove_prefix(num_parsed);
        ++count;
        ++out;
    }
    return std::make_pair(count, line);
}

inline auto ParseXReals(std::string_view line, size_t max_count, Buffer<float>* out)
{
    size_t count = 0;
    out->ensure_enough_room_for(max_count);
    while (!line.empty() && count < max_count) {
        TrimLeft(line);
        auto value     = float();
        auto [ptr, rc] = fast_float::from_chars(line.data(), line.data() + line.size(), value);
        if (rc != kSuccess) {
            return std::make_pair(count, line);
        }
        auto num_parsed = static_cast<size_t>(ptr - line.data());
        out->push_back(value);
        line.remove_prefix(num_parsed);
        ++count;
    }
    return std::make_pair(count, line);
}

inline auto ParseReal(std::string_view line, float* out1)
{
    return ParseXReals(line, 1, out1);
}

inline auto ParseReals(std::string_view line, float* out1, float* out2)
{
    float temp[2];
    auto  result = ParseXReals(line, 2, temp);
    *out1        = result.first > 0 ? temp[0] : *out1;
    *out2        = result.first > 1 ? temp[1] : *out2;
    return result;
}

inline auto ParseReals(std::string_view line, float* out1, float* out2, float* out3)
{
    float temp[3];
    auto  result = ParseXReals(line, 3, temp);
    *out1        = result.first > 0 ? temp[0] : *out1;
    *out2        = result.first > 1 ? temp[1] : *out2;
    *out3        = result.first > 2 ? temp[2] : *out3;
    return result;
}

inline size_t ParseReals(std::string_view text, size_t max_count, float* out)
{
    auto count = size_t{};

    TrimLeft(text);

    while (!text.empty() && count < max_count) {
        auto [ptr, rc] = fast_float::from_chars(text.data(), text.data() + text.size(), *out);
        if (rc != kSuccess) {
            return 0;
        }
        auto num_parsed = static_cast<size_t>(ptr - text.data());
        text.remove_prefix(num_parsed);

        TrimLeft(text);

        ++count;
        ++out;
    }

    TrimLeft(text);

    return text.empty() ? count : 0;
}

inline size_t ParseReals(std::string_view text, size_t max_count, Float3* out)
{
    return ParseReals(text, max_count, out->data());
}

inline size_t ParseReals(std::string_view text, size_t max_count, Buffer<float>* out)
{
    auto count = size_t{};

    out->ensure_enough_room_for(max_count);

    TrimLeft(text);

    while (!text.empty() && count < max_count) {
        auto value     = float();
        auto [ptr, rc] = fast_float::from_chars(text.data(), text.data() + text.size(), value);
        if (rc != kSuccess) {
            return 0;
        }
        auto num_parsed = static_cast<size_t>(ptr - text.data());
        out->push_back(value);
        text.remove_prefix(num_parsed);

        TrimLeft(text);

        ++count;
    }

    TrimLeft(text);

    return text.empty() ? count : 0;
}

inline auto ParseFace(
    std::string_view     text,
    size_t               position_count,
    size_t               texcoord_count,
    size_t               normal_count,
    size_t               min_count,
    size_t               max_count,
    OffsetFlags          permitted_flags,
    Buffer<Index>*       indices,
    Buffer<OffsetFlags>* offset_flags)
{
    using std::make_pair;

    auto count = size_t{};
    auto value = 0;

    indices->ensure_enough_room_for(max_count);
    offset_flags->ensure_enough_room_for(max_count);

    while (count <= max_count) {
        TrimLeft(text);

        // exit if there is nothing left to process
        if (text.empty()) {
            break;
        }

        // parse position index
        {
            auto [ptr, rc] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (rc != kSuccess) {
                return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
            }
            auto num_parsed = static_cast<size_t>(ptr - text.data());
            text.remove_prefix(num_parsed);

            offset_flags->push_back(static_cast<OffsetFlags>(ApplyOffset::None));
            if (value > 0) {
                --value;
            } else if (value < 0) {
                value += static_cast<int>(position_count);
                offset_flags->back() |= static_cast<OffsetFlags>(ApplyOffset::Position);
            } else {
                return make_pair(size_t{ 0 }, rapidobj_errc::IndexOutOfBoundsError);
            }
            if (permitted_flags & ApplyOffset::Position) {
                indices->push_back({ value, -1, -1 });
                ++count;
            } else {
                return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
            }
        }

        // exit if there is nothing left to process
        if (text.empty()) {
            break;
        }

        // Parse UV
        if (text.front() == '/') {
            text.remove_prefix(1);
            if (text.empty()) {
                return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
            }
            if (text.front() != '/') {
                // parse texcoord index
                auto [ptr, rc] = std::from_chars(text.data(), text.data() + text.size(), value);
                if (rc != kSuccess) {
                    return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
                }
                auto num_parsed = static_cast<size_t>(ptr - text.data());
                text.remove_prefix(num_parsed);

                if (value > 0) {
                    --value;
                } else if (value < 0) {
                    value += static_cast<int>(texcoord_count);
                    offset_flags->back() |= static_cast<OffsetFlags>(ApplyOffset::Texcoord);
                } else {
                    return make_pair(size_t{ 0 }, rapidobj_errc::IndexOutOfBoundsError);
                }
                if (permitted_flags & ApplyOffset::Texcoord) {
                    indices->back().texcoord_index = value;
                } else {
                    return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
                }
            }
        }

        // exit if there is nothing left to process
        if (text.empty()) {
            break;
        }

        // Parse Normal
        if (text.front() == '/') {
            text.remove_prefix(1);
            if (text.empty()) {
                return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
            }

            // parse normal index
            auto [ptr, rc] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (rc != kSuccess) {
                return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
            }
            auto num_parsed = static_cast<size_t>(ptr - text.data());
            text.remove_prefix(num_parsed);

            if (value > 0) {
                --value;
            } else if (value < 0) {
                value += static_cast<int>(normal_count);
                offset_flags->back() |= static_cast<OffsetFlags>(ApplyOffset::Normal);
            } else {
                return make_pair(size_t{ 0 }, rapidobj_errc::IndexOutOfBoundsError);
            }
            if (permitted_flags & ApplyOffset::Normal) {
                indices->back().normal_index = value;
            } else {
                return make_pair(size_t{ 0 }, rapidobj_errc::ParseError);
            }
        }
    }

    if (count < min_count) {
        return make_pair(size_t{ 0 }, rapidobj_errc::TooFewIndicesError);
    }
    if (count > max_count) {
        return make_pair(size_t{ 0 }, rapidobj_errc::TooManyIndicesError);
    }

    return make_pair(count, rapidobj_errc::Success);
}

inline auto ParseTextureOption(std::string_view line, TextureOption* texture_option)
{
    assert(texture_option);

    const auto fail = std::make_pair(std::string_view(), false);

    TrimRight(line);

    while (true) {
        TrimLeft(line);
        if (line.empty()) {
            break;
        }
        if (line.front() != '-') {
            return std::make_pair(line, true);
        } else {
            line.remove_prefix(1);
            if (StartsWith(line, "blendu ") || StartsWith(line, "blendu\t")) {
                line.remove_prefix(7);
                TrimLeft(line);
                if (StartsWith(line, "on")) {
                    line.remove_prefix(2);
                    texture_option->blendu = true;
                } else if (StartsWith(line, "off")) {
                    line.remove_prefix(3);
                    texture_option->blendu = false;
                } else {
                    return fail;
                }
            } else if (StartsWith(line, "blendv ") || StartsWith(line, "blendv\t")) {
                line.remove_prefix(7);
                TrimLeft(line);
                if (StartsWith(line, "on")) {
                    line.remove_prefix(2);
                    texture_option->blendv = true;
                } else if (StartsWith(line, "off")) {
                    line.remove_prefix(3);
                    texture_option->blendv = false;
                } else {
                    return fail;
                }
            } else if (StartsWith(line, "boost ") || StartsWith(line, "boost\t")) {
                line.remove_prefix(6);
                auto [count, remainder] = ParseReal(line, &texture_option->sharpness);
                if (count != 1) {
                    return fail;
                }
                line = remainder;
            } else if (StartsWith(line, "mm ") || StartsWith(line, "mm\t")) {
                line.remove_prefix(3);
                auto [count, remainder] = ParseReals(line, &texture_option->brightness, &texture_option->contrast);
                if (count != 2) {
                    return fail;
                }
                line = remainder;
            } else if (StartsWith(line, "o ") || StartsWith(line, "o\t")) {
                line.remove_prefix(2);
                auto [count, remainder] = ParseReals(
                    line,
                    &texture_option->origin_offset[0],
                    &texture_option->origin_offset[1],
                    &texture_option->origin_offset[2]);
                if (count < 1) {
                    return fail;
                }
                line = remainder;
            } else if (StartsWith(line, "s ") || StartsWith(line, "s\t")) {
                line.remove_prefix(2);
                auto [count, remainder] =
                    ParseReals(line, &texture_option->scale[0], &texture_option->scale[1], &texture_option->scale[2]);
                if (count < 1) {
                    return fail;
                }
                line = remainder;
            } else if (StartsWith(line, "t ") || StartsWith(line, "t\t")) {
                line.remove_prefix(2);
                auto [count, remainder] = ParseReals(
                    line,
                    &texture_option->turbulence[0],
                    &texture_option->turbulence[1],
                    &texture_option->turbulence[2]);
                if (count < 1) {
                    return fail;
                }
                line = remainder;
            } else if (StartsWith(line, "texres ") || StartsWith(line, "texres\t")) {
                line.remove_prefix(7);
                TrimLeft(line);
                auto [ptr, rc] =
                    std::from_chars(line.data(), line.data() + line.size(), texture_option->texture_resolution);
                if (rc != kSuccess) {
                    return fail;
                }
                auto num_parsed = static_cast<size_t>(ptr - line.data());
                line.remove_prefix(num_parsed);
            } else if (StartsWith(line, "clamp ") || StartsWith(line, "clamp\t")) {
                line.remove_prefix(6);
                TrimLeft(line);
                if (StartsWith(line, "on")) {
                    line.remove_prefix(2);
                    texture_option->clamp = true;
                } else if (StartsWith(line, "off")) {
                    line.remove_prefix(3);
                    texture_option->clamp = false;
                } else {
                    return fail;
                }
            } else if (StartsWith(line, "bm ") || StartsWith(line, "bm\t")) {
                line.remove_prefix(3);
                auto [count, remainder] = ParseReal(line, &texture_option->bump_multiplier);
                if (count < 1) {
                    return fail;
                }
                line = remainder;
            } else if (StartsWith(line, "imfchan ") || StartsWith(line, "imfchan\t")) {
                line.remove_prefix(8);
                char c = line.empty() ? '?' : line.front();
                if (c == 'r' || c == 'g' || c == 'b' || c == 'm' || c == 'l' || c == 'z') {
                    texture_option->imfchan = c;
                    line.remove_prefix(1);
                } else {
                    return fail;
                }
            } else if (StartsWith(line, "type ") || StartsWith(line, "type\t")) {
                line.remove_prefix(5);
                if (StartsWith(line, "sphere")) {
                    line.remove_prefix(6);
                    texture_option->type = TextureType::Sphere;
                } else if (StartsWith(line, "cube_top")) {
                    line.remove_prefix(8);
                    texture_option->type = TextureType::CubeTop;
                } else if (StartsWith(line, "cube_bottom")) {
                    line.remove_prefix(11);
                    texture_option->type = TextureType::CubeBottom;
                } else if (StartsWith(line, "cube_front")) {
                    line.remove_prefix(10);
                    texture_option->type = TextureType::CubeFront;
                } else if (StartsWith(line, "cube_back")) {
                    line.remove_prefix(9);
                    texture_option->type = TextureType::CubeBack;
                } else if (StartsWith(line, "cube_left")) {
                    line.remove_prefix(9);
                    texture_option->type = TextureType::CubeLeft;
                } else if (StartsWith(line, "cube_right")) {
                    line.remove_prefix(10);
                    texture_option->type = TextureType::CubeRight;
                } else {
                    return fail;
                }
            } else {
                return fail;
            }
            if (!line.empty() && line.front() != ' ' && line.front() != '\t') {
                return fail;
            }
        }
    }

    return std::make_pair(line, true);
}

inline ParseMaterialsResult ParseMaterials(std::string_view text)
{
    auto material_map = MaterialMap{};
    auto material_id  = 0;
    auto materials    = Materials{};
    auto material     = Material{};

    material.bump_texopt.imfchan = 'l';

    auto line_num = size_t{};

    bool has_dissolve = false;

    while (text.empty() == false) {
        auto line_parsed = false;
        auto line        = std::string_view();
        {
            auto ptr = static_cast<const char*>(memchr(text.data(), '\n', text.size()));
            auto len = static_cast<size_t>(ptr - text.data());
            bool eol = ptr;
            line     = eol ? text.substr(0, len) : text;
            text.remove_prefix(line.size() + eol);
            if (EndsWith(line, '\r')) {
                line.remove_suffix(1);
            }
            ++line_num;
        }
        auto line_clone = line;
        Trim(line);
        if (line.empty()) {
            continue;
        }
        switch (line.front()) {
        case 'n': {
            if (StartsWith(line, "newmtl ")) {
                line.remove_prefix(7);
                Trim(line);
                if (!material.name.empty()) {
                    materials.push_back(std::move(material));
                    material = {};
                }
                material.name                = line;
                material.bump_texopt.imfchan = 'l';
                material_map.try_emplace(std::string(line), material_id++);
                has_dissolve = false;
                line_parsed  = true;
            } else if (StartsWith(line, "norm ") || StartsWith(line, "norm\t")) {
                line.remove_prefix(5);
                auto [remainder, success] = ParseTextureOption(line, &material.normal_texopt);
                material.normal_texname   = remainder;
                line_parsed               = success;
            }
            break;
        }
        case 'K': {
            if (StartsWith(line, "Ka ") || StartsWith(line, "Ka\t")) {
                line.remove_prefix(3);
                line_parsed = 3 == ParseReals(line, 3, &material.ambient);
            } else if (StartsWith(line, "Kd ") || StartsWith(line, "Kd\t")) {
                line.remove_prefix(3);
                line_parsed = 3 == ParseReals(line, 3, &material.diffuse);
            } else if (StartsWith(line, "Ks ") || StartsWith(line, "Ks\t")) {
                line.remove_prefix(3);
                line_parsed = 3 == ParseReals(line, 3, &material.specular);
            } else if (StartsWith(line, "Kt ") || StartsWith(line, "Kt\t")) {
                line.remove_prefix(3);
                line_parsed = 3 == ParseReals(line, 3, &material.transmittance);
            } else if (StartsWith(line, "Ke ") || StartsWith(line, "Ke\t")) {
                line.remove_prefix(3);
                line_parsed = 3 == ParseReals(line, 3, &material.emission);
            } else if (StartsWith(line, "Km ") || StartsWith(line, "Km\t")) {
                // ignore
                line_parsed = true;
            }
            break;
        }
        case 'T': {
            if (StartsWith(line, "Tf ") || StartsWith(line, "Tf\t")) {
                line.remove_prefix(3);
                line_parsed = 3 == ParseReals(line, 3, &material.transmittance);
            } else if (StartsWith(line, "Tr ") || StartsWith(line, "Tr\t")) {
                line.remove_prefix(3);
                auto transparency = 0.0f;
                line_parsed       = 1 == ParseReals(line, 1, &transparency);
                if (!has_dissolve) {
                    material.dissolve = 1.0f - transparency;
                }
            }
            break;
        }
        case 'd': {
            if (StartsWith(line, "disp ") || StartsWith(line, "disp\t")) {
                line.remove_prefix(5);
                auto [remainder, success]     = ParseTextureOption(line, &material.displacement_texopt);
                material.displacement_texname = remainder;
                line_parsed                   = success;
            } else if (StartsWith(line, "d ") || StartsWith(line, "d\t")) {
                line.remove_prefix(2);
                line_parsed  = 1 == ParseReals(line, 1, &material.dissolve);
                has_dissolve = true;
            }
            break;
        }
        case 'a': {
            if (StartsWith(line, "aniso ") || StartsWith(line, "aniso\t")) {
                line.remove_prefix(6);
                line_parsed = 1 == ParseReals(line, 1, &material.anisotropy);
            } else if (StartsWith(line, "anisor ") || StartsWith(line, "anisor\t")) {
                line.remove_prefix(7);
                line_parsed = 1 == ParseReals(line, 1, &material.anisotropy_rotation);
            }
            break;
        }
        case 'i': {
            if (StartsWith(line, "illum ") || StartsWith(line, "illum\t")) {
                line.remove_prefix(6);
                TrimLeft(line);
                auto [ptr, rc] = std::from_chars(line.data(), line.data() + line.size(), material.illum);
                if (rc == kSuccess) {
                    auto num_parsed = static_cast<size_t>(ptr - line.data());
                    line.remove_prefix(num_parsed);
                    Trim(line);
                    line_parsed = line.empty();
                }
            }
            break;
        }
        case 'b': {
            if (StartsWith(line, "bump ") || StartsWith(line, "bump\t")) {
                line.remove_prefix(5);
                auto [remainder, success] = ParseTextureOption(line, &material.bump_texopt);
                material.bump_texname     = remainder;
                line_parsed               = success;
            }
            break;
        }
        case 'r': {
            if (StartsWith(line, "refl ") || StartsWith(line, "refl\t")) {
                line.remove_prefix(5);
                auto [remainder, success]   = ParseTextureOption(line, &material.reflection_texopt);
                material.reflection_texname = remainder;
                line_parsed                 = success;
            }
            break;
        }
        case 'm': {
            if (StartsWith(line, "map_Ka ") || StartsWith(line, "map_Ka\t")) {
                line.remove_prefix(7);
                auto [remainder, success] = ParseTextureOption(line, &material.ambient_texopt);
                material.ambient_texname  = remainder;
                line_parsed               = success;
            } else if (StartsWith(line, "map_Kd ") || StartsWith(line, "map_Kd\t")) {
                line.remove_prefix(7);
                auto [remainder, success] = ParseTextureOption(line, &material.diffuse_texopt);
                material.diffuse_texname  = remainder;
                line_parsed               = success;
            } else if (StartsWith(line, "map_Ks ") || StartsWith(line, "map_Ks\t")) {
                line.remove_prefix(7);
                auto [remainder, success] = ParseTextureOption(line, &material.specular_texopt);
                material.specular_texname = remainder;
                line_parsed               = success;
            } else if (StartsWith(line, "map_Ns ") || StartsWith(line, "map_Ns\t")) {
                line.remove_prefix(7);
                auto [remainder, success]           = ParseTextureOption(line, &material.specular_highlight_texopt);
                material.specular_highlight_texname = remainder;
                line_parsed                         = success;
            } else if (
                StartsWith(line, "map_bump ") || StartsWith(line, "map_bump\t") || StartsWith(line, "map_Bump ") ||
                StartsWith(line, "map_Bump\t")) {
                line.remove_prefix(9);
                auto [remainder, success] = ParseTextureOption(line, &material.bump_texopt);
                material.bump_texname     = remainder;
                line_parsed               = success;
            } else if (StartsWith(line, "map_d ") || StartsWith(line, "map_d\t")) {
                line.remove_prefix(6);
                auto [remainder, success] = ParseTextureOption(line, &material.alpha_texopt);
                material.alpha_texname    = remainder;
                line_parsed               = success;
            } else if (StartsWith(line, "map_Pr ") || StartsWith(line, "map_Pr\t")) {
                line.remove_prefix(7);
                auto [remainder, success]  = ParseTextureOption(line, &material.roughness_texopt);
                material.roughness_texname = remainder;
                line_parsed                = success;
            } else if (StartsWith(line, "map_Pm ") || StartsWith(line, "map_Pm\t")) {
                line.remove_prefix(7);
                auto [remainder, success] = ParseTextureOption(line, &material.metallic_texopt);
                material.metallic_texname = remainder;
                line_parsed               = success;
            } else if (StartsWith(line, "map_Ps ") || StartsWith(line, "map_Ps\t")) {
                line.remove_prefix(7);
                auto [remainder, success] = ParseTextureOption(line, &material.sheen_texopt);
                material.sheen_texname    = remainder;
                line_parsed               = success;
            } else if (StartsWith(line, "map_Ke ") || StartsWith(line, "map_Ke\t")) {
                line.remove_prefix(7);
                auto [remainder, success] = ParseTextureOption(line, &material.emissive_texopt);
                material.emissive_texname = remainder;
                line_parsed               = success;
            }
            break;
        }
        case 'N': {
            if (StartsWith(line, "Ns ") || StartsWith(line, "Ns\t")) {
                line.remove_prefix(3);
                line_parsed = 1 == ParseReals(line, 1, &material.shininess);
            } else if (StartsWith(line, "Ni ") || StartsWith(line, "Ni\t")) {
                line.remove_prefix(3);
                line_parsed = 1 == ParseReals(line, 1, &material.ior);
            }
            break;
        }
        case 'P': {
            if (StartsWith(line, "Pr ") || StartsWith(line, "Pr\t")) {
                line.remove_prefix(3);
                line_parsed = 1 == ParseReals(line, 1, &material.roughness);
            } else if (StartsWith(line, "Pm ") || StartsWith(line, "Pm\t")) {
                line.remove_prefix(3);
                line_parsed = 1 == ParseReals(line, 1, &material.metallic);
            } else if (StartsWith(line, "Ps ") || StartsWith(line, "Ps\t")) {
                line.remove_prefix(3);
                line_parsed = 1 == ParseReals(line, 1, &material.sheen);
            } else if (StartsWith(line, "Pc ") || StartsWith(line, "Pc\t")) {
                line.remove_prefix(3);
                line_parsed = 1 == ParseReals(line, 1, &material.clearcoat_thickness);
            } else if (StartsWith(line, "Pcr ") || StartsWith(line, "Pcr\t")) {
                line.remove_prefix(4);
                line_parsed = 1 == ParseReals(line, 1, &material.clearcoat_roughness);
            }
            break;
        }
        case '#': {
            line_parsed = true;
            break;
        }
        } // end switch
        if (!line_parsed && line_num > 1) {
            return { MaterialMap{},
                     Materials{},
                     Error{ make_error_code(rapidobj_errc::MaterialParseError), std::string(line_clone), line_num } };
        }
    } // end loop

    if (!material.name.empty()) {
        materials.push_back(std::move(material));
    }

    return { std::move(material_map), std::move(materials), Error{} };
}

inline auto FindBestPath(SharedContext* context)
{
    const auto& basepath = context->material.basepath;
    const auto& paths    = std::get<std::vector<std::filesystem::path>>(context->material.library->Value());

    for (const auto& path : paths) {
        auto bestpath = path.is_absolute() ? path : (basepath / path);
        if (std::filesystem::is_directory(bestpath)) {
            bestpath /= context->material.library_name;
        }
        if (std::filesystem::exists(bestpath) && std::filesystem::is_regular_file(bestpath)) {
            return bestpath;
        }
    }

    return std::filesystem::path();
}

inline auto ParseMaterialLibrary(SharedContext* context)
{
    if (std::holds_alternative<std::string_view>(context->material.library->Value())) {
        return ParseMaterials(std::get<std::string_view>(context->material.library->Value()));
    }

    auto filepath = FindBestPath(context);

    if (filepath.empty()) {
        return ParseMaterialsResult{ {}, {}, Error{ make_error_code(rapidobj_errc::MaterialFileError) } };
    }

    auto file = std::ifstream(filepath, std::ios::in | std::ios::binary | std::ios::ate);

    auto filesize = file.tellg();
    if (filesize <= 0) {
        return ParseMaterialsResult{ {}, {}, Error{ std::make_error_code(std::io_errc::stream) } };
    }
    file.seekg(0, std::ios::beg);

    auto filedata = Buffer<char>(static_cast<size_t>(filesize));
    if (file.is_open()) {
        file.read(filedata.data(), filesize);
    } else {
        return ParseMaterialsResult{ {}, {}, Error{ std::make_error_code(std::io_errc::stream) } };
    }
    file.close();

    return ParseMaterials(std::string_view(filedata.data(), static_cast<size_t>(filesize)));
}

inline void DispatchMergeTasks(const MergeTasks& tasks, std::shared_ptr<SharedContext> context)
{
    while (true) {
        auto fetched_index = std::atomic_fetch_add(&context->merging.task_index, size_t(1));

        if (fetched_index >= tasks.size()) {
            break;
        }

        const auto& merge_task = tasks[fetched_index];

        if (auto rc = std::visit([](const auto& task) { return task.Execute(); }, merge_task);
            rc != rapidobj_errc::Success) {
            std::lock_guard lock(context->merging.mutex);
            if (context->merging.error == rapidobj_errc::Success) {
                context->merging.error = rc;
            }
            break;
        }
    }

    if (1 == std::atomic_fetch_sub(&context->merging.thread_count, size_t(1))) {
        context->merging.completed.set_value();
    }
}

inline void MergeSequential(MergeTasks* tasks, std::shared_ptr<SharedContext> context)
{
    DispatchMergeTasks(std::vector<MergeTask>(tasks->begin(), tasks->end()), context);
}

inline void MergeParallel(MergeTasks* merge_tasks, std::shared_ptr<SharedContext> context)
{
    auto tasks = MergeTasks();
    tasks.reserve(merge_tasks->size());

    for (auto& merge_task : *merge_tasks) {
        auto cost = std::visit([](const auto& task) { return task.Cost(); }, merge_task);

        if (cost >= kMergeSubdivideCost) {
            auto divisor  = std::max(size_t(2), cost / kMergeSubdivideCost);
            auto subtasks = std::visit([divisor](const auto& task) { return task.Subdivide(divisor); }, merge_task);
            for (auto& subtask : subtasks) {
                tasks.push_back(std::move(subtask));
            }
        } else {
            tasks.push_back(std::move(merge_task));
        }
    }

    auto threads = std::vector<std::thread>{};
    threads.reserve(context->thread.concurrency);

    context->merging.thread_count = context->thread.concurrency;

    for (size_t i = 0; i != context->thread.concurrency; ++i) {
        threads.emplace_back(DispatchMergeTasks, tasks, context);
        threads.back().detach();
    }

    // wait for merging to finish
    context->merging.completed.get_future().wait();
}

// Merge function helper structs
struct ListInfo final {
    size_t face_buffers_size{};
    size_t shape_records_size{};
    size_t material_offsets_size{};
    size_t smoothing_offsets_size{};

    ListInfo& operator+=(const ListInfo& rhs) noexcept
    {
        face_buffers_size += rhs.face_buffers_size;
        shape_records_size += rhs.shape_records_size;
        material_offsets_size += rhs.material_offsets_size;
        smoothing_offsets_size += rhs.smoothing_offsets_size;
        return *this;
    }
};

struct ShapeInfo final {
    struct Mesh final {
        size_t index_array_size{};
        size_t faces_array_size{};
    };
    struct Line final {
        size_t index_array_size{};
        size_t segment_array_size{};
    };
    struct Point final {
        size_t index_array_size{};
    };
    Mesh  mesh;
    Line  line;
    Point point;
};

struct Offset final {
    size_t position{};
    size_t texcoord{};
    size_t normal{};
    size_t face{};

    Offset& operator+=(const Offset& rhs) noexcept
    {
        position += rhs.position;
        texcoord += rhs.texcoord;
        normal += rhs.normal;
        face += rhs.face;
        return *this;
    }
};

inline Result Merge(const std::vector<Chunk>& chunks, std::shared_ptr<SharedContext> context)
{
    // compute overall sizes of lists
    auto list_info = ListInfo{};
    for (const Chunk& chunk : chunks) {
        list_info += { chunk.mesh.faces.buffer.size(),
                       chunk.shapes.list.size(),
                       chunk.materials.list.size(),
                       chunk.smoothing.list.size() };
    }

    // compute offsets for each chunk and total attribute count
    auto offsets = std::vector<Offset>();
    auto count   = AttributeInfo{};
    {
        offsets.reserve(chunks.size());
        auto running = Offset{};
        for (const Chunk& chunk : chunks) {
            offsets.push_back(running);
            running += { chunk.positions.count, chunk.texcoords.count, chunk.normals.count, chunk.mesh.faces.count };
        }
        count          = { running.position, running.texcoord, running.normal };
        context->stats = { running.position, running.texcoord, running.normal };
    }

    // coalesce per-chunk shape records into a single shape records list
    auto shape_records = std::vector<ShapeRecord>();
    {
        shape_records.reserve(list_info.shape_records_size + 2);
        shape_records.push_back({});
        for (size_t i = 0; i != chunks.size(); ++i) {
            for (const ShapeRecord& record : chunks[i].shapes.list) {
                shape_records.push_back(record);
                shape_records.back().chunk_index = i;
            }
        }
        shape_records.push_back({});
        shape_records.back().mesh.index_buffer_start    = chunks.back().mesh.indices.buffer.size();
        shape_records.back().mesh.face_buffer_start     = chunks.back().mesh.faces.buffer.size();
        shape_records.back().lines.index_buffer_start   = chunks.back().lines.indices.buffer.size();
        shape_records.back().lines.segment_buffer_start = chunks.back().lines.segments.buffer.size();
        shape_records.back().points.index_buffer_start  = chunks.back().points.indices.buffer.size();
        shape_records.back().chunk_index                = chunks.size() - 1;
    }

    // prepare smoothing-source list from per-chunk smoothing-record lists
    auto smoothing_src = std::vector<FillSrc<uint32_t>>();
    {
        smoothing_src.reserve(list_info.smoothing_offsets_size + 2);
        smoothing_src.push_back({ 0, 0 });
        for (size_t i = 0; i != chunks.size(); ++i) {
            for (const SmoothingRecord& record : chunks[i].smoothing.list) {
                if (!smoothing_src.empty()) {
                    if (record.face_buffer_start + offsets[i].face == smoothing_src.back().offset) {
                        smoothing_src.pop_back();
                    }
                }
                smoothing_src.push_back({ record.group_id, record.face_buffer_start + offsets[i].face });
            }
        }
        smoothing_src.push_back({ 0, list_info.face_buffers_size });
    }

    auto parsed_materials = ParseMaterialsResult{};

    if (context->material.parse_result.valid()) {
        parsed_materials = context->material.parse_result.get();
        if (parsed_materials.error.code) {
            if (context->material.library->Policy() == Load::Optional) {
                parsed_materials = {};
                int id           = 0;
                for (const Chunk& chunk : chunks) {
                    for (const MaterialRecord& record : chunk.materials.list) {
                        auto [it, emplaced] = parsed_materials.material_map.try_emplace(std::string(record.name), id);
                        id += emplaced ? 1 : 0;
                    }
                }
            } else {
                return Result{ Attributes{}, Shapes{}, Materials{}, { parsed_materials.error.code } };
            }
        }
    }

    // prepare material-source list from per-chunk material-record lists
    auto material_src = std::vector<FillSrc<int32_t>>();
    if (context->material.library) {
        material_src.reserve(list_info.material_offsets_size + 2);
        material_src.push_back({ -1, 0 });
        for (size_t i = 0; i != chunks.size(); ++i) {
            for (const MaterialRecord& record : chunks[i].materials.list) {
                if (!material_src.empty()) {
                    if (record.face_buffer_start + offsets[i].face == material_src.back().offset) {
                        material_src.pop_back();
                    }
                }
                auto it = parsed_materials.material_map.find(record.name);
                if (it != parsed_materials.material_map.end()) {
                    material_src.push_back({ it->second, record.face_buffer_start + offsets[i].face });
                } else {
                    auto line_num = record.line_num;
                    for (size_t j = 0; j != i; ++j) {
                        line_num += chunks[j].text.line_count;
                    }
                    auto error = Error{ rapidobj_errc::MaterialNotFoundError, record.line, line_num };
                    return Result{ Attributes{}, Shapes{}, Materials{}, std::move(error) };
                }
            }
        }
        material_src.push_back({ -1, list_info.face_buffers_size });
    }

    auto shapes = Shapes();
    auto tasks  = MergeTasks();

    shapes.reserve(shape_records.size());

    // this big loop allocates shapes and computes tasks that construct them
    for (size_t i = 0; i != shape_records.size() - 1; ++i) {
        auto& shape = shape_records[i];
        auto& next  = shape_records[i + 1];

        // compute shape info
        auto shape_info = ShapeInfo{};

        for (size_t j = shape.chunk_index; j <= next.chunk_index; ++j) {
            auto mesh_index_buffer_size    = chunks[j].mesh.indices.buffer.size();
            auto mesh_faces_buffer_size    = chunks[j].mesh.faces.buffer.size();
            auto lines_index_buffer_size   = chunks[j].lines.indices.buffer.size();
            auto lines_segment_buffer_size = chunks[j].lines.segments.buffer.size();
            auto points_index_buffer_size  = chunks[j].points.indices.buffer.size();
            if (shape.chunk_index == next.chunk_index) {
                shape_info.mesh.index_array_size   = next.mesh.index_buffer_start - shape.mesh.index_buffer_start;
                shape_info.mesh.faces_array_size   = next.mesh.face_buffer_start - shape.mesh.face_buffer_start;
                shape_info.line.index_array_size   = next.lines.index_buffer_start - shape.lines.index_buffer_start;
                shape_info.line.segment_array_size = next.lines.segment_buffer_start - shape.lines.segment_buffer_start;
                shape_info.point.index_array_size  = next.points.index_buffer_start - shape.points.index_buffer_start;
            } else if (j == shape.chunk_index) {
                shape_info.mesh.index_array_size   = mesh_index_buffer_size - shape.mesh.index_buffer_start;
                shape_info.mesh.faces_array_size   = mesh_faces_buffer_size - shape.mesh.face_buffer_start;
                shape_info.line.index_array_size   = lines_index_buffer_size - shape.lines.index_buffer_start;
                shape_info.line.segment_array_size = lines_segment_buffer_size - shape.lines.segment_buffer_start;
                shape_info.point.index_array_size  = points_index_buffer_size - shape.points.index_buffer_start;
            } else if (j == next.chunk_index) {
                shape_info.mesh.index_array_size += next.mesh.index_buffer_start;
                shape_info.mesh.faces_array_size += next.mesh.face_buffer_start;
                shape_info.line.index_array_size += next.lines.index_buffer_start;
                shape_info.line.segment_array_size += next.lines.segment_buffer_start;
                shape_info.point.index_array_size += next.points.index_buffer_start;
            } else {
                shape_info.mesh.index_array_size += mesh_index_buffer_size;
                shape_info.mesh.faces_array_size += mesh_faces_buffer_size;
                shape_info.line.index_array_size += lines_index_buffer_size;
                shape_info.line.segment_array_size += lines_segment_buffer_size;
                shape_info.point.index_array_size += points_index_buffer_size;
            }
        }

        // skip empty shape
        if (!shape_info.mesh.index_array_size && !shape_info.line.index_array_size &&
            !shape_info.point.index_array_size) {
            continue;
        }

        auto num_indices       = shape_info.mesh.index_array_size;
        auto num_faces         = shape_info.mesh.faces_array_size;
        auto num_material_ids  = context->material.library ? num_faces : 0;
        auto num_smoothing_ids = num_faces;

        // allocate Shape
        shapes.push_back(Shape{
            std::move(shape.name),
            Mesh{ Array<Index>(num_indices),
                  Array<uint8_t>(num_faces),
                  Array<int32_t>(num_material_ids),
                  Array<uint32_t>(num_smoothing_ids) },
            Lines{ Array<Index>(shape_info.line.index_array_size), Array<int32_t>(shape_info.line.segment_array_size) },
            Points{ Array<Index>(shape_info.point.index_array_size) } });

        // compute tasks to construct shape mesh
        if (shape_info.mesh.index_array_size) {
            // compute mesh mem copies
            {
                auto index_dst = shapes.back().mesh.indices.begin();
                auto nface_dst = shapes.back().mesh.num_face_vertices.begin();

                for (size_t j = shape.chunk_index; j <= next.chunk_index; ++j) {
                    auto index_src   = chunks[j].mesh.indices.buffer.data();
                    auto index_flags = chunks[j].mesh.indices.flags.data();
                    auto index_size  = chunks[j].mesh.indices.buffer.size();
                    auto nface_src   = chunks[j].mesh.faces.buffer.data();
                    auto nface_size  = chunks[j].mesh.faces.buffer.size();
                    if (shape.chunk_index == next.chunk_index) {
                        index_src   = index_src + shape.mesh.index_buffer_start;
                        index_flags = index_flags + shape.mesh.index_buffer_start;
                        index_size  = next.mesh.index_buffer_start - shape.mesh.index_buffer_start;
                        nface_src   = nface_src + shape.mesh.face_buffer_start;
                        nface_size  = next.mesh.face_buffer_start - shape.mesh.face_buffer_start;
                    } else if (j == shape.chunk_index) {
                        index_src   = index_src + shape.mesh.index_buffer_start;
                        index_flags = index_flags + shape.mesh.index_buffer_start;
                        index_size  = index_size - shape.mesh.index_buffer_start;
                        nface_src   = nface_src + shape.mesh.face_buffer_start;
                        nface_size  = nface_size - shape.mesh.face_buffer_start;
                    } else if (j == next.chunk_index) {
                        index_size = next.mesh.index_buffer_start;
                        nface_size = next.mesh.face_buffer_start;
                    }
                    if (index_size) {
                        auto offset = AttributeInfo{ offsets[j].position, offsets[j].texcoord, offsets[j].normal };
                        tasks.push_back(CopyIndices(index_dst, index_src, index_flags, index_size, offset, count));
                        index_dst += index_size;
                    }
                    if (nface_size) {
                        tasks.push_back(CopyBytes(nface_dst, nface_src, nface_size));
                        nface_dst += nface_size;
                    }
                }
            }
            // compute material id fills
            if (!material_src.empty()) {
                auto dst   = shapes.back().mesh.material_ids.begin();
                auto size  = shape_info.mesh.faces_array_size;
                auto start = shape.mesh.face_buffer_start + offsets[shape.chunk_index].face;
                tasks.push_back(FillMaterialIds(dst, material_src, size, start));
            }
            // compute smoothing group id fills
            {
                auto dst   = shapes.back().mesh.smoothing_group_ids.begin();
                auto size  = shape_info.mesh.faces_array_size;
                auto start = shape.mesh.face_buffer_start + offsets[shape.chunk_index].face;
                tasks.push_back(FillSmoothingGroupIds(dst, smoothing_src, size, start));
            }
        }

        // compute tasks to construct shape lines
        if (shape_info.line.index_array_size) {
            auto index_dst = shapes.back().lines.indices.begin();
            auto nline_dst = shapes.back().lines.num_line_vertices.begin();

            for (size_t j = shape.chunk_index; j <= next.chunk_index; ++j) {
                auto index_src   = chunks[j].lines.indices.buffer.data();
                auto index_flags = chunks[j].lines.indices.flags.data();
                auto index_size  = chunks[j].lines.indices.buffer.size();
                auto nline_src   = chunks[j].lines.segments.buffer.data();
                auto nline_size  = chunks[j].lines.segments.buffer.size();
                if (shape.chunk_index == next.chunk_index) {
                    index_src   = index_src + shape.lines.index_buffer_start;
                    index_flags = index_flags + shape.lines.index_buffer_start;
                    index_size  = next.lines.index_buffer_start - shape.lines.index_buffer_start;
                    nline_src   = nline_src + shape.lines.segment_buffer_start;
                    nline_size  = next.lines.segment_buffer_start - shape.lines.segment_buffer_start;
                } else if (j == shape.chunk_index) {
                    index_src   = index_src + shape.lines.index_buffer_start;
                    index_flags = index_flags + shape.lines.index_buffer_start;
                    index_size  = index_size - shape.lines.index_buffer_start;
                    nline_src   = nline_src + shape.lines.segment_buffer_start;
                    nline_size  = nline_size - shape.lines.segment_buffer_start;
                } else if (j == next.chunk_index) {
                    index_size = next.lines.index_buffer_start;
                    nline_size = next.lines.segment_buffer_start;
                }
                if (index_size) {
                    auto offset = AttributeInfo{ offsets[j].position, offsets[j].texcoord, offsets[j].normal };
                    tasks.push_back(CopyIndices(index_dst, index_src, index_flags, index_size, offset, count));
                    index_dst += index_size;
                }
                if (nline_size) {
                    tasks.push_back(CopyInts(nline_dst, nline_src, nline_size));
                    nline_dst += nline_size;
                }
            }
        }

        // compute tasks to construct shape points
        if (shape_info.point.index_array_size) {
            auto index_dst = shapes.back().points.indices.begin();

            for (size_t j = shape.chunk_index; j <= next.chunk_index; ++j) {
                auto index_src   = chunks[j].points.indices.buffer.data();
                auto index_flags = chunks[j].points.indices.flags.data();
                auto index_size  = chunks[j].points.indices.buffer.size();
                if (shape.chunk_index == next.chunk_index) {
                    index_src   = index_src + shape.points.index_buffer_start;
                    index_flags = index_flags + shape.points.index_buffer_start;
                    index_size  = next.points.index_buffer_start - shape.points.index_buffer_start;
                } else if (j == shape.chunk_index) {
                    index_src   = index_src + shape.points.index_buffer_start;
                    index_flags = index_flags + shape.points.index_buffer_start;
                    index_size  = index_size - shape.points.index_buffer_start;
                } else if (j == next.chunk_index) {
                    index_size = next.points.index_buffer_start;
                }
                if (index_size) {
                    auto offset = AttributeInfo{ offsets[j].position, offsets[j].texcoord, offsets[j].normal };
                    tasks.push_back(CopyIndices(index_dst, index_src, index_flags, index_size, offset, count));
                    index_dst += index_size;
                }
            }
        }
    }

    // compute overall attribute array sizes
    auto attribute_size    = AttributeInfo{};
    bool has_vertex_colors = false;

    for (const Chunk& chunk : chunks) {
        attribute_size += { chunk.positions.buffer.size(), chunk.texcoords.buffer.size(), chunk.normals.buffer.size() };
        has_vertex_colors |= chunk.colors.count ? true : false;
    }

    auto attribute_size_color = has_vertex_colors ? attribute_size.position : size_t{};

    // allocate attribute arrays
    auto attributes = Attributes{ { attribute_size.position },
                                  { attribute_size.texcoord },
                                  { attribute_size.normal },
                                  { attribute_size_color } };

    // compute tasks to construct attribute arrays
    auto positions_destination = attributes.positions.data();
    auto texcoords_destination = attributes.texcoords.data();
    auto normals_destination   = attributes.normals.data();
    auto colors_destination    = attributes.colors.data();

    for (const Chunk& chunk : chunks) {
        if (chunk.positions.buffer.size()) {
            auto dst  = positions_destination;
            auto src  = chunk.positions.buffer.data();
            auto size = chunk.positions.buffer.size();
            tasks.push_back(CopyFloats(dst, src, size));
            positions_destination += size;
        }
        if (chunk.texcoords.buffer.size()) {
            auto dst  = texcoords_destination;
            auto src  = chunk.texcoords.buffer.data();
            auto size = chunk.texcoords.buffer.size();
            tasks.push_back(CopyFloats(dst, src, size));
            texcoords_destination += size;
        }
        if (chunk.normals.buffer.size()) {
            auto dst  = normals_destination;
            auto src  = chunk.normals.buffer.data();
            auto size = chunk.normals.buffer.size();
            tasks.push_back(CopyFloats(dst, src, size));
            normals_destination += size;
        }
        if (chunk.colors.buffer.size()) {
            auto dst  = colors_destination;
            auto src  = chunk.colors.buffer.data();
            auto size = chunk.colors.buffer.size();
            tasks.push_back(CopyFloats(dst, src, size));
            colors_destination += size;
        } else if (has_vertex_colors) {
            auto dst  = colors_destination;
            auto val  = 1.0f;
            auto size = chunk.positions.buffer.size();
            tasks.push_back(FillFloats(dst, val, size));
            colors_destination += size;
        }
    }

    // perform merge
    if (context->thread.concurrency > 1) {
        MergeParallel(&tasks, context);
    } else {
        MergeSequential(&tasks, context);
    }

    if (context->merging.error != rapidobj_errc::Success) {
        auto error = make_error_code(context->merging.error);
        return Result{ Attributes{}, Shapes{}, Materials{}, Error{ error } };
    }

    return Result{ std::move(attributes), std::move(shapes), std::move(parsed_materials.materials), Error{} };
}

inline auto ParsePosition(std::string_view line, Chunk* chunk)
{
    auto [count, remainder] = ParseXReals(line, 3, &chunk->positions.buffer);
    if (count < 3) {
        return rapidobj_errc::ParseError;
    }
    ++chunk->positions.count;
    auto [count2, remainder2] = ParseXReals(remainder, 3, &chunk->colors.buffer);
    if (count2 == 0) {
        if (chunk->colors.buffer.size()) {
            chunk->colors.buffer.ensure_enough_room_for(3);
            chunk->colors.buffer.fill_n(3, 1.0f);
            ++chunk->colors.count;
        }
    } else if (count2 == 3) {
        if ((chunk->colors.buffer.size() == 3) && (chunk->positions.buffer.size() > 3)) {
            size_t n = chunk->positions.buffer.size();
            float  b = chunk->colors.buffer.pop_back();
            float  g = chunk->colors.buffer.pop_back();
            float  r = chunk->colors.buffer.pop_back();
            chunk->colors.buffer.ensure_enough_room_for(n);
            chunk->colors.buffer.fill_n(n - 3, 1.0f);
            chunk->colors.buffer.push_back(r);
            chunk->colors.buffer.push_back(g);
            chunk->colors.buffer.push_back(b);
        }
        ++chunk->colors.count;
    } else {
        return rapidobj_errc::ParseError;
    }
    TrimLeft(remainder2);
    if (!remainder2.empty()) {
        return rapidobj_errc::ParseError;
    }
    return rapidobj_errc::Success;
}

inline rapidobj_errc ProcessLine(std::string_view line, Chunk* chunk, SharedContext* context)
{
    const auto text = line;

    TrimLeft(line);

    // skip empty lines
    if (line.empty()) {
        return rapidobj_errc::Success;
    }

    // process token
    switch (line.front()) {
    case 'v': {
        if (StartsWith(line, "v ") || StartsWith(line, "v\t")) {
            line.remove_prefix(2);
            if (auto rc = ParsePosition(line, chunk); rc != rapidobj_errc::Success) {
                return rc;
            }
        } else if (StartsWith(line, "vt ") || StartsWith(line, "vt\t")) {
            line.remove_prefix(3);
            auto count = ParseReals(line, 3, &chunk->texcoords.buffer);
            if (count < 2) {
                return rapidobj_errc::ParseError;
            } else if (count == 3) {
                chunk->texcoords.buffer.pop_back();
            }
            ++chunk->texcoords.count;
        } else if (StartsWith(line, "vn ") || StartsWith(line, "vn\t")) {
            line.remove_prefix(3);
            auto count = ParseReals(line, 3, &chunk->normals.buffer);
            if (count < 3) {
                return rapidobj_errc::ParseError;
            }
            ++chunk->normals.count;
        } else {
            return rapidobj_errc::ParseError;
        }
        break;
    }
    case 'f': {
        if (StartsWith(line, "f ") || StartsWith(line, "f\t")) {
            line.remove_prefix(2);
            auto [count, rc] = ParseFace(
                line,
                chunk->positions.count,
                chunk->texcoords.count,
                chunk->normals.count,
                kMinVerticesInFace,
                kMaxVerticesInFace,
                static_cast<OffsetFlags>(ApplyOffset::All),
                &chunk->mesh.indices.buffer,
                &chunk->mesh.indices.flags);
            if (rc != rapidobj_errc::Success) {
                return rc;
            }
            chunk->mesh.faces.buffer.ensure_enough_room_for(1);
            chunk->mesh.faces.buffer.push_back(static_cast<unsigned char>(count));
            ++chunk->mesh.faces.count;
        }
        break;
    }
    case '#': break; // ignore comments
    case 'g':
    case 'o': {
        if (StartsWith(line, "g ") || StartsWith(line, "g\t") || StartsWith(line, "o ") || StartsWith(line, "o\t")) {
            line.remove_prefix(2);
            auto name = std::string(line);
            Trim(name);
            chunk->shapes.list.push_back({});
            chunk->shapes.list.back().name                       = std::move(name);
            chunk->shapes.list.back().mesh.index_buffer_start    = chunk->mesh.indices.buffer.size();
            chunk->shapes.list.back().mesh.face_buffer_start     = chunk->mesh.faces.buffer.size();
            chunk->shapes.list.back().lines.index_buffer_start   = chunk->lines.indices.buffer.size();
            chunk->shapes.list.back().lines.segment_buffer_start = chunk->lines.segments.buffer.size();
            chunk->shapes.list.back().points.index_buffer_start  = chunk->points.indices.buffer.size();
        } else {
            return rapidobj_errc::ParseError;
        }
        break;
    }
    case 'm': {
        if (StartsWith(line, "mtllib ") || StartsWith(line, "mtllib\t")) {
            if (context->material.library) {
                line.remove_prefix(7);
                Trim(line);
                bool ambiguous_material_library = false;
                if (context->material.library_name.empty()) {
                    std::lock_guard lock(context->material.mutex);
                    if (context->material.library_name.empty()) {
                        context->material.library_name = line;
                        context->material.parse_result = std::async(std::launch::async, ParseMaterialLibrary, context);
                    } else if (context->material.library_name != line) {
                        ambiguous_material_library = true;
                    }
                } else if (context->material.library_name != line) {
                    ambiguous_material_library = true;
                }
                if (ambiguous_material_library) {
                    return rapidobj_errc::AmbiguousMaterialLibraryError;
                }
            }
        } else {
            return rapidobj_errc::ParseError;
        }
        break;
    }
    case 'u': {
        if (StartsWith(line, "usemtl ") || StartsWith(line, "usemtl\t")) {
            if (context->material.library) {
                line.remove_prefix(7);
                chunk->materials.list.push_back(
                    { std::string(line), std::string(text), chunk->text.line_count, chunk->mesh.faces.buffer.size() });
            }
        } else {
            return rapidobj_errc::ParseError;
        }
        break;
    }
    case 's': {
        if (StartsWith(line, "s ") || StartsWith(line, "s\t")) {
            line.remove_prefix(2);
            Trim(line);
            if (line.empty()) {
                return rapidobj_errc::ParseError;
            } else if (line == "off") {
                chunk->smoothing.list.push_back({ 0, chunk->mesh.faces.buffer.size() });
            } else {
                auto value = 0U;
                auto data  = &line[0];
                if (auto [p, e] = std::from_chars(data, data + line.size(), value); e != kSuccess) {
                    return rapidobj_errc::ParseError;
                }
                chunk->smoothing.list.push_back({ value, chunk->mesh.faces.buffer.size() });
            }
        } else {
            return rapidobj_errc::ParseError;
        }
        break;
    }
    case 'l': {
        if (StartsWith(line, "l ") || StartsWith(line, "l\t")) {
            line.remove_prefix(2);
            auto [count, rc] = ParseFace(
                line,
                chunk->positions.count,
                chunk->texcoords.count,
                chunk->normals.count,
                kMinVerticesInLine,
                kMaxVerticesInLine,
                static_cast<OffsetFlags>(ApplyOffset::Position | ApplyOffset::Texcoord),
                &chunk->lines.indices.buffer,
                &chunk->lines.indices.flags);
            if (rc != rapidobj_errc::Success) {
                return rc;
            }
            chunk->lines.segments.buffer.ensure_enough_room_for(1);
            chunk->lines.segments.buffer.push_back(static_cast<unsigned char>(count));
            ++chunk->lines.segments.count;
        }
        break;
    }
    case 'p': {
        if (StartsWith(line, "p ") || StartsWith(line, "p\t")) {
            line.remove_prefix(2);
            auto [count, rc] = ParseFace(
                line,
                chunk->positions.count,
                chunk->texcoords.count,
                chunk->normals.count,
                kMinVerticesInPoint,
                kMaxVerticesInPoint,
                static_cast<OffsetFlags>(ApplyOffset::Position),
                &chunk->points.indices.buffer,
                &chunk->points.indices.flags);
            if (rc != rapidobj_errc::Success) {
                return rc;
            }
        }
        break;
    }
    default: {
        return rapidobj_errc::ParseError;
    }
    }

    return rapidobj_errc::Success;
}

inline void ProcessBlocksImpl(
    Reader*        reader,
    size_t         block_begin,
    size_t         block_end,
    bool           stop_parsing_after_eol,
    Chunk*         chunk,
    SharedContext* context)
{
    assert(reader);

    bool begin_parsing_after_eol = block_begin > 0;
    bool reached_eof             = false;

    auto buffer_size = kMaxLineLength + kBlockSize;
    auto buffer1     = std::unique_ptr<char, sys::AlignedDeleter>(sys::AlignedAllocate(buffer_size, 4_KiB));
    auto buffer2     = std::unique_ptr<char, sys::AlignedDeleter>(sys::AlignedAllocate(buffer_size, 4_KiB));

    auto front_buffer = buffer1.get();
    auto back_buffer  = buffer2.get();

    auto file_offset = block_begin * kBlockSize;

    auto line = std::string_view();
    auto text = std::string_view();

    if (auto ec = reader->ReadBlock(file_offset, kBlockSize, front_buffer + kMaxLineLength)) {
        chunk->error = Error{ ec };
        return;
    }

    if (auto [bytes_read, ec] = reader->WaitForResult(); ec) {
        chunk->error = Error{ ec };
        return;
    } else {
        reached_eof = bytes_read < kBlockSize;
        text        = std::string_view(front_buffer + kMaxLineLength, bytes_read);
    }

    if (begin_parsing_after_eol) {
        if (auto ptr = static_cast<const char*>(memchr(text.data(), '\n', kMaxLineLength))) {
            auto pos = static_cast<size_t>(ptr - text.data());
            text.remove_prefix(pos + 1);
        } else {
            ++chunk->text.line_count;
            auto ec      = make_error_code(rapidobj_errc::LineTooLongError);
            chunk->error = Error{ ec, std::string(text, 0, kMaxLineLength), chunk->text.line_count };
            return;
        }
    }

    for (size_t i = block_begin; i != block_end; ++i) {
        auto remainder = size_t{};

        bool last_block = (i + 1 == block_end) || reached_eof;

        if (!last_block) {
            file_offset = (i + 1) * kBlockSize;

            if (auto ec = reader->ReadBlock(file_offset, kBlockSize, back_buffer + kMaxLineLength)) {
                chunk->error = Error{ ec };
                return;
            }

        } else if (stop_parsing_after_eol) {
            if (auto ptr = static_cast<const char*>(memchr(text.data(), '\n', kMaxLineLength))) {
                auto pos = static_cast<size_t>(ptr - text.data());
                line     = text.substr(0, pos);
                if (EndsWith(line, '\r')) {
                    line.remove_suffix(1);
                }
                ++chunk->text.line_count;
                if (auto rc = ProcessLine(line, chunk, context); rc != rapidobj_errc::Success) {
                    chunk->error = Error{ make_error_code(rc), std::string(line), chunk->text.line_count };
                }
            } else {
                ++chunk->text.line_count;
                auto ec      = make_error_code(rapidobj_errc::LineTooLongError);
                chunk->error = Error{ ec, std::string(text, 0, kMaxLineLength), chunk->text.line_count };
            }
            return;
        }

        while (!text.empty()) {
            if (auto ptr = static_cast<const char*>(memchr(text.data(), '\n', text.size()))) {
                auto pos = static_cast<size_t>(ptr - text.data());
                ++chunk->text.line_count;
                if (pos > kMaxLineLength) {
                    auto ec      = make_error_code(rapidobj_errc::LineTooLongError);
                    chunk->error = Error{ ec, std::string(text, 0, kMaxLineLength), chunk->text.line_count };
                    return;
                }
                line = text.substr(0, pos);
                if (EndsWith(line, '\r')) {
                    line.remove_suffix(1);
                }
                text.remove_prefix(pos + 1);
            } else {
                if (text.size() > kMaxLineLength) {
                    ++chunk->text.line_count;
                    auto ec      = make_error_code(rapidobj_errc::LineTooLongError);
                    chunk->error = Error{ ec, std::string(text, 0, kMaxLineLength), chunk->text.line_count };
                    return;
                }
                if (last_block) {
                    line = text;
                    ++chunk->text.line_count;
                    if (auto rc = ProcessLine(line, chunk, context); rc != rapidobj_errc::Success) {
                        chunk->error = Error{ make_error_code(rc), std::string(line), chunk->text.line_count };
                        return;
                    }
                } else {
                    remainder = text.size();
                    memcpy(back_buffer + kMaxLineLength - remainder, text.data(), remainder);
                }
                text = {};
                break;
            }

            if (auto rc = ProcessLine(line, chunk, context); rc != rapidobj_errc::Success) {
                chunk->error = Error{ make_error_code(rc), std::string(line), chunk->text.line_count };
                return;
            }
        }

        if (!last_block) {
            auto [bytes_read, ec] = reader->WaitForResult();
            if (ec) {
                chunk->error = Error{ ec };
                return;
            }
            reached_eof = bytes_read < kBlockSize;
            std::swap(front_buffer, back_buffer);
            text = std::string_view(front_buffer + kMaxLineLength - remainder, bytes_read + remainder);
        } else if (reached_eof) {
            break;
        }
    }
}

inline void ProcessBlocks(
    DataSource                     source,
    size_t                         thread_index,
    size_t                         block_begin,
    size_t                         block_end,
    bool                           stop_parsing_after_eol,
    Chunk*                         chunk,
    std::shared_ptr<SharedContext> context)
{
    assert(chunk);
    assert(context);
    assert(block_begin < block_end);

    auto t1 = std::chrono::steady_clock::now();

    auto reader = CreateReader(source);

    if (reader->Error()) {
        chunk->error = Error{ reader->Error() };
    } else {
        ProcessBlocksImpl(reader.get(), block_begin, block_end, stop_parsing_after_eol, chunk, context.get());
    }

    if (1 == std::atomic_fetch_sub(&context->parsing.thread_count, size_t(1))) {
        context->parsing.completed.set_value();
    }

    auto t2 = std::chrono::steady_clock::now();

    auto parse_time = t2 - t1;

    context->debug.io.num_requests[thread_index]  = reader->NumRequests();
    context->debug.io.num_bytes_read[thread_index] = reader->BytesRead();
    context->debug.io.submit_time[thread_index] = reader->SubmitTime();
    context->debug.io.wait_time[thread_index]   = reader->WaitTime();
    context->debug.parse.time[thread_index]     = parse_time;
}

inline void ParseFileSequential(sys::File* file, std::vector<Chunk>* chunks, std::shared_ptr<SharedContext> context)
{
    context->thread.concurrency   = 1;
    context->parsing.thread_count = 1;

    chunks->resize(1);

    context->debug.io.num_requests.resize(1);
    context->debug.io.num_bytes_read.resize(1);
    context->debug.io.submit_time.resize(1);
    context->debug.io.wait_time.resize(1);
    context->debug.parse.time.resize(1);

    auto source = DataSource(file);

    auto num_blocks             = file->size() / kBlockSize + (file->size() % kBlockSize != 0);
    auto stop_parsing_after_eol = false;
    auto chunk                  = &chunks->front();

    ProcessBlocks(source, 0, 0, num_blocks, stop_parsing_after_eol, chunk, context);
}

inline void ParseFileParallel(sys::File* file, std::vector<Chunk>* chunks, std::shared_ptr<SharedContext> context)
{
    auto source                = DataSource(file);
    auto num_blocks            = file->size() / kBlockSize + (file->size() % kBlockSize != 0);
    auto num_threads           = std::thread::hardware_concurrency();
    auto num_blocks_per_thread = num_blocks / num_threads;
    auto num_remainder_blocks  = num_blocks - (num_blocks_per_thread * num_threads);

    auto tasks = std::vector<size_t>();

    tasks.reserve(num_threads);

    // allocate blocks to tasks
    {
        auto block = size_t{};

        while (block < num_blocks) {
            tasks.push_back(block);
            block += num_blocks_per_thread;
            if (num_remainder_blocks > 0) {
                ++block;
                --num_remainder_blocks;
            }
        }
    }

    auto num_tasks = tasks.size();
    auto threads   = std::vector<std::thread>{};

    chunks->resize(num_tasks);

    context->thread.concurrency   = num_threads;
    context->parsing.thread_count = num_tasks;

    context->debug.io.num_requests.resize(num_threads);
    context->debug.io.num_bytes_read.resize(num_threads);
    context->debug.io.submit_time.resize(num_threads);
    context->debug.io.wait_time.resize(num_threads);
    context->debug.parse.time.resize(num_threads);

    threads.reserve(num_tasks);

    // allocate tasks to threads
    for (size_t i = 0; i != tasks.size(); ++i) {
        bool is_last                = i + 1 == tasks.size();
        auto begin                  = tasks[i];
        auto end                    = is_last ? num_blocks : (tasks[i + 1] + 1);
        bool stop_parsing_after_eol = !is_last;
        auto chunk                  = &(*chunks)[i];

        threads.emplace_back(ProcessBlocks, source, i, begin, end, stop_parsing_after_eol, chunk, context);
        threads.back().detach();
    }

    // wait for parsing to finish
    context->parsing.completed.get_future().wait();
}

inline Result ParseFile(const std::filesystem::path& filepath, const MaterialLibrary& material_library)
{
    if (filepath.empty()) {
        auto error = std::make_error_code(std::errc::invalid_argument);
        return Result{ Attributes{}, Shapes{}, Materials{}, Error{ error } };
    }

    auto file = sys::File(filepath);

    if (!file) {
        return Result{ Attributes{}, Shapes{}, Materials{}, Error{ file.error() } };
    }

    const auto material_library_value   = &material_library.Value();
    const auto default_loading_policy   = material_library.Policy().value_or(Load::Mandatory);
    const auto default_material_library = MaterialLibrary::SearchPath(".", default_loading_policy);

    auto context = std::make_shared<SharedContext>();

    context->material.basepath = filepath.parent_path();

    if (std::get_if<std::nullptr_t>(material_library_value) != nullptr) {
        context->material.library = nullptr;
    } else if (std::get_if<std::monostate>(material_library_value) != nullptr) {
        context->material.library = &default_material_library;
    } else if (auto* paths = std::get_if<std::vector<std::filesystem::path>>(material_library_value)) {
        if (paths->empty()) {
            return Result{ Attributes{}, Shapes{}, Materials{}, Error{ rapidobj_errc::InvalidArgumentsError } };
        }
        context->material.library = &material_library;
    } else if (std::get_if<std::string_view>(material_library_value) != nullptr) {
        context->material.library = &material_library;
    } else {
        return Result{ Attributes{}, Shapes{}, Materials{}, Error{ rapidobj_errc::InternalError } };
    }

    auto chunks = std::vector<Chunk>();

    auto t1 = std::chrono::steady_clock::now();

    if (file.size() <= kSingleThreadCutoff) {
        ParseFileSequential(&file, &chunks, context);
    } else {
        ParseFileParallel(&file, &chunks, context);
    }

    auto t2 = std::chrono::steady_clock::now();

    context->debug.parse.total_time = t2 - t1;

    // check if an error occured
    size_t running_line_num = size_t{};
    for (auto& chunk : chunks) {
        if (chunk.error.code) {
            chunk.error.line_num += running_line_num;
            return Result{ Attributes{}, Shapes{}, Materials{}, chunk.error };
        }
        running_line_num += chunk.text.line_count;
    }

    t1 = std::chrono::steady_clock::now();

    auto result = Merge(chunks, context);

    t2 = std::chrono::steady_clock::now();

    context->debug.merge.total_time = t2 - t1;

    // std::cout << DumpDebug(*context);

    auto memory = size_t{ 0 };

    for (const auto& chunk : chunks) {
        memory += SizeInBytes(chunk);
    }

    // Free memory in a different thread
    if (memory > kMemoryRecyclingSize) {
        auto recycle = std::thread([](std::vector<Chunk>&&) {}, std::move(chunks));
        recycle.detach();
    }

    return result;
}

inline Result ParseStream(std::istream& is, const MaterialLibrary& material_library)
{
    auto context                  = std::make_shared<SharedContext>();
    auto chunks                   = std::vector<Chunk>(1);
    auto source                   = DataSource(&is);
    auto num_blocks               = std::numeric_limits<size_t>::max();
    auto stop_parsing_after_eol   = false;
    auto chunk                    = &chunks.front();
    auto material_library_value   = &material_library.Value();
    auto default_material_library = MaterialLibrary::SearchPaths({}, Load::Optional);

    if (std::get_if<std::nullptr_t>(material_library_value) != nullptr) {
        context->material.library = nullptr;
    } else if (std::get_if<std::monostate>(material_library_value) != nullptr) {
        if (material_library.Policy()) {
            return Result{ Attributes{}, Shapes{}, Materials{}, Error{ rapidobj_errc::InvalidArgumentsError } };
        }
        context->material.library = &default_material_library;
    } else if (auto* paths = std::get_if<std::vector<std::filesystem::path>>(material_library_value)) {
        if (paths->empty()) {
            return Result{ Attributes{}, Shapes{}, Materials{}, Error{ rapidobj_errc::InvalidArgumentsError } };
        }
        if (std::any_of(paths->begin(), paths->end(), [](auto& path) { return path.is_relative(); })) {
            return Result{ Attributes{}, Shapes{}, Materials{}, Error{ rapidobj_errc::MaterialRelativePathError } };
        }
        context->material.library = &material_library;
    } else if (std::get_if<std::string_view>(material_library_value) != nullptr) {
        context->material.library = &material_library;
    } else {
        return Result{ Attributes{}, Shapes{}, Materials{}, Error{ rapidobj_errc::InternalError } };
    }

    context->thread.concurrency   = 1;
    context->parsing.thread_count = 1;

    context->debug.io.num_requests.resize(1);
    context->debug.io.num_bytes_read.resize(1);
    context->debug.io.submit_time.resize(1);
    context->debug.io.wait_time.resize(1);
    context->debug.parse.time.resize(1);

    auto t1 = std::chrono::steady_clock::now();

    ProcessBlocks(source, 0, 0, num_blocks, stop_parsing_after_eol, chunk, context);

    auto t2 = std::chrono::steady_clock::now();

    context->debug.parse.total_time = t2 - t1;

    if (chunks.front().error.code) {
        return Result{ Attributes{}, Shapes{}, Materials{}, chunks.front().error };
    }

    t1 = std::chrono::steady_clock::now();

    auto result = Merge(chunks, context);

    t2 = std::chrono::steady_clock::now();

    context->debug.merge.total_time = t2 - t1;

    // std::cout << DumpDebug(*context);

    // Free memory in a different thread
    if (SizeInBytes(chunks.front()) > kMemoryRecyclingSize) {
        auto recycle = std::thread([](std::vector<Chunk>&&) {}, std::move(chunks));
        recycle.detach();
    }

    return result;
}

struct TriangulateTask final {
    TriangulateTask(
        const Mesh* src_,
        Mesh*       dst_,
        size_t      cost_,
        size_t      isrc_,
        size_t      idst_,
        size_t      fsrc_,
        size_t      fdst_,
        size_t      size_) noexcept
        : src(src_), dst(dst_), cost(cost_), isrc(isrc_), idst(idst_), fsrc(fsrc_), fdst(fdst_), size(size_)
    {}

    const Mesh* src{};
    Mesh*       dst{};
    size_t      cost{};
    size_t      isrc{};
    size_t      idst{};
    size_t      fsrc{};
    size_t      fdst{};
    size_t      size{};
};

inline auto CalculatePolygonArea(float* x, float* y, size_t size) noexcept
{
    auto area = 0.0f;

    for (size_t i = 1; i != size; ++i) {
        auto avg_height = (y[i - 1] + y[i]) / 2;
        auto width      = x[i] - x[i - 1];
        area += width * avg_height;
    }

    auto avg_height = (y[0] + y[size - 1]) / 2;
    auto width      = x[0] - x[size - 1];
    area += width * avg_height;

    return std::abs(area);
}

enum class ProjectionPlane { X, Y, Z };

inline bool TriangulateSingleTask(const Array<float>& positions, const TriangulateTask& task)
{
    auto [src, dst, cost, isrc, idst, fsrc, fdst, size] = task;

    auto  complex   = std::array<std::vector<std::array<float, 2>>, 1>();
    auto& polygon   = complex.front();
    auto  index_map = std::vector<size_t>();

    for (size_t i = 0; i != size; ++i) {
        auto num_vertices = src->num_face_vertices[fsrc + i];
        auto material_id  = src->material_ids[fsrc + i];
        auto smoothing_id = src->smoothing_group_ids[fsrc + i];

        if (num_vertices == 3) {
            dst->indices[idst + 0] = src->indices[isrc + 0];
            dst->indices[idst + 1] = src->indices[isrc + 1];
            dst->indices[idst + 2] = src->indices[isrc + 2];

            isrc += 3;
            idst += 3;

            dst->num_face_vertices[fdst]   = 3;
            dst->material_ids[fdst]        = material_id;
            dst->smoothing_group_ids[fdst] = smoothing_id;

            fdst++;

        } else if (num_vertices == 4) {
            const auto& index0 = src->indices[isrc + 0];
            const auto& index1 = src->indices[isrc + 1];
            const auto& index2 = src->indices[isrc + 2];
            const auto& index3 = src->indices[isrc + 3];

            isrc += 4;

            bool d02_is_less{};
            {
                auto position_index0 = 3 * static_cast<size_t>(index0.position_index);
                auto position_index1 = 3 * static_cast<size_t>(index1.position_index);
                auto position_index2 = 3 * static_cast<size_t>(index2.position_index);
                auto position_index3 = 3 * static_cast<size_t>(index3.position_index);

                auto pos0_x = positions[position_index0 + 0];
                auto pos0_y = positions[position_index0 + 1];
                auto pos0_z = positions[position_index0 + 2];

                auto pos1_x = positions[position_index1 + 0];
                auto pos1_y = positions[position_index1 + 1];
                auto pos1_z = positions[position_index1 + 2];

                auto pos2_x = positions[position_index2 + 0];
                auto pos2_y = positions[position_index2 + 1];
                auto pos2_z = positions[position_index2 + 2];

                auto pos3_x = positions[position_index3 + 0];
                auto pos3_y = positions[position_index3 + 1];
                auto pos3_z = positions[position_index3 + 2];

                auto e02_x = pos0_x - pos2_x;
                auto e02_y = pos0_y - pos2_y;
                auto e02_z = pos0_z - pos2_z;

                auto e13_x = pos1_x - pos3_x;
                auto e13_y = pos1_y - pos3_y;
                auto e13_z = pos1_z - pos3_z;

                auto d02 = e02_x * e02_x + e02_y * e02_y + e02_z * e02_z;
                auto d13 = e13_x * e13_x + e13_y * e13_y + e13_z * e13_z;

                d02_is_less = d02 < d13;
            }

            dst->indices[idst + 0] = index0;
            dst->indices[idst + 1] = index1;
            dst->indices[idst + 2] = d02_is_less ? index2 : index3;
            dst->indices[idst + 3] = d02_is_less ? index0 : index1;
            dst->indices[idst + 4] = index2;
            dst->indices[idst + 5] = index3;

            idst += 6;

            dst->num_face_vertices[fdst + 0]   = 3;
            dst->num_face_vertices[fdst + 1]   = 3;
            dst->material_ids[fdst + 0]        = material_id;
            dst->material_ids[fdst + 1]        = material_id;
            dst->smoothing_group_ids[fdst + 0] = smoothing_id;
            dst->smoothing_group_ids[fdst + 1] = smoothing_id;

            fdst += 2;
        } else {
            auto xs = std::array<float, kMaxVerticesInFace>();
            auto ys = std::array<float, kMaxVerticesInFace>();
            auto zs = std::array<float, kMaxVerticesInFace>();

            polygon.clear();
            index_map.clear();

            for (size_t k = 0; k != num_vertices; ++k) {
                auto position_index = 3 * static_cast<size_t>(src->indices[isrc + k].position_index);
                index_map.push_back(isrc + k);
                xs[k] = positions[position_index + 0];
                ys[k] = positions[position_index + 1];
                zs[k] = positions[position_index + 2];
            }

            isrc += num_vertices;

            auto area_x = CalculatePolygonArea(ys.data(), zs.data(), num_vertices);
            auto area_y = CalculatePolygonArea(xs.data(), zs.data(), num_vertices);
            auto area_z = CalculatePolygonArea(xs.data(), ys.data(), num_vertices);

            if (FLT_MIN > std::max({ area_x, area_y, area_z })) {
                return false;
            }

            auto proj_plane = ProjectionPlane{};

            if (area_x > area_y) {
                proj_plane = area_x > area_z ? ProjectionPlane::X : ProjectionPlane::Z;
            } else {
                proj_plane = area_y > area_z ? ProjectionPlane::Y : ProjectionPlane::Z;
            }

            if (proj_plane == ProjectionPlane::X) {
                for (size_t k = 0; k != num_vertices; ++k) {
                    polygon.push_back({ ys[k], zs[k] });
                }
            } else if (proj_plane == ProjectionPlane::Y) {
                for (size_t k = 0; k != num_vertices; ++k) {
                    polygon.push_back({ xs[k], zs[k] });
                }
            } else {
                for (size_t k = 0; k != num_vertices; ++k) {
                    polygon.push_back({ xs[k], ys[k] });
                }
            }

            auto result = mapbox::earcut(complex);

            if (result.empty() || result.size() % 3 != 0) {
                return false;
            }

            for (size_t k = 0; k < result.size(); k += 3) {
                std::swap(result[k], result[k + 1]);
            }

            for (size_t k = 0; k != result.size(); ++k) {
                dst->indices[idst + k] = src->indices[index_map[result[k]]];
            }

            idst += result.size();

            auto num_triangles = result.size() / 3;

            for (size_t k = 0; k != num_triangles; ++k) {
                dst->num_face_vertices[fdst + k]   = 3;
                dst->material_ids[fdst + k]        = material_id;
                dst->smoothing_group_ids[fdst + k] = smoothing_id;
            }

            fdst += num_triangles;
        }
    }
    return true;
}

inline bool
TriangulateTasksParallel(size_t concurrency, const Array<float>& positions, const std::vector<TriangulateTask>& tasks)
{
    auto task_index  = std::atomic_size_t{ 0 };
    auto num_threads = std::atomic_size_t{ concurrency };
    auto completed   = std::promise<void>();
    auto success     = std::atomic_bool{ true };

    auto func = [&]() {
        auto fetched_index = std::atomic_fetch_add(&task_index, size_t(1));

        while (fetched_index < tasks.size()) {
            if (false == TriangulateSingleTask(positions, tasks[fetched_index])) {
                success = false;
                break;
            }
            fetched_index = std::atomic_fetch_add(&task_index, size_t(1));
        }

        if (1 == std::atomic_fetch_sub(&num_threads, size_t(1))) {
            completed.set_value();
        }
    };

    auto threads = std::vector<std::thread>();
    threads.reserve(concurrency);

    for (size_t i = 0; i != concurrency; ++i) {
        threads.emplace_back(func);
        threads.back().detach();
    }

    // wait for triangulation to finish
    completed.get_future().wait();

    return success;
}

inline bool TriangulateTasksSequential(const Array<float>& positions, const std::vector<TriangulateTask>& tasks)
{
    for (const auto& task : tasks) {
        bool success = TriangulateSingleTask(positions, task);
        if (!success) {
            return false;
        }
    }
    return true;
}

inline bool Triangulate(Result& result)
{
    auto mesh_tasks = std::vector<TriangulateTask>();
    auto tasks      = std::vector<TriangulateTask>();
    auto meshes     = std::vector<Mesh>(result.shapes.size());

    tasks.reserve(result.shapes.size());

    for (size_t i = 0; i != result.shapes.size(); ++i) {
        const auto& mesh = result.shapes[i].mesh;

        mesh_tasks.clear();

        auto cost = size_t{ 0 };

        auto isrc_begin = size_t{ 0 };
        auto isrc_end   = size_t{ 0 };
        auto idst_begin = size_t{ 0 };
        auto idst_end   = size_t{ 0 };

        auto fsrc_begin = size_t{ 0 };
        auto fsrc_end   = size_t{ 0 };
        auto fdst_begin = size_t{ 0 };
        auto fdst_end   = size_t{ 0 };

        auto triangle_sum = size_t{ 0 };

        for (auto num_vertices : mesh.num_face_vertices) {
            auto num_triangles = static_cast<size_t>(num_vertices - 2);

            triangle_sum += num_triangles;

            if (num_vertices == 3) {
                cost += kTriangulateTriangleCost;
            } else if (num_vertices == 4) {
                cost += kTriangulateQuadCost;
            } else {
                auto n = static_cast<size_t>(num_vertices);
                cost += kTriangulatePerIndexCost * n * n;
            }
            if (cost >= kTriangulateSubdivideCost) {
                auto size = fsrc_end - fsrc_begin;
                mesh_tasks.emplace_back(&mesh, nullptr, cost, isrc_begin, idst_begin, fsrc_begin, fdst_begin, size);

                isrc_begin = isrc_end;
                idst_begin = idst_end;
                fsrc_begin = fsrc_end;
                fdst_begin = fdst_end;

                cost = 0;
            }

            isrc_end += num_vertices;
            idst_end += 3 * num_triangles;
            fsrc_end += 1;
            fdst_end += num_triangles;
        }

        if (triangle_sum == mesh.num_face_vertices.size()) {
            continue;
        }

        if (auto size = fsrc_end - fsrc_begin; size > 0) {
            mesh_tasks.emplace_back(&mesh, nullptr, cost, isrc_begin, idst_begin, fsrc_begin, fdst_begin, size);
        }

        meshes[i].indices             = Array<Index>(3 * triangle_sum);
        meshes[i].num_face_vertices   = Array<uint8_t>(triangle_sum);
        meshes[i].material_ids        = Array<int32_t>(triangle_sum);
        meshes[i].smoothing_group_ids = Array<uint32_t>(triangle_sum);

        for (auto& task : mesh_tasks) {
            task.dst = &meshes[i];
        }

        tasks.insert(tasks.end(), mesh_tasks.begin(), mesh_tasks.end());
    }

    // already triangulated
    if (tasks.empty()) {
        return true;
    }

    auto hardware_threads = static_cast<size_t>(std::thread::hardware_concurrency());
    auto concurrency      = std::min(hardware_threads, tasks.size());
    bool success          = true;

    if (concurrency > 1) {
        success = TriangulateTasksParallel(concurrency, result.attributes.positions, tasks);
    } else {
        success = TriangulateTasksSequential(result.attributes.positions, tasks);
    }

    if (!success) {
        result.error = Error{ make_error_code(rapidobj_errc::TriangulationError), {}, {} };
    }

    auto old_meshes = std::vector<Mesh>();
    old_meshes.reserve(result.shapes.size());

    for (size_t i = 0; i != result.shapes.size(); ++i) {
        if (!meshes[i].indices.empty()) {
            old_meshes.push_back(std::move(result.shapes[i].mesh));
            result.shapes[i].mesh = std::move(meshes[i]);
        }
    }

    auto memory = size_t{ 0 };

    for (const auto& old_mesh : old_meshes) {
        memory += SizeInBytes(old_mesh);
    }

    // Free memory in a different thread
    if (memory > kMemoryRecyclingSize) {
        auto recycle = std::thread([](std::vector<Mesh>&&) {}, std::move(old_meshes));
        recycle.detach();
    }

    return success;
}

} // namespace detail

/// <summary>
/// Loads and parses Wavefront geometry definition file (.obj file).
/// </summary>
/// <param name="obj_filepath"> : path of the .obj file to parse.</param>
/// <param name="mtl_library"> : optional material library.</param>
/// <returns>Parsed data stored in Result class.</returns>
inline Result ParseFile(const std::filesystem::path& obj_filepath, const MaterialLibrary& mtl_library)
{
    return detail::ParseFile(obj_filepath, mtl_library);
}

/// <summary>
/// Loads and parses Wavefront geometry definition data from an input stream.
/// Because input streams are sequential, which prevents parsing parallelization,
/// this function is less performant compared to the ParseFile() function.
/// </summary>
/// <param name="obj_stream"> : input stream to parse.</param>
/// <param name="mtl_library"> : optional material library.</param>
/// <returns>Parsed data stored in Result class.</returns>
inline Result ParseStream(std::istream& obj_stream, const MaterialLibrary& mtl_library)
{
    return detail::ParseStream(obj_stream, mtl_library);
}

inline bool Triangulate(Result& result)
{
    return detail::Triangulate(result);
}

} // namespace rapidobj

#endif
