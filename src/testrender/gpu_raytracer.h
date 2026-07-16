// New - Na
#pragma once
#include <string>
#include <cstddef>

// Uniwersalna struktura do wczytywania bajtów
struct GPUShaderModuleDesc {
    std::string architecture;
    std::string format;
    const void* data_ptr;
    size_t data_size;
};

// Abstrakcyjna klasa bazowa
class GPURaytracer {
public:
    virtual ~GPURaytracer() = default;

    virtual bool init() = 0;
    virtual bool load_shader(const GPUShaderModuleDesc& desc) = 0;
    virtual void render(int width, int height) = 0;
};
