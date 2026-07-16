// New - Na
#pragma once
#include "gpu_raytracer.h"
#include <hip/hip_runtime.h> // Nagłówek dla typów HIP 

class HipRaytracer : public GPURaytracer {
public:
    HipRaytracer() = default;
    
    // Nadpisujemy destruktor, żeby zwalniał pamięć VRAM przy wyjściu z programu
    ~HipRaytracer() override;

    bool init() override;
    bool load_shader(const GPUShaderModuleDesc& desc) override;
    void render(int width, int height) override;
    
    // --- NASZE BOCZNE DRZWI DLA PAMIĘCI ---
    void set_host_buffer(float* buffer) { m_host_buffer = buffer; }

private:
    // Zmienna przechowująca skompilowany kod na karcie graficznej
    hipModule_t m_module = nullptr;
    hipFunction_t m_kernel = nullptr;
    
    float* m_host_buffer = nullptr;
};
