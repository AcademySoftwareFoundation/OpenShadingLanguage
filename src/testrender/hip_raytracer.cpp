// New - Ka
#include "hip_raytracer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <cstring>
#include <OpenImageIO/imageio.h>
#include <OSL/oslexec.h>
#include <cstring>

// Makro do wygodnego sprawdzania, czy funkcje HIP nie zwracają błędów
#define HIP_CHECK(command) \
{ \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " w linii " << __LINE__ << " w pliku " << __FILE__ << std::endl; \
        return false; \
    } \
}

// NOWE makro dla funkcji zwracających void (render)
#define HIP_CHECK_VOID(command) \
{ \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(status) \
                  << " w linii " << __LINE__ << " w pliku " << __FILE__ << std::endl; \
        return; \
    } \
}

// -------------------------------------------------------------------------
// 1. Inicjalizacja środowiska i karty graficznej
// -------------------------------------------------------------------------
bool HipRaytracer::init() {
    return true;
    std::cout << "\n=== [Inicjalizacja AMD HIP] ===\n";
    
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0) {
        std::cerr << "BŁĄD: Nie znaleziono żadnych urządzeń obsługujących HIP!" << std::endl;
        return false;
    }

    std::cout << "Znaleziono " << deviceCount << " urządzenie/a HIP.\n";

    /*for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t deviceProp;
        HIP_CHECK(hipGetDeviceProperties(&deviceProp, i));
        
        std::cout << "Urządzenie [" << i << "]: " << deviceProp.name << "\n";
        std::cout << "  Architektura (GCN/RDNA): " << deviceProp.gcnArchName << "\n";
        std::cout << "  Całkowita pamięć VRAM: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        std::cout << "  Max wątków na blok: " << deviceProp.maxThreadsPerBlock << "\n";
    }*/
    
    // Wybieramy domyślną kartę (indeks 0)
    HIP_CHECK(hipSetDevice(0));
    std::cout << "Pomyślnie podpięto do GPU 0.\n";
    std::cout << "===============================\n\n";
    
    return true;
}

// Destruktor (Zwalnianie pamięci z karty graficznej)

HipRaytracer::~HipRaytracer() {
    if (m_module) {
        hipModuleUnload(m_module);
        std::cout << "[HIP] Zwalnianie pamięci: Usunięto moduł z pamięci VRAM.\n";
    }
}
//NEW
// 2. Ładowanie skompilowanego modułu OSL (pliku binarnego HSACO dla AMD)
bool HipRaytracer::load_shader(const GPUShaderModuleDesc& desc) {
    if (!desc.data_ptr || desc.data_size == 0) {
        std::cerr << "[Błąd HIP] Pusty moduł przekazany do load_shader()!\n";
        return false;
    }

    // W PODEJŚCIU AOT OTRZYMUJEMY JUŻ GOTOWY KOD HSACO
    std::cout << "[HIP] Ładowanie gotowego modułu HSACO z pamięci (" << desc.data_size << " bajtów)...\n";

    // Używamy hipModuleLoadData, żeby załadować kernel PROSTO Z PAMIĘCI RAM.
    HIP_CHECK(hipModuleLoadData(&m_module, desc.data_ptr));

    // Pobierz wskaźnik na kernel
    std::cout << "[HIP] Szukam kernela: osl_kernel\n";
    hipError_t err = hipModuleGetFunction(&m_kernel, m_module, "osl_kernel");
    if (err != hipSuccess) {
        std::cerr << "BŁĄD: Nie znaleziono kernela 'osl_kernel'. HIP: " << hipGetErrorString(err) << "\n";
        return false;
    }

    std::cout << "[HIP] Załadowano kernel: osl_kernel\n";
    return true;
}

// 3. Uruchomienie kernela renderującego na GPU

void HipRaytracer::render(int width, int height) {

    std::cout << "[HIP] Rozpoczęcie renderowania. Rozdzielczość: " 

              << width << "x" << height << "\n";

    

    if (!m_kernel) {

        std::cerr << "[Błąd HIP] Kernel nie jest załadowany! Przerywam renderowanie.\n";

        return;

    }

             

    // 1. Alokacja pamięci dla obrazka wyjściowego

    size_t buffer_size = width * height * 3 * sizeof(float);

    hipDeviceptr_t d_output;

    HIP_CHECK_VOID(hipMalloc((void**)&d_output, buffer_size));



    // 2. Alokacja zerowych ShaderGlobals na GPU 

    size_t sg_size = width * height * 256; 

    void* d_shaderglobals = nullptr;

    HIP_CHECK_VOID(hipMalloc(&d_shaderglobals, sg_size));

    HIP_CHECK_VOID(hipMemset(d_shaderglobals, 0, sg_size));



    // 3. Skonfigurowanie bloków i siatki wątków

    dim3 blockSize(16, 16);

    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,

                  (height + blockSize.y - 1) / blockSize.y);



    // 4. PRZYGOTOWANIE ARGUMENTÓW ZGODNYCH Z NOWYM WRAPPEREM LLVM 

    void* d_groupdata = nullptr;

    void* userdata_base_ptr = nullptr;

    int shadeindex = 0;

    void* interactive_params_ptr = nullptr;



    void* args[] = {

        &d_shaderglobals,        // 1. ShaderGlobals*

        &d_groupdata,            // 2. void* groupdata

        &userdata_base_ptr,      // 3. void* userdata

        &d_output,               // 4. void* output_base

        &shadeindex,             // 5. int shadeindex

        &interactive_params_ptr, // 6. void* interactive_params

        &width                   // 7. int width

    };



    // 5. Uruchomienie kernela

    HIP_CHECK_VOID(hipModuleLaunchKernel(

        m_kernel,

        gridSize.x, gridSize.y, gridSize.z,

        blockSize.x, blockSize.y, blockSize.z,

        0, 0, // Pamięć współdzielona i strumień domyślny

        args, nullptr

    ));



    // 6. Czekamy aż karta skończy renderować

    HIP_CHECK_VOID(hipDeviceSynchronize());

    std::cout << "[HIP] Kernel zakończył pracę.\n";

    

    if (m_host_buffer != nullptr) {

    std::cout << "[HIP] Kopiowanie wyrenderowanego obrazu prosto do pamieci RAM hosta...\n";

    HIP_CHECK_VOID(hipMemcpy(m_host_buffer, (void*)d_output, buffer_size, hipMemcpyDeviceToHost));

} else {

    std::cerr << "[HIP] OSTRZEZENIE: m_host_buffer jest pusty! Wyniki znikna w prozni.\n";

}



    // POBRANIE WYNIKÓW Z VRAM NA CPU 

    std::cout << "[HIP] Kopiowanie wyrenderowanego obrazu do pamięci RAM...\n";

    std::vector<float> h_output(width * height * 3, 0.0f); // Wektor na dane na CPU

    

HIP_CHECK_VOID(hipMemcpy(h_output.data(), (void*)d_output, buffer_size, hipMemcpyDeviceToHost));
    

    // Wypiszmy wartość pierwszego wyrenderowanego piksela w lewym górnym rogu

    std::cout << "\n=== [WYNIKI RENDEROWANIA] ===\n";

    std::cout << "Piksel [0,0] RGB: (" 

              << h_output[0] << ", " 

              << h_output[1] << ", " 

              << h_output[2] << ")\n";



    // Wypiszmy wartość piksela ze środka ekranu

    int mid_x = width / 2;

    int mid_y = height / 2;

    int mid_idx = (mid_y * width + mid_x) * 3;

    

    std::cout << "Piksel [" << mid_x << "," << mid_y << "] RGB: (" 

              << h_output[mid_idx] << ", " 

              << h_output[mid_idx + 1] << ", " 

              << h_output[mid_idx + 2] << ")\n";

    std::cout << "=============================\n\n";



    // --- WYMUSZONY ZAPIS OBRAZU NA DYSK PRZEZ OIIO ---

    std::string custom_output = "sukces_hip.png";

    auto out_file = OIIO::ImageOutput::create(custom_output);

    if (out_file) {

        // Definiujemy strukturę obrazu: szerokość, wysokość, 3 kanały (RGB), typ float

        OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::FLOAT);

        out_file->open(custom_output, spec);

        out_file->write_image(OIIO::TypeDesc::FLOAT, h_output.data());

        out_file->close();

        std::cout << "[HIP SUCCESS] Obraz został pomyślnie zapisany w: " << custom_output << "\n";

    } else {

        std::cerr << "[BŁĄD HIP] OIIO nie mogło utworzyć pliku: " << custom_output << "\n";

    }



    // 7. Sprzątanie VRAM

    HIP_CHECK_VOID(hipFree((void*)d_output));

    HIP_CHECK_VOID(hipFree(d_shaderglobals));

} 

