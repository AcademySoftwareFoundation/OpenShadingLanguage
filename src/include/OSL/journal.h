// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <list>
#ifdef __CUDACC__
#    include <cuda/atomic>
#endif

#include <OSL/encodedtypes.h>
#include <OSL/oslconfig.h>


OSL_NAMESPACE_ENTER

/// Journal Buffer approach:  each thread gets its own chain of pages within
/// the shared journal buffer for recording errors, warnings, etc.
/// By maintaining a separate chain per thread we can record errors (etc.) in a
/// non-blocking fashion. Shade index is recorded for use by error reporter to
/// sort messages but it is not used at present.
/// Using a Journal buffer is up to a Renderer.  The Renderer use the new
/// customization points (rs_errorfmt, rs_filefmt, rs_printfmt, rs_warningfmt
/// or their virtual function equivalents) to interact with a journal::Writer
/// and journal buffer owned by its RenderState.  The Renderer would of have to
/// allocate and initialized the journal buffer with the requested page size
/// and number of threads before executing a shader. After a shader is done
/// executing, the contents of the journal buffer can be transfered back to the
/// host and processed with a journal::Reader, which reports them through a
/// journal::Reporter's virtual interface. Intent is for Renderers's to use the
/// Journal buffer and provide their own overriden version of the
/// journal::Reporter.  Testshade provides example usage.
/// For legacy purposes, the default virtual RendererServices will route errors
/// to ShadingSystem's OSL::ErrorHandler, but intent is for Renderer's to
/// switch over to using a Journal buffer or their own custom logging.

namespace journal {

namespace pvt {
//Per thread information for its page
struct alignas(64) PageInfo {
    uint32_t pos;
    uint32_t remaining;
    uint32_t warning_count;
};

struct Organization  //Initial bookkeeping
{
    int thread_count;
    uint32_t buf_size;
    uint32_t page_size;

// rename free_pos bytes_used;
#ifdef __CUDACC__
    using AtomicUint32 = cuda::std::atomic<std::uint32_t>;
#else
    using AtomicUint32 = std::atomic<std::uint32_t>;
#endif
    alignas(64)
        AtomicUint32 free_pos;  // cache line alignment to avoid false sharing
    alignas(64) AtomicUint32
        additional_bytes_required;  // when the journal is full, how many more bytes would of been needed
    alignas(64) AtomicUint32
        exceeded_page_size;  // when the a single entry exceeds the page size, we can't record it, but we can track how space it needed

    uint32_t calc_end_of_page_infos() const
    {
        return sizeof(PageInfo) * thread_count + sizeof(Organization);
    }
    uint32_t calc_head_pos(int thread_index) const
    {
        return calc_end_of_page_infos() + (thread_index * page_size);
    }

    PageInfo& get_pageinfo(int thread_index)
    {
        return reinterpret_cast<PageInfo*>(
            reinterpret_cast<uint8_t*>(this)
            + sizeof(Organization))[thread_index];
    }
};

enum class Content : uint8_t {
    PageTransition,
    // Shader Language generated
    Error,
    Warning,
    Print,
    FilePrint
};

static_assert(sizeof(PageInfo) % 64 == 0, "PageInfo needs to be cache aligned");
}  // namespace pvt


/// Call before launching shaders or after processing
/// returns: true for success,
///          false if buf_size can not accomadate at least 1 page per thread
OSLEXECPUBLIC bool
initialize_buffer(uint8_t* const buffer, uint32_t buf_size_,
                  uint32_t page_size_, int thread_count_);

/// Abstract base class intended for Renders to override and handle messages
/// decoded from a journal buffer
class OSLEXECPUBLIC Reporter {
    //Methods will NOT be called by multiple threads (no synchronization needed)
public:
    virtual ~Reporter() {}
    virtual void report_error(int thread_index, int shade_index,
                              const OSL::string_view& message)
        = 0;
    virtual void report_warning(int thread_index, int shade_index,
                                const OSL::string_view& message)
        = 0;
    virtual void report_print(int thread_index, int shade_index,
                              const OSL::string_view& message)
        = 0;
    virtual void report_file_print(int thread_index, int shade_index,
                                   const OSL::string_view& filename,
                                   const OSL::string_view& message)
        = 0;
};

/// Utility class to look for repeated errors or warnings over
/// a limited history window
class OSLEXECPUBLIC TrackRecentlyReported {
public:
    TrackRecentlyReported(bool limit_errors            = true,
                          int error_history_capacity   = 32,
                          bool limit_warnings          = true,
                          int warning_history_capacity = 32);

    bool shouldReportError(const OSL::string_view& message);
    bool shouldReportWarning(const OSL::string_view& message);

protected:
    const bool m_limit_errors;
    const int m_error_history_capacity;
    const bool m_limit_warnings;
    const int m_warning_history_capacity;
    std::list<std::string> m_errseen, m_warnseen;
};

/// Concrete Reporter that utilizes a TrackRecentlyReported to filter and
/// forwards to a legacy OSL::ErrorHandler
class OSLEXECPUBLIC Report2ErrorHandler : public Reporter {
public:
    Report2ErrorHandler(OSL::ErrorHandler* eh, TrackRecentlyReported& tracker);
    void report_error(int thread_index, int shade_index,
                      const OSL::string_view& message) override;
    void report_warning(int thread_index, int shade_index,
                        const OSL::string_view& message) override;
    void report_print(int thread_index, int shade_index,
                      const OSL::string_view& message) override;
    void report_file_print(int thread_index, int shade_index,
                           const OSL::string_view& filename,
                           const OSL::string_view& message) override;

protected:
    OSL::ErrorHandler* m_eh;
    TrackRecentlyReported& m_tracker;
};

/// Reader is a lightweight class meant to be constructed as a local variable
/// with an already initalized (and populated) journal buffer and a concreate
/// Reporter.  Calling process will decode all of the messages in the journal
/// calling the appropriate methods on the provided Reportor.
/// NOTE: Renderer is expected to call initialize_buffer after Reader::process
///       before reusing the buffer for additional shading.
class OSLEXECPUBLIC Reader {
public:
    Reader(const uint8_t* buffer_, Reporter& reporter);
    void process();

private:
    void process_entries_for_thread(int thread_index);

    const uint8_t* const m_buffer;   //Read  from this?
    const pvt::Organization& m_org;  //
    const pvt::PageInfo* m_pageinfo_by_thread_index;
    Reporter& m_reporter;
};

/// Writer is a lightweight class meant to be constructed as a local variable
/// with an already initalized journal buffer and used to record error,
/// warnings, prints, or fprints into the buffer and then go out of scope.
/// Writer implementation entirely inlinable and intended for use on devices.
class OSLEXECPUBLIC Writer {
public:
    Writer(void* buffer)
        : m_buffer(static_cast<uint8_t*>(buffer))
        , m_org(*(reinterpret_cast<pvt::Organization*>(buffer)))
        , m_pageinfo_by_thread_index(reinterpret_cast<pvt::PageInfo*>(
              m_buffer + sizeof(pvt::Organization)))
    {
    }

    bool record_errorfmt(int thread_index, int shade_index,
                         OSL::ustringhash fmt_specification, int32_t arg_count,
                         const EncodedType* arg_types, uint32_t arg_values_size,
                         uint8_t* arg_values)
    {
        return write_entry(thread_index, shade_index, pvt::Content::Error,
                           fmt_specification, OSL::ustringhash {} /*filename*/,
                           arg_count, arg_types, arg_values_size, arg_values);
    }

    bool record_warningfmt(int max_warnings_per_thread, int thread_index,
                           int shade_index, OSL::ustringhash fmt_specification,
                           int32_t arg_count, const EncodedType* arg_types,
                           uint32_t arg_values_size, uint8_t* arg_values)
    {
        auto& info = m_pageinfo_by_thread_index[thread_index];

        if (static_cast<int>(info.warning_count) >= max_warnings_per_thread) {
            return false;
        }
        ++info.warning_count;
        return write_entry(thread_index, shade_index, pvt::Content::Warning,
                           fmt_specification, OSL::ustringhash {} /*filename*/,
                           arg_count, arg_types, arg_values_size, arg_values);
    }

    bool record_printfmt(int thread_index, int shade_index,
                         OSL::ustringhash fmt_specification, int32_t arg_count,
                         const EncodedType* arg_types, uint32_t arg_values_size,
                         uint8_t* arg_values)
    {
        return write_entry(thread_index, shade_index, pvt::Content::Print,
                           fmt_specification, OSL::ustringhash {} /*filename*/,
                           arg_count, arg_types, arg_values_size, arg_values);
    }

    bool record_filefmt(int thread_index, int shade_index,
                        OSL::ustringhash filename_hash,
                        OSL::ustringhash fmt_specification, int32_t arg_count,
                        const EncodedType* arg_types, uint32_t arg_values_size,
                        uint8_t* arg_values)
    {
        return write_entry(thread_index, shade_index, pvt::Content::FilePrint,
                           fmt_specification, filename_hash, arg_count,
                           arg_types, arg_values_size, arg_values);
    }

private:
    static constexpr size_t requiredForPageTransition()
    {
        return sizeof(pvt::Content::PageTransition) + sizeof(uint32_t);
    }
    bool allocatePage(int thread_index)
    {
        using pvt::Content;
        auto& info = m_pageinfo_by_thread_index[thread_index];

        OSL_ASSERT(info.remaining >= requiredForPageTransition());

        // pre check if next_pos >= buf_size to prevent continuing to
        // increment free_pos once its already past the end of a page
        if ((m_org.free_pos.load() + m_org.page_size) >= m_org.buf_size)
            return false;

        uint32_t next_pos = m_org.free_pos.fetch_add(m_org.page_size);
        // post check if next_pos >= buf_size because another thread could
        // have bumped free_pos
        if (next_pos >= m_org.buf_size)
            return false;

        // write page transition
        uint8_t* dest_ptr         = m_buffer + info.pos;
        constexpr Content content = Content::PageTransition;
        memcpy(dest_ptr, &content, sizeof(content));
        memcpy(dest_ptr + sizeof(content), &next_pos, sizeof(next_pos));
        // Don't bother update page info for the memory used by the transition
        // because we are about to overwrite it.

        // Update page info to point to newly allocated page
        info.pos           = next_pos;
        info.remaining     = m_org.page_size;
        info.warning_count = 0;
        return true;
    }

    bool write_entry(int thread_index, int shade_index, pvt::Content content,
                     OSL::ustringhash fmt_specification,
                     OSL::ustringhash filename, int32_t arg_count,
                     const EncodedType* arg_types, uint32_t arg_values_size,
                     uint8_t* arg_values)
    {
        using pvt::Content;

        uint64_t fname_hash;
        // calc required size for this entry
        uint64_t fmt_spec_hash  = fmt_specification.hash();
        uint32_t required_bytes = sizeof(Content) + sizeof(int)
                                  + sizeof(fmt_spec_hash) + sizeof(arg_count)
                                  + sizeof(EncodedType) * arg_count
                                  + arg_values_size;

        if (filename.hash()) {
            fname_hash = filename.hash();
            required_bytes += sizeof(fname_hash);
        }


        if (required_bytes > m_org.page_size) {
            uint32_t exceeded = m_org.exceeded_page_size;
            while (required_bytes > exceeded) {
                if (m_org.exceeded_page_size.compare_exchange_weak(
                        exceeded, required_bytes)) {
                    break;
                }
            }
            return false;
        }

        auto& info = m_pageinfo_by_thread_index[thread_index];

        if ((info.remaining - requiredForPageTransition()) < required_bytes) {
            // TODO (extra credit): get fancy spanning entry over multiple pages
            if (!allocatePage(thread_index)) {
                m_org.additional_bytes_required += required_bytes;
                return false;
            }
        }

        OSL_ASSERT(info.remaining >= required_bytes);
        uint8_t* dest_ptr = m_buffer + info.pos;

        memcpy(dest_ptr, &content, sizeof(content));
        memcpy(dest_ptr + sizeof(content), &shade_index, sizeof(shade_index));
        memcpy(dest_ptr + sizeof(content) + sizeof(shade_index), &fmt_spec_hash,
               sizeof(fmt_spec_hash));
        memcpy(dest_ptr + sizeof(content) + sizeof(shade_index)
                   + sizeof(fmt_spec_hash),
               &arg_count, sizeof(arg_count));
        memcpy(dest_ptr + sizeof(content) + sizeof(shade_index)
                   + sizeof(fmt_spec_hash) + sizeof(arg_count),
               arg_types, sizeof(EncodedType) * arg_count);
        memcpy(dest_ptr + sizeof(content) + sizeof(shade_index)
                   + sizeof(fmt_spec_hash) + sizeof(arg_count)
                   + sizeof(EncodedType) * arg_count,
               arg_values, arg_values_size);
        if (content == Content::FilePrint) {
            uint64_t filename_hash = filename.hash();
            memcpy(dest_ptr + sizeof(content) + sizeof(shade_index)
                       + sizeof(fmt_spec_hash) + sizeof(arg_count)
                       + sizeof(EncodedType) * arg_count + arg_values_size,
                   &filename_hash, sizeof(filename_hash));
        }

        info.remaining -= required_bytes;
        info.pos += required_bytes;
        return true;
    }

    uint8_t* m_buffer;
    pvt::Organization& m_org;
    pvt::PageInfo* m_pageinfo_by_thread_index;
};

}  // namespace journal

OSL_NAMESPACE_EXIT