// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <OSL/encodedtypes.h>
#include <OSL/journal.h>
#include <OSL/oslconfig.h>

#include <fstream>
#include <iostream>

OSL_NAMESPACE_ENTER

int
decode_message(uint64_t format_hash, int32_t arg_count,
               const EncodedType* arg_types, const uint8_t* arg_values,
               std::string& built_str)
{
    // set max size of each output string
    // replacement region buffer
    char rr_buf[128];
    built_str.clear();

    const char* format = OSL::ustring::from_hash(format_hash).c_str();
    OSL_ASSERT(format != nullptr
               && "The string should have been a valid ustring");
    const int len = static_cast<int>(strlen(format));

    int arg_index                  = 0;
    int arg_offset                 = 0;
    constexpr size_t rs_max_length = 1024;
    char replacement_str[rs_max_length];
    for (int j = 0; j < len;) {
        // If we encounter a '%', then we'll copy the format string to 'fmt_string'
        // and provide that to printf() directly along with a pointer to the argument
        // we're interested in printing.
        char cur_char = format[j++];
        if (cur_char == '{') {
            bool is_rr_complete = false;
            rr_buf[0]           = cur_char;
            int rr_len          = 1;
            do {
                char next_char = format[j++];

                if ((rr_len == 1) && next_char == '{') {
                    OSL_DASSERT((rr_buf[0] == '{'));
                    // Not a replacement region, but an escaped {
                    // We don't want to copy over {{, just {
                    // so we will just eat the 2nd {
                    break;
                }

                rr_buf[rr_len++] = next_char;

                if (next_char == '}') {
                    is_rr_complete = true;
                }
            } while (!is_rr_complete && j < len
                     && rr_len < static_cast<int>(sizeof(rr_buf)));
            OSL::string_view replacement_region { rr_buf, size_t(rr_len) };

            if (!is_rr_complete) {
                built_str += replacement_region;
            } else {
                if (arg_index < arg_count) {
                    EncodedType arg_type = arg_types[arg_index];


#if OSL_GNUC_VERSION >= 90000
// ignore -Wclass-memaccess to avoid "error: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘class OpenImageIO_v2_3::ustringhash’
//                                    with no trivial copy-assignment; use copy-assignment or copy-initialization instead"
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wclass-memaccess"
                    // TODO: remove diagnostic workaround once OIIO::ustringhash is changed such that
                    //       static_assert(std::is_trivially_copyable<OSL::ustringhash>::value, "Make ustringhash::ustringhash(const ustringhash&) = default;");
#endif
                    switch (arg_type) {
                    case EncodedType::kUstringHash: {
                        OSL::ustringhash arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        const char* arg_string = arg_value.c_str();
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length + 1,
                                                          replacement_region,
                                                          arg_string);
                        *result.out = '\0';
                    } break;
                    case EncodedType::kInt32: {
                        int32_t arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;
                    case EncodedType::kFloat: {
                        float arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;
                    case EncodedType::kDouble: {
                        double arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;
                    case EncodedType::kInt64: {
                        int64_t arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;
                    case EncodedType::kUInt32: {
                        uint32_t arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;
                    case EncodedType::kUInt64: {
                        uint64_t arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;

                    case EncodedType::kPointer: {
                        const void* arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;

                    case EncodedType::kTypeDesc: {
                        OSL::TypeDesc arg_value;
                        memcpy(&arg_value, &arg_values[arg_offset],
                               sizeof(arg_value));
                        auto result = OSL::fmtformat_to_n(replacement_str,
                                                          rs_max_length,
                                                          replacement_region,
                                                          arg_value);
                        *result.out = '\0';
                    } break;
                    default: OSL_ASSERT(0 && "unhandled EncodedType");
                    };

#if OSL_GNUC_VERSION >= 90000
#    pragma GCC diagnostic pop
#endif

                    arg_offset += pvt::size_of_encoded_type(arg_type);
                    ++arg_index;
                    built_str += replacement_str;
                }
            }
        } else {
            //if we are not "{" for printf; all others contain {} as part of the message
            built_str += cur_char;
            if (cur_char == '}' && j < len && format[j] == '}') {
                // }} should just be output as }
                // so skip the next character
                ++j;
            }
        }  //else within for

    }  //initial for loop
    return arg_offset;
}

namespace journal {

bool
initialize_buffer(uint8_t* const buffer, uint32_t buf_size, uint32_t page_size,
                  int thread_count)

{
    using namespace journal::pvt;
    auto& org        = *(reinterpret_cast<Organization*>(buffer));
    org.thread_count = thread_count;
    org.buf_size     = buf_size;
    org.page_size    = page_size;

    org.additional_bytes_required = 0;
    org.exceeded_page_size        = 0;
    org.free_pos                  = org.calc_end_of_page_infos()
                   // Pre-allocate 1 page per thread
                   + org.page_size * org.thread_count;

    if (org.free_pos > org.buf_size) {
        return false;
    }

    // Populate each thread's initial PageInfo
    for (int thread_index = 0; thread_index < org.thread_count;
         ++thread_index) {
        PageInfo& info = org.get_pageinfo(thread_index);

        info.pos           = org.calc_head_pos(thread_index);
        info.remaining     = org.page_size;
        info.warning_count = 0;
    }
    return true;
}

TrackRecentlyReported::TrackRecentlyReported(bool limit_errors,
                                             int error_history_capacity,
                                             bool limit_warnings,
                                             int warning_history_capacity)
    : m_limit_errors(limit_errors)
    , m_error_history_capacity(error_history_capacity)
    , m_limit_warnings(limit_warnings)
    , m_warning_history_capacity(warning_history_capacity)
{
}

bool
TrackRecentlyReported::shouldReportError(const OSL::string_view& message)
{
    int n = 0;
    if (!m_limit_errors)
        return true;
    for (auto&& s : m_errseen) {
        if (s == message)
            return false;
        ++n;
    }
    if (n >= m_error_history_capacity)
        m_errseen.pop_front();
    m_errseen.push_back(message);
    return true;
}

bool
TrackRecentlyReported::shouldReportWarning(const OSL::string_view& message)
{
    int n = 0;
    if (!m_limit_errors)
        return true;
    for (auto&& s : m_warnseen) {
        if (s == message)
            return false;
        ++n;
    }
    if (n >= m_warning_history_capacity)
        m_warnseen.pop_front();
    m_warnseen.push_back(message);
    return true;
}

Report2ErrorHandler::Report2ErrorHandler(OSL::ErrorHandler* eh,
                                         TrackRecentlyReported& tracker)
    : m_eh(eh), m_tracker(tracker)
{
}

void
Report2ErrorHandler::report_error(int thread_index, int shade_index,
                                  const OSL::string_view& message)
{
    if (m_tracker.shouldReportError(message)) {
        m_eh->error(std::string(message));
    }
}

void
Report2ErrorHandler::report_warning(int thread_index, int shade_index,
                                    const OSL::string_view& message)
{
    if (m_tracker.shouldReportWarning(message)) {
        m_eh->warning(std::string(message));
    }
}

void
Report2ErrorHandler::report_print(int thread_index, int shade_index,
                                  const OSL::string_view& message)
{
    m_eh->message(message);
}

void
Report2ErrorHandler::report_file_print(int thread_index, int shade_index,
                                       const OSL::string_view& filename,
                                       const OSL::string_view& message)
{
    // NOTE: behavior change for OSL runtime, we will no longer open files by default
    // but instead just prefix the fprintf message with the filename and pass it along
    // as a regular message.
    // A renderer is free to override report_fprintf and open files under its own purview

    m_eh->message(OSL::fmtformat("{}:{}", filename, message));
}

Reader::Reader(const uint8_t* buffer_, Reporter& reporter)
    : m_buffer(buffer_)
    , m_org(*(reinterpret_cast<const pvt::Organization*>(buffer_)))
    , m_pageinfo_by_thread_index(reinterpret_cast<const pvt::PageInfo*>(
          buffer_ + sizeof(pvt::Organization)))
    , m_reporter(reporter)
{
}

void
Reader::process()
{
    const int tc = m_org.thread_count;
    for (int thread_index = 0; thread_index < tc; ++thread_index) {
        process_entries_for_thread(thread_index);
    }
    if (m_org.additional_bytes_required != 0) {
        std::string overfill_message = OSL::fmtformat(
            "Journal sized {} bytes couldn't capture all prints, warnings, and errors.  Additional {} bytes would be required",
            m_org.buf_size, m_org.additional_bytes_required.load());
        m_reporter.report_warning(-1, -1, overfill_message);
    }
    if (m_org.exceeded_page_size != 0) {
        std::string exceeded_message = OSL::fmtformat(
            "Journal page size {} exceeded, largest individual message sized {} bytes.  Consider increasing your page size.",
            m_org.page_size, m_org.exceeded_page_size.load());
        m_reporter.report_warning(-1, -1, exceeded_message);
    }
}

void
Reader::process_entries_for_thread(int thread_index)
{
    uint32_t read_pos = m_org.calc_head_pos(thread_index);
    const auto& info  = m_pageinfo_by_thread_index[thread_index];
    // We are done processing entries when our read_pos reaches the end_pos;
    uint32_t end_pos = info.pos;

    using pvt::Content;

    std::string message;
    message.reserve(m_org.page_size);
    int shade_index;
    auto decodeMessage = [&](const uint8_t* src_ptr) -> void {
        memcpy(&shade_index, src_ptr + sizeof(Content), sizeof(shade_index));
        uint64_t format_hash;
        memcpy(&format_hash, src_ptr + sizeof(Content) + sizeof(shade_index),
               sizeof(format_hash));
        int32_t arg_count;
        memcpy(&arg_count,
               src_ptr + sizeof(Content) + sizeof(shade_index)
                   + sizeof(format_hash),
               sizeof(arg_count));

        auto arg_types = reinterpret_cast<const EncodedType*>(
            src_ptr + sizeof(Content) + sizeof(shade_index)
            + sizeof(format_hash) + sizeof(arg_count));

        const uint8_t* arg_values = src_ptr + sizeof(Content)
                                    + sizeof(shade_index) + sizeof(format_hash)
                                    + sizeof(arg_count)
                                    + sizeof(EncodedType) * arg_count;
        int arg_values_size = decode_message(format_hash, arg_count, arg_types,
                                             arg_values, message);
        read_pos += sizeof(Content) + sizeof(shade_index) + sizeof(format_hash)
                    + sizeof(arg_count) + sizeof(EncodedType) * arg_count
                    + arg_values_size;
    };

    while (read_pos != end_pos) {
        const uint8_t* src_ptr = m_buffer + read_pos;

        Content content;
        memcpy(&content, src_ptr, sizeof(content));
        switch (content) {
        case Content::PageTransition: {
            uint32_t next_pos;
            memcpy(&next_pos, src_ptr + sizeof(content), sizeof(next_pos));

            read_pos = next_pos;
            break;
        }

        case Content::Error: {
            decodeMessage(src_ptr);
            m_reporter.report_error(thread_index, shade_index, message);
            break;
        }

        case Content::Warning: {
            decodeMessage(src_ptr);
            m_reporter.report_warning(thread_index, shade_index, message);
            break;
        }

        case Content::Print: {
            decodeMessage(src_ptr);
            m_reporter.report_print(thread_index, shade_index, message);
            break;
        }

        case Content::FilePrint: {
            decodeMessage(src_ptr);
            uint64_t filname_hash;
            memcpy(&filname_hash, m_buffer + read_pos, sizeof(filname_hash));
            read_pos += sizeof(filname_hash);
            m_reporter.report_file_print(thread_index, shade_index,
                                         OSL::ustring::from_hash(filname_hash),
                                         message);
            break;
        }
        };
    }
}

}  //namespace journal

OSL_NAMESPACE_EXIT
