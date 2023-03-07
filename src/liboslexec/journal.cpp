// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#include <OSL/oslconfig.h>
#include <OSL/encodedtypes.h>
#include <OSL/journal.h>

OSL_NAMESPACE_ENTER

namespace journal
{


// TODO: update hashcode variable name to format_hash
void construct_message(uint64_t format_hash, int32_t arg_count, 
                                const EncodedType *argTypes/*(&argTypes)[128]*/, uint32_t argValuesSize, 
                                uint8_t *argValues/*(&argValues)[4096]*/, std::string &built_str)
{

        // set max size of each output string
        // replacement region buffer
        char rr_buf[128];
        built_str.clear();

        const char* format = OSL::ustring::from_hash(format_hash).c_str();

        OSL_ASSERT(format != nullptr
                   && "The string should have been a valid ustring");
        const int len = static_cast<int>(strlen(format));

        int arg_index = 0;
        int arg_offset = 0;

        constexpr size_t rs_max_length = 1024;
        char replacement_str[rs_max_length];

      
        for (int j = 0; j < len; j++) {
            // If we encounter a '%', then we'll copy the format string to 'fmt_string'
            // and provide that to printf() directly along with a pointer to the argument
            // we're interested in printing.
            if (format[j] == '{' && (j==0 || format[j-1] != '\\')) {
                int rr_len=0;
                for (;j < len && rr_len < static_cast<int>(sizeof(rr_buf));++j, ++rr_len) 
                {
                    
                    rr_buf[rr_len] = format[j]; 

                } while (j < len && rr_len < static_cast<int>(sizeof(rr_buf)) );
                OSL::string_view replacementRegion{rr_buf, size_t(rr_len)};
                 //OSL::string_view replacementRegion{rr_buf, size_t(rr_len)};
                if (arg_index < arg_count) {
                    EncodedType arg_type = argTypes[arg_index];

                    // Avoid "error: ‘void* memcpy(void*, const void*, size_t)’ writing to an object of type ‘class OpenImageIO_v2_3::ustringhash’ with no trivial copy-assignment; use copy-assignment or copy-initialization instead"
                    // TODO: remove diagnostic workaround once OIIO::ustringhash is changed such that
                    //       static_assert(std::is_trivially_copyable<OSL::ustringhash>::value, "Make ustringhash::ustringhash(const ustringhash&) = default;");
                    #pragma GCC diagnostic push
                    #pragma GCC diagnostic ignored "-Wclass-memaccess"
                    switch(arg_type) {
                        case EncodedType::kUstringHash: {
                            OSL::ustringhash arg_value;
                            memcpy(&arg_value, &argValues[arg_offset], sizeof(arg_value));
                            const char* arg_string  = arg_value.c_str();

                            auto result = OSL::fmtformat_to_n(replacement_str, rs_max_length, 
                                replacementRegion, arg_string);
                            *result.out = '\0';
                           

                        }
                        break;
                        case EncodedType::kInt32: {
                            int32_t arg_value;
                            memcpy(&arg_value, &argValues[arg_offset], sizeof(arg_value));
                            auto result = OSL::fmtformat_to_n(replacement_str, 4, replacementRegion, arg_value);
                            *result.out = '\0';

                        }
                        break;
                        case EncodedType::kFloat: {
                            float arg_value;
                            memcpy(&arg_value, &argValues[arg_offset], sizeof(arg_value));
                            auto result = OSL::fmtformat_to_n(replacement_str, rs_max_length, replacementRegion, arg_value);
                            *result.out = '\0';
                        }
                        break;
                        case EncodedType::kDouble: {
                            double arg_value;
                            memcpy(&arg_value, &argValues[arg_offset], sizeof(arg_value));
                            auto result = OSL::fmtformat_to_n(replacement_str, rs_max_length, replacementRegion, arg_value);
                            *result.out = '\0';
                        }
                        break;
                    };

                    #pragma GCC diagnostic pop

                    arg_offset += SizeByEncodedType[static_cast<int>(arg_type)];
                    ++arg_index;
                    built_str += replacement_str;                    
                }
            } else {
                built_str += format[j];
            }
        }  
    }


bool initialize_buffer(uint8_t * const buffer, uint32_t buf_size_, uint32_t page_size_, int thread_count_)

{

    Organization & org = *(reinterpret_cast<Organization *>(buffer));
    org.thread_count = thread_count_;
    org.buf_size = buf_size_;
    org.page_size = page_size_;


    PageInfo *pageinfo_by_thread_index = reinterpret_cast<PageInfo *>(buffer + sizeof(Organization));
    
    org.buf_pos = org.calc_end_of_page_infos() + org.page_size * org.thread_count;

 

    if (org.buf_pos > org.buf_size) {
        
        return false;
    }

    for(int thread_index=0;thread_index < org.thread_count; ++thread_index) {
        PageInfo &info = pageinfo_by_thread_index[thread_index];
        info.pos = org.calc_head_pos(thread_index);
        info.remaining = org.page_size;
        
    }
    

    return true;
}


TrackRecentlyReported::TrackRecentlyReported(bool limit_errors = true, int max_error_count=32,
                        bool limit_warnings = true, int max_warning_count=32
    )
    : m_limit_errors(limit_errors)
    , m_max_error_count(max_error_count)
    , m_limit_warnings(limit_warnings)
    , m_max_warning_count(max_warning_count)
    {

    }

bool TrackRecentlyReported::shouldReportError(const OSL::string_view & message)
    {
        int n = 0;
        if (!m_limit_errors)
            return true;
        for (auto&& s : m_errseen) {
            if (s == message)
                return false;
            ++n;
        }
        if (n >= m_max_error_count)
            m_errseen.pop_front();
        m_errseen.push_back(message);
        return true;
    }

bool TrackRecentlyReported::shouldReportWarning(const OSL::string_view & message)
    {
        int n = 0;
        if (!m_limit_errors)
            return true;
        for (auto&& s : m_warnseen) {
            if (s == message)
                return false;
            ++n;
        }
        if (n >= m_max_error_count)
            m_warnseen.pop_front();
        m_warnseen.push_back(message);
        return true;
    }

Report2ErrorHandler::Report2ErrorHandler(OSL::ErrorHandler *eh, TrackRecentlyReported m_tracker) : m_eh(eh), m_tracker(m_tracker) {}
  
void Report2ErrorHandler::report_error(int thread_index, int shade_index, const OSL::string_view & message) 
{ 
   if (m_tracker.shouldReportError(message))
   {
       m_eh->error(std::string(message));
   }
}

void Report2ErrorHandler::report_warning(int thread_index, int shade_index, const OSL::string_view & message) 
{ 
       
    if (m_tracker.shouldReportWarning(message))
    {
        m_eh->warning(std::string(message));
    }        
}

void Report2ErrorHandler::report_print(int thread_index, int shade_index, const OSL::string_view & message) 
{ 
    m_eh->message(message); 
}

void Report2ErrorHandler::report_file_print(int thread_index, int shade_index, const OSL::string_view & filename, const OSL::string_view & message)  
{
    // NOTE: behavior change for OSL runtime, we will no longer open files by default
    // but instead just prefix the fprintf message with the filename and pass it along 
    // as a regular message.
    // A renderer is free to override report_fprintf and open files under its own purview
        
    m_eh->message(OSL::fmtformat("{}:{}", filename,  message)); 

}

Reader::Reader(const uint8_t * buffer_, Reporter &reporter)
    : m_buffer(buffer_), 
      m_org(*(reinterpret_cast<const Organization *>(buffer_))), 
      m_pageinfo_by_thread_index ( reinterpret_cast<const PageInfo *>(buffer_ + sizeof(Organization))),
      m_reporter(reporter)
    {}   

    void Reader::process()
    {        
        const int tc = m_org.thread_count;
        for (int thread_index=0; thread_index < tc; ++thread_index) {
            process_entries_for_thread(thread_index);
        }
    }

    void Reader::process_entries_for_thread(int thread_index)
    {
        uint32_t read_pos = m_org.calc_head_pos(thread_index);
        const PageInfo &info =  m_pageinfo_by_thread_index[thread_index];
        // We are done processing entries when our read_pos reaches the end_pos;
        uint32_t end_pos = info.pos;

        std::string message;
        message.reserve(m_org.page_size);
        int shade_index;
        auto decodeMessage = [&](const uint8_t *src_ptr)->void {
            uint64_t format_hash;
            int32_t arg_count;
            EncodedType arg_types[128];
            uint8_t arg_values[4096];
            memcpy(&shade_index, src_ptr + sizeof(Content), sizeof(shade_index));
            memcpy(&format_hash, src_ptr + sizeof(Content) + sizeof(shade_index), sizeof(format_hash));
            memcpy(&arg_count, src_ptr + sizeof(Content) + sizeof(shade_index) + sizeof(format_hash), sizeof(arg_count));
            memcpy(arg_types, src_ptr + sizeof(Content) + sizeof(shade_index) + sizeof(format_hash) + sizeof(arg_count), sizeof(EncodedType)*arg_count);
            uint32_t arg_values_size = 0u;
            for(int arg_index=0; arg_index < arg_count; ++arg_index) {
                arg_values_size += SizeByEncodedType[static_cast<int>(arg_types[arg_index])];
            }
            memcpy(arg_values, src_ptr + sizeof(Content) + sizeof(shade_index) + sizeof(format_hash) + sizeof(arg_count) + sizeof(EncodedType)*arg_count, arg_values_size);
            read_pos += sizeof(Content) + sizeof(shade_index) + sizeof(format_hash) + sizeof(arg_count) + sizeof(EncodedType)*arg_count + arg_values_size;

            const char* format_string = OSL::ustring::from_hash(format_hash).c_str();
            
          
            construct_message(format_hash, arg_count, &arg_types[0], arg_values_size, 
                                &arg_values[0], message);
        };
                
        while(read_pos !=end_pos) {
            const uint8_t *src_ptr = m_buffer + read_pos; 
          
            
            Content content;
            memcpy(&content, src_ptr, sizeof(content));
            switch(content) {
                case Content::PageTransition: {
                    uint32_t next_pos;
                    memcpy(&next_pos, src_ptr + sizeof(content), sizeof(content));
                    read_pos = next_pos;
                    }
                    break;
                case Content::Error: {
                    decodeMessage(src_ptr);
                    m_reporter.report_error(thread_index, shade_index, message);
                    }
                    break;
                case Content::Warning: {
                    decodeMessage(src_ptr);            
                    m_reporter.report_warning(thread_index, shade_index, message);
                    }
                    break;
                case Content::Print: {
                    decodeMessage(src_ptr);
                    m_reporter.report_print(thread_index, shade_index, message);
                    }
                    break;
                case Content::FilePrint: {
                    decodeMessage(src_ptr);
                    uint64_t filname_hash;
                    memcpy(&filname_hash, m_buffer + read_pos, sizeof(filname_hash));
                    read_pos += sizeof(filname_hash);
                    m_reporter.report_file_print(thread_index, shade_index, OSL::ustring::from_hash(filname_hash), message);
                }
                break;
            };
        }
    }

} //namespace journal

OSL_NAMESPACE_EXIT