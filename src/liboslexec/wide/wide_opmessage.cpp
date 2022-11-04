// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/oslconfig.h>

#include <OSL/batched_rendererservices.h>
#include <OSL/batched_shaderglobals.h>

#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER

namespace __OSL_WIDE_PVT {

OSL_USING_DATA_WIDTH(__OSL_WIDTH)
using BatchedRendererServices = OSL::BatchedRendererServices<__OSL_WIDTH>;
using WidthTag                = OSL::WidthOf<__OSL_WIDTH>;

#include "define_opname_macros.h"

struct MessageBlock {
    MessageBlock(ustring name, const TypeDesc& type, MessageBlock* next)
        : valid_mask(false)
        , get_before_set_mask(false)
        , name(name)
        , data(nullptr)
        , type(type)
        , next(next)
    {
    }

    MessageBlock(const MessageBlock&)            = delete;
    MessageBlock& operator=(const MessageBlock&) = delete;

    /// Some messages don't have data because getmessage() was called before setmessage
    /// (which is flagged as an error to avoid ambiguities caused by execution order)
    ///
    bool has_data() const { return data != nullptr; }

    // Place larger blocks at front of structure as they have alignment
    // requirements which could cause excess padding if smaller members
    // where placed ahead of them
    Block<int> wlayeridx;  ///< layer index where this was message was created
    Block<ustring>
        wsourcefile;  ///< source code file that contains the call that created this message
    Block<int>
        wsourceline;  ///< source code line that contains the call that created this message
    Mask valid_mask;           ///< which lanes of have been set
    Mask get_before_set_mask;  ///< which lanes had get called before a set, track to create errors if set is called later
    ustring name;              ///< name of this message
    char* data;  ///< actual data of the message (will never change once the message is created)
    TypeDesc
        type;  ///< what kind of data is stored here? FIXME: should be TypeSpec
    MessageBlock*
        next;  ///< linked list of messages (managed by MessageList below)

public:
    void import_data(MaskedData wsrcval, Mask lanes_to_populate, int layeridx,
                     ustring sourcefile, int sourceline)
    {
        Masked<int> alayeridx(wlayeridx, lanes_to_populate);
        Masked<ustring> asourcefile(wsourcefile, lanes_to_populate);
        Masked<int> asourceline(wsourceline, lanes_to_populate);
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            alayeridx[lane]   = layeridx;
            asourcefile[lane] = sourcefile;
            asourceline[lane] = sourceline;
        }
        if (!wsrcval.valid()) {
            // We always needed to copy the debug info (sourcefile, line, layer)
            // but we may not have any data to copy
            get_before_set_mask |= lanes_to_populate;
            return;
        }
        valid_mask |= lanes_to_populate;

        MaskedData dest(wsrcval.type(), /*has_derivs=*/false, lanes_to_populate,
                        data);
        dest.assign_val_from_wide(wsrcval.ptr());
    }

    void export_data(MaskedData wdestval)
    {
        wdestval.assign_val_from_wide(data);
    }
};

struct BatchedMessageList {
private:
    BatchedMessageBuffer& m_buffer;

public:
    BatchedMessageList(BatchedMessageBuffer& buffer) : m_buffer(buffer) {}

    BatchedMessageList(const BatchedMessageList&)            = delete;
    BatchedMessageList& operator=(const BatchedMessageList&) = delete;

    MessageBlock* list_head() const
    {
        return reinterpret_cast<MessageBlock*>(m_buffer.list_head);
    }

    void set_list_head(MessageBlock* new_list_head)
    {
        m_buffer.list_head = new_list_head;
    }

    MessageBlock* find(ustring name) const
    {
        for (MessageBlock* m = list_head(); m != nullptr; m = m->next)
            if (m->name == name)
                return m;  // name matches
        return nullptr;    // not found
    }


    void add(ustring name /*varying name*/, MaskedData wsrcval,
             Mask lanes_to_populate, int layeridx, ustring sourcefile,
             int sourceline)
    {
        constexpr size_t alignment = sizeof(float) * __OSL_WIDTH;
        set_list_head(
            new (m_buffer.message_data.alloc(sizeof(MessageBlock), alignment))
                MessageBlock(name, wsrcval.type(), list_head()));
        list_head()->data
            = m_buffer.message_data.alloc(wsrcval.val_size_in_bytes(),
                                          alignment);
        list_head()->import_data(wsrcval, lanes_to_populate, layeridx,
                                 sourcefile, sourceline);
    }
};

namespace {  // anonymous

OSL_NOINLINE void
impl_setmessage(BatchedShaderGlobals* bsg, ustring sourcefile, int sourceline,
                MaskedData wsrcval, int layeridx, ustring name,
                Mask matching_lanes)
{
    TypeDesc type   = wsrcval.type();
    bool is_closure = (type.basetype
                       == TypeDesc::UNKNOWN);  // secret code for closure
    if (is_closure) {
        OSL_ASSERT(0 && "Incomplete closure support for setmessage");
    }

    BatchedMessageList messages {
        bsg->uniform.context->batched_messages_buffer()
    };

    auto* m = messages.find(name);
    if (m != nullptr) {
        OSL_DASSERT(m->name == name);
        // message already exists?
        Wide<const ustring> msg_wsourcefile(m->wsourcefile);
        Wide<const int> msg_wsourceline(m->wsourceline);
        OSL_ASSERT(m->has_data());
        {
            auto lanes_with_data = m->valid_mask & matching_lanes;

            lanes_with_data.foreach ([=](ActiveLane lane) -> void {
                ustring msg_sourcefile = msg_wsourcefile[lane];
                int msg_sourceline     = msg_wsourceline[lane];

                bsg->uniform.context->batched<__OSL_WIDTH>().errorfmt(
                    lanes_with_data,
                    "message \"{}\" already exists (created here: {}:{})"
                    " cannot set again from {}:{}",
                    name, msg_sourcefile, msg_sourceline, sourcefile,
                    sourceline);
            });
            auto lanes_to_populate = (~m->valid_mask & matching_lanes)
                                     & ~m->get_before_set_mask;
            if (lanes_to_populate.any_on()) {
                m->import_data(wsrcval, lanes_to_populate, layeridx, sourcefile,
                               sourceline);
            }
        }
        Mask lanes_that_getmessage_called_on = m->get_before_set_mask
                                               & matching_lanes;
        lanes_that_getmessage_called_on.foreach ([=](ActiveLane lane) -> void {
            ustring msg_sourcefile = msg_wsourcefile[lane];
            int msg_sourceline     = msg_wsourceline[lane];
            bsg->uniform.context->batched<__OSL_WIDTH>().errorfmt(
                Mask(Lane(lane)),
                "message \"{}\" was queried before being set (queried here: {}:{})"
                " setting it now ({}:{}) would lead to inconsistent results",
                name, msg_sourcefile, msg_sourceline, sourcefile, sourceline);
        });
    } else {
        // The message didn't exist - create it
        messages.add(name, wsrcval, matching_lanes, layeridx, sourcefile,
                     sourceline);
    }
}

}  // namespace


OSL_BATCHOP void
__OSL_MASKED_OP2(setmessage, s, WX)(BatchedShaderGlobals* bsg_,
                                    ustring_pod name_, long long type,
                                    void* wvalue, int layeridx,
                                    ustring_pod sourcefile_, int sourceline,
                                    unsigned int mask_value)
{
    Mask mask(mask_value);
    OSL_ASSERT(mask.any_on());

    ustringrep name = USTR(name_);
    MaskedData wsrcval(TYPEDESC(type), false /*has_derivs*/, mask, wvalue);

    ustringrep sourcefile = USTR(sourcefile_);

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);


    OSL_DASSERT(wsrcval.type().basetype
                != TypeDesc::UNKNOWN);  // secret code for closure
#if 0                                   // TBD closure support
    // recreate TypeDesc -- we just crammed it into an int!
    TypeDesc type = TYPEDESC(type_);
    bool is_closure = (type.basetype == TypeDesc::UNKNOWN); // secret code for closure
    if (is_closure) {
        OSL_ASSERT(0 && "Incomplete add closure support to setmessage");
        type.basetype = TypeDesc::PTR;

    }
    // for closures, we store a pointer
#endif

    impl_setmessage(bsg, sourcefile, sourceline, wsrcval, layeridx, name, mask);
}


OSL_BATCHOP void
__OSL_MASKED_OP2(setmessage, Ws, WX)(BatchedShaderGlobals* bsg_, void* wname,
                                     long long type, void* wvalue, int layeridx,
                                     ustring_pod sourcefile_, int sourceline,
                                     unsigned int mask_value)
{
    Mask mask(mask_value);
    OSL_ASSERT(mask.any_on());

    Wide<const ustringrep> wName(wname);
    MaskedData wsrcval(TYPEDESC(type), false /*has_derivs*/, mask, wvalue);

    ustringrep sourcefile = USTR(sourcefile_);

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);

#if 0
    // recreate TypeDesc -- we just crammed it into an int!
    TypeDesc type = TYPEDESC(type_);
    bool is_closure = (type.basetype == TypeDesc::UNKNOWN); // secret code for closure
    if (is_closure) {
        OSL_ASSERT(0 && "Incomplete add closure support to setmessage");
        type.basetype = TypeDesc::PTR;

    }
    // for closures, we store a pointer
#else
    OSL_DASSERT(wsrcval.type().basetype
                != TypeDesc::UNKNOWN);  // secret code for closure
#endif

    foreach_unique(wName, mask, [=](ustring name, Mask matching_lanes) -> void {
        impl_setmessage(bsg, sourcefile, sourceline, wsrcval, layeridx, name,
                        matching_lanes);
    });
}



OSL_BATCHOP void
__OSL_MASKED_OP(getmessage)(void* bsg_, void* result, ustring_pod source_,
                            ustring_pod name_, long long type_, void* val,
                            int derivs, int layeridx, ustring_pod sourcefile_,
                            int sourceline, unsigned int mask_value)
{
    ustringrep source     = USTR(source_);
    ustringrep name       = USTR(name_);
    ustringrep sourcefile = USTR(sourcefile_);

    Mask mask(mask_value);

    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    Masked<int> wR(result, mask);

    // recreate TypeDesc -- we just crammed it into an int!
    TypeDesc type   = TYPEDESC(type_);
    bool is_closure = (type.basetype
                       == TypeDesc::UNKNOWN);  // secret code for closure
    if (is_closure) {
        OSL_ASSERT(0 && "Incomplete add closure support to getmessage");
        type.basetype = TypeDesc::PTR;  // for closures, we store a pointer
    }

    static ustring ktrace("trace");
    OSL_ASSERT(val != nullptr);
    MaskedData valRef(type, derivs, mask, val);
    if (USTR(source_) == ktrace) {
        // Source types where we need to ask the renderer
        return bsg->uniform.renderer->batched(WidthTag())
            ->getmessage(bsg, wR, source, name, valRef);
    }


    BatchedMessageList messages {
        bsg->uniform.context->batched_messages_buffer()
    };


    MessageBlock* m = messages.find(name);
    if (m != nullptr) {
        Wide<const ustring> msg_wsourcefile(m->wsourcefile);
        Wide<const int> msg_wsourceline(m->wsourceline);
        Wide<const int> msg_wlayeridx(m->wlayeridx);

        if (m->type != type) {
            // found message, but types don't match
            mask.foreach ([=](ActiveLane lane) -> void {
                //int msg_layerid = msg_wlayeridx[lane];
                ustring msg_sourcefile = msg_wsourcefile[lane];
                int msg_sourceline     = msg_wsourceline[lane];
                // found message, but was set by a layer deeper than the one querying the message
                bool has_data = m->has_data() ? m->valid_mask[lane] : false;

                bsg->uniform.context->batched<__OSL_WIDTH>().errorfmt(
                    Mask(lane),
                    "type mismatch for message \"{}\" ({} as {} here: {}:{})"
                    " cannot fetch as {} from {}:{}",
                    name.c_str(), has_data ? "created" : "queried",
                    m->type == TypeDesc::PTR ? "closure color"
                                             : m->type.c_str(),
                    msg_sourcefile, msg_sourceline,
                    is_closure ? "closure color" : type.c_str(), sourcefile,
                    sourceline);
            });

            assign_all(wR, 0);
            return;
        }
        OSL_ASSERT(m->has_data());
        if (!m->has_data()) {
            // getmessage ran before and found nothing - just return 0
            assign_all(wR, 0);
            return;
        }
        auto found_lanes   = mask & m->valid_mask;
        auto missing_lanes = mask & ~m->valid_mask & ~m->get_before_set_mask;

        // Use int instead of Mask<> to allow reduction clause in openmp simd declaration
        int lanes_set_by_deeper_layer_bits { 0 };
        OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH)
                           reduction(|
                                     : lanes_set_by_deeper_layer_bits))
        for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
            int msg_layerid = msg_wlayeridx[lane];
            // NOTE: using bitwise & to avoid branches
            if (found_lanes[lane] & (msg_layerid > layeridx)) {
                // inline of Mask::set_on(lane)
                lanes_set_by_deeper_layer_bits |= 1 << lane;
            }
        }
        Mask lanes_set_by_deeper_layer(lanes_set_by_deeper_layer_bits);

        lanes_set_by_deeper_layer.foreach ([=](ActiveLane lane) -> void {
            int msg_layerid        = msg_wlayeridx[lane];
            ustring msg_sourcefile = msg_wsourcefile[lane];
            int msg_sourceline     = msg_wsourceline[lane];

            // found message, but was set by a layer deeper than the one querying the message
            bsg->uniform.context->batched<__OSL_WIDTH>().errorfmt(
                Mask(lane),
                "message \"{}\" was set by layer #{} ({}:{})"
                " but is being queried by layer #{} ({}:{})"
                " - messages may only be transferred from nodes "
                "that appear earlier in the shading network",
                name, msg_layerid, msg_sourcefile, msg_sourceline, layeridx,
                sourcefile, sourceline);
            wR[lane] = 0;
        });

        auto lanes_to_copy = found_lanes & ~lanes_set_by_deeper_layer;
        if (lanes_to_copy.any_on()) {
            m->export_data(valRef & lanes_to_copy);
            assign_all(wR & lanes_to_copy, 1);
        }

        if (missing_lanes.any_on()) {
            assign_all(wR & missing_lanes, 0);
            if (bsg->uniform.context->shadingsys().strict_messages()) {
                m->get_before_set_mask |= missing_lanes;
                Masked<int> wlayeridx(m->wlayeridx, missing_lanes);
                Masked<ustring> wsourcefile(m->wsourcefile, missing_lanes);
                Masked<int> wsourceline(m->wsourceline, missing_lanes);
                OSL_OMP_PRAGMA(omp simd simdlen(__OSL_WIDTH))
                for (int lane = 0; lane < __OSL_WIDTH; ++lane) {
                    wlayeridx[lane]   = layeridx;
                    wsourcefile[lane] = sourcefile;
                    wsourceline[lane] = sourceline;
                }
            }
        }
        return;
    }

    if (bsg->uniform.context->shadingsys().strict_messages()) {
        // The message didn't exist - create it
        MaskedData wsrcval(type, false /*has_derivs*/, mask, nullptr);
        messages.add(name, wsrcval, mask, layeridx, sourcefile, sourceline);
    }
    assign_all(wR, 0);
    return;
}



// Trace

// Utility: retrieve a pointer to the ShadingContext's trace options
// struct, also re-initialize its contents.

OSL_BATCHOP void
__OSL_MASKED_OP(trace)(void* bsg_, void* result, void* opt_, void* Pos_,
                       void* dPosdx_, void* dPosdy_, void* Dir_, void* dDirdx_,
                       void* dDirdy_, unsigned int mask_value)
{
    auto* bsg = reinterpret_cast<BatchedShaderGlobals*>(bsg_);
    auto* opt = reinterpret_cast<BatchedRendererServices::TraceOpt*>(opt_);

    Masked<int> wR(result, Mask(mask_value));

    Block<Vec3> zero_block;
    assign_all(zero_block, Vec3(0.0f));

    Wide<const Vec3> wP(Pos_);
    Wide<const Vec3> wPdx(dPosdx_ ? dPosdx_ : &zero_block);
    Wide<const Vec3> wPdy(dPosdy_ ? dPosdy_ : &zero_block);

    Wide<const Vec3> wD(Dir_);
    Wide<const Vec3> wDdx(dDirdx_ ? dDirdx_ : &zero_block);
    Wide<const Vec3> wDdy(dDirdy_ ? dDirdy_ : &zero_block);

    bsg->uniform.renderer->batched(WidthTag())
        ->trace(*opt, bsg, wR, wP, wPdx, wPdy, wD, wDdx, wDdy);
}

}  // namespace __OSL_WIDE_PVT
OSL_NAMESPACE_EXIT
