/*****************************************************************************
 *
 *             Copyright (c) 2009 Sony Pictures Imageworks, Inc.
 *                            All rights reserved.
 *
 *  This material contains the confidential and proprietary information
 *  of Sony Pictures Imageworks, Inc. and may not be disclosed, copied or
 *  duplicated in any form, electronic or hardcopy, in whole or in part,
 *  without the express prior written consent of Sony Pictures Imageworks,
 *  Inc. This copyright notice does not imply publication.
 *
 *****************************************************************************/


#include <vector>
#include <string>
#include <cstdio>

#include <boost/algorithm/string.hpp>

#include "OpenImageIO/strutil.h"
#include "OpenImageIO/dassert.h"
#include "OpenImageIO/thread.h"
#include "OpenImageIO/filesystem.h"

#include "oslexec_pvt.h"
using namespace OSL;
using namespace OSL::pvt;



namespace OSL {


ShadingSystem *
ShadingSystem::create ()
{
    // Doesn't need a shared cache
    ShadingSystemImpl *ts = new ShadingSystemImpl;
#ifdef DEBUG
    std::cerr << "creating new ShadingSystem " << (void *)ts << "\n";
#endif
    return ts;
}



void
ShadingSystem::destroy (ShadingSystem *x)
{
    delete (ShadingSystemImpl *) x;
}



ShadingSystem::ShadingSystem ()
{
}



ShadingSystem::~ShadingSystem ()
{
}




namespace pvt {   // OSL::pvt



ShadingSystemImpl::ShadingSystemImpl ()
    : m_statslevel(0)
{
}



ShadingSystemImpl::~ShadingSystemImpl ()
{
    printstats ();
}



bool
ShadingSystemImpl::attribute (const std::string &name, TypeDesc type,
                              const void *val)
{
    lock_guard guard (m_mutex);  // Thread safety
    if (name == "searchpath:shader" && type == TypeDesc::STRING) {
        m_searchpath = ustring (*(const char **)val);
        Filesystem::searchpath_split (m_searchpath, m_searchpath_dirs);
        return true;
    }
    if (name == "statistics:level" && type == TypeDesc::INT) {
        m_statslevel = *(const int *)val;
        return true;
    }
    return false;
}



bool
ShadingSystemImpl::getattribute (const std::string &name, TypeDesc type,
                                 void *val)
{
    lock_guard guard (m_mutex);  // Thread safety
    if (name == "searchpath:shader" && type == TypeDesc::INT) {
        *(const char **)val = m_searchpath.c_str();
        return true;
    }
    if (name == "statistics:level" && type == TypeDesc::INT) {
        *(int *)val = m_statslevel;
        return true;
    }
    return false;
}



std::string
ShadingSystemImpl::geterror () const
{
    lock_guard lock (m_errmutex);
    std::string e = m_errormessage;
    m_errormessage.clear();
    return e;
}



void
ShadingSystemImpl::error (const char *message, ...)
{
    lock_guard lock (m_errmutex);
    va_list ap;
    va_start (ap, message);
    if (m_errormessage.size())
        m_errormessage += '\n';
    m_errormessage += Strutil::vformat (message, ap);
    va_end (ap);
}



std::string
ShadingSystemImpl::getstats (int level) const
{
    if (level <= 0)
        return "";
    fast_mutex::lock_guard statguard (m_stats_mutex);
    std::ostringstream out;
    out << "OSL ShadingSystem statistics (" << (void*)this << ")\n";
    out << "  Shaders:\n";
    out << "    Requested: " << m_stat_shaders_requested << "\n";
    out << "    Loaded:    " << m_stat_shaders_loaded << "\n";
    out << "    Masters:   " << m_stat_shaders_loaded << "\n";
    out << "    Instances: " << m_stat_instances.requested() << " requested, "
        << m_stat_instances.peak() << " peak, "
        << m_stat_instances.current() << " current\n";
    // FIXME
    return out.str();
}



void
ShadingSystemImpl::printstats () const
{
    if (m_statslevel == 0)
        return;
    std::cout << getstats (m_statslevel) << "\n\n";
}



}; // namespace pvt
}; // namespace OSL
