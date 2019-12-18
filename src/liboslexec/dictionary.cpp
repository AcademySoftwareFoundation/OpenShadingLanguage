/*
Copyright (c) 2009-2011 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <unordered_map>

#include <OpenImageIO/strutil.h>

#include <pugixml.hpp>

#ifdef USING_OIIO_PUGI
namespace pugi = OIIO::pugi;
#endif


#include "oslexec_pvt.h"

OSL_NAMESPACE_ENTER

namespace pvt {   // OSL::pvt


// Helper class to manage the dictionaries.
//
// Shaders are written as if they parse arbitrary things from whole
// cloth on every call: from potentially loading XML from disk, parsing
// it, doing queries, and converting the string data to other types.
//
// But that is expensive, so we really cache all this stuff at several
// levels.
//
// We have parsed xml (as pugi::xml_document *'s) cached in a hash table,
// looked up by the xml and/or dictionary name.  Either will do, if it
// looks like a filename, it will read the XML from the file, otherwise it
// will interpret it as xml directly.
//
// Also, individual queries are cached in a hash table.  The key is a
// tuple of (nodeID, query_string, type_requested), so that asking for a
// particular query to return a string is a totally different cache
// entry than asking for it to be converted to a matrix, say.
//
class Dictionary {
public:
    Dictionary (ShadingContext *ctx) : m_context(ctx)
    {
        // Create placeholder element 0 == 'not found'
        m_nodes.emplace_back(0, pugi::xml_node());
    }
    ~Dictionary () {
        // Free all the documents.
        for (auto& doc : m_documents)
            delete doc;
    }

    int dict_find (ustring dictionaryname, ustring query);
    int dict_find (int nodeID, ustring query);
    int dict_next (int nodeID);
    int dict_value (int nodeID, ustring attribname, TypeDesc type, void *data);

private:
    // We cache individual queries with a key that is a tuple of the
    // (nodeID, query_string, type_requested).
    struct Query {
        int document;   // which dictionary document
        int node;       // root node for the search
        ustring name;   // name for the the search
        TypeDesc type;  // UNKNOWN signifies a node, versus an attribute value
        Query (int doc_, int node_, ustring name_,
               TypeDesc type_=TypeDesc::UNKNOWN) :
            document(doc_), node(node_), name(name_), type(type_) { }
        bool operator== (const Query &q) const {
            return document == q.document && node == q.node &&
                   name == q.name && type == q.type;
        }
    };

    // Must define a hash operation to build the unordered_map.
    struct QueryHash {
        size_t operator() (const Query &key) const {
            return key.name.hash() + 17*key.node + 79*key.document;
        }
    };

    // The cached query result is mostly just a 'valueoffset', which is
    // the index into floatdata/intdata/stringdata (depending on the type
    // being asked for) at which the decoded data live, or a node ID
    // if the query was for a node rather than for an attribute.
    struct QueryResult {
        int valueoffset;  // Offset into one of the 'data' vectors, or nodeID
        bool is_valid;    // true: query found
        QueryResult (bool valid=true) : valueoffset(0), is_valid(valid) { }
        QueryResult (bool isnode, int value)
            : valueoffset(value), is_valid(true) { }
    };

    // Nodes we've looked up.  Includes a 'next' index of the matching node
    // for the query that generated this one.
    struct Node {
        int document;         // which document the node belongs to
        pugi::xml_node node;  // which node within the dictionary
        int next;             // next node for the same query
        Node (int d, const pugi::xml_node &n)
            : document(d), node(n), next(0) { }
    };

    typedef std::unordered_map <Query, QueryResult, QueryHash> QueryMap;
    typedef std::unordered_map<ustring, int, ustringHash> DocMap;

    ShadingContext *m_context;  // back-pointer to shading context

    // List of XML documents we've read in.
    std::vector<pugi::xml_document *> m_documents;

    // Map xml strings and/or filename to indices in m_documents.
    DocMap m_document_map;

    // Cache of fully resolved queries.
    Dictionary::QueryMap m_cache;  // query cache

    // List of all the nodes we've found by queries.
    std::vector<Dictionary::Node> m_nodes;

    // m_floatdata, m_intdata, and m_stringdata hold the decoded data
    // results (including type conversion) of cached queries.
    std::vector<float>   m_floatdata;
    std::vector<int>     m_intdata;
    std::vector<ustring> m_stringdata;

    // Helper function: return the document index given dictionary name.
    int get_document_index (ustring dictionaryname);
};



int
Dictionary::get_document_index (ustring dictionaryname)
{
    DocMap::iterator dm = m_document_map.find(dictionaryname);
    int dindex;
    if (dm == m_document_map.end()) {
        dindex = m_documents.size();
        m_document_map[dictionaryname] = dindex;
        pugi::xml_document *doc = new pugi::xml_document;
        m_documents.push_back (doc);
        pugi::xml_parse_result parse_result;
        if (Strutil::ends_with (dictionaryname.string(), ".xml")) {
            // xml file -- read it
            parse_result = doc->load_file (dictionaryname.c_str());
        } else {
            // load xml directly from the string
            parse_result = doc->load_buffer (dictionaryname.c_str(),
                                             dictionaryname.length());
        }
        if (! parse_result) {
            m_context->errorf("XML parsed with errors: %s, at offset %d",
                              parse_result.description(),
                              parse_result.offset);
            m_document_map[dictionaryname] = -1;
            return -1;
        }
    } else {
        dindex = dm->second;
    }

    OSL_DASSERT(dindex < (int)m_documents.size());
    return dindex;
}



int
Dictionary::dict_find (ustring dictionaryname, ustring query)
{
    int dindex = get_document_index (dictionaryname);
    if (dindex < 0)
        return dindex;

    Query q (dindex, 0, query);
    QueryMap::iterator qfound = m_cache.find (q);
    if (qfound != m_cache.end()) {
        return qfound->second.valueoffset;
    }

    pugi::xml_document *doc = m_documents[dindex];

    // Query was not found.  Do the expensive lookup and cache it
    pugi::xpath_node_set matches;

    try {
        matches = doc->select_nodes (query.c_str());
    }
    catch (const pugi::xpath_exception& e) {
        m_context->errorf("Invalid dict_find query '%s': %s",
                          query.c_str(), e.what());
        return 0;
    }

    if (matches.empty()) {
        m_cache[q] = QueryResult (false);  // mark invalid
        return 0;   // Not found
    }
    int firstmatch = (int) m_nodes.size();
    int last = -1;
    for (auto&& m : matches) {
        m_nodes.emplace_back(dindex, m.node());
        int nodeid = (int) m_nodes.size()-1;
        if (last < 0) {
            // If this is the first match, add a cache entry for it
            m_cache[q] = QueryResult (true /* it's a node */, nodeid);
        } else {
            // If this is a subsequent match, set the last match's 'next'
            m_nodes[last].next = nodeid;
        }
        last = nodeid;
    }
    return firstmatch;
}



int
Dictionary::dict_find (int nodeID, ustring query)
{
    if (nodeID <= 0 || nodeID >= (int)m_nodes.size())
        return 0;     // invalid node ID

    int document = m_nodes[nodeID].document;
    Query q (document, nodeID, query);
    QueryMap::iterator qfound = m_cache.find (q);
    if (qfound != m_cache.end()) {
        return qfound->second.valueoffset;
    }

    // Query was not found.  Do the expensive lookup and cache it
    pugi::xpath_node_set matches;
    try {
        matches = m_nodes[nodeID].node.select_nodes (query.c_str());
    }
    catch (const pugi::xpath_exception& e) {
        m_context->errorf("Invalid dict_find query '%s': %s",
                          query.c_str(), e.what());
        return 0;
    }

    if (matches.empty()) {
        m_cache[q] = QueryResult (false);  // mark invalid
        return 0;   // Not found
    }
    int firstmatch = (int) m_nodes.size();
    int last = -1;
    for (auto&& m : matches) {
        m_nodes.emplace_back(document, m.node());
        int nodeid = (int) m_nodes.size()-1;
        if (last < 0) {
            // If this is the first match, add a cache entry for it
            m_cache[q] = QueryResult (true /* it's a node */, nodeid);
        } else {
            // If this is a subsequent match, set the last match's 'next'
            m_nodes[last].next = nodeid;
        }
        last = nodeid;
    }
    return firstmatch;
}



int
Dictionary::dict_next (int nodeID)
{
    if (nodeID <= 0 || nodeID >= (int)m_nodes.size())
        return 0;     // invalid node ID
    return m_nodes[nodeID].next;
}



int
Dictionary::dict_value (int nodeID, ustring attribname,
                        TypeDesc type, void *data)
{
    if (nodeID <= 0 || nodeID >= (int)m_nodes.size())
        return 0;     // invalid node ID

    const Dictionary::Node &node (m_nodes[nodeID]);
    Dictionary::Query q (node.document, nodeID, attribname, type);
    Dictionary::QueryMap::iterator qfound = m_cache.find (q);
    if (qfound != m_cache.end()) {
        // previously found
        int offset = qfound->second.valueoffset;
        int n = type.numelements() * type.aggregate;
        if (type.basetype == TypeDesc::STRING) {
            OSL_DASSERT (n == 1 && "no string arrays in XML");
            ((ustring *)data)[0] = m_stringdata[offset];
            return 1;
        }
        if (type.basetype == TypeDesc::INT) {
            for (int i = 0;  i < n;  ++i)
                ((int *)data)[i] = m_intdata[offset++];
            return 1;
        }
        if (type.basetype == TypeDesc::FLOAT) {
            for (int i = 0;  i < n;  ++i)
                ((float *)data)[i] = m_floatdata[offset++];
            return 1;
        }
        return 0;  // Unknown type
    }

    // OK, the entry wasn't in the cache, we need to decode it and cache it.

    const char *val = NULL;
    if (attribname.empty()) {
        val = node.node.value();
    } else {
        for (pugi::xml_attribute_iterator ait = node.node.attributes_begin();
             ait != node.node.attributes_end(); ++ait) {
            if (ait->name() == attribname) {
                val = ait->value();
                break;
            }
        }
    }
    if (val == NULL)
        return 0;   // not found

    Dictionary::QueryResult r (false, 0);
    int n = type.numelements() * type.aggregate;
    if (type.basetype == TypeDesc::STRING && n == 1) {
        r.valueoffset = (int) m_stringdata.size();
        ustring s (val);
        m_stringdata.push_back (s);
        ((ustring *)data)[0] = s;
        m_cache[q] = r;
        return 1;
    }
    if (type.basetype == TypeDesc::INT) {
        r.valueoffset = (int) m_intdata.size();
        string_view valstr (val);
        for (int i = 0;  i < n;  ++i) {
            int v;
            OIIO::Strutil::parse_int (valstr, v);
            OIIO::Strutil::parse_char (valstr, ',');
            m_intdata.push_back (v);
            ((int *)data)[i] = v;
        }
        m_cache[q] = r;
        return 1;
    }
    if (type.basetype == TypeDesc::FLOAT) {
        r.valueoffset = (int) m_floatdata.size();
        string_view valstr (val);
        for (int i = 0;  i < n;  ++i) {
            float v;
            OIIO::Strutil::parse_float (valstr, v);
            OIIO::Strutil::parse_char (valstr, ',');
            m_floatdata.push_back (v);
            ((float *)data)[i] = v;
        }
        m_cache[q] = r;
        return 1;
    }

    // Anything that's left is an unsupported type
    return 0;
}


}; // namespace pvt



int
ShadingContext::dict_find (ustring dictionaryname, ustring query)
{
    if (! m_dictionary) {
        m_dictionary = new Dictionary (this);
    }
    return m_dictionary->dict_find (dictionaryname, query);
}



int
ShadingContext::dict_find (int nodeID, ustring query)
{
    if (! m_dictionary) {
        m_dictionary = new Dictionary (this);
    }
    return m_dictionary->dict_find (nodeID, query);
}



int
ShadingContext::dict_next (int nodeID)
{
    if (! m_dictionary)
        return 0;
    return m_dictionary->dict_next (nodeID);
}



int
ShadingContext::dict_value (int nodeID, ustring attribname,
                            TypeDesc type, void *data)
{
    if (! m_dictionary)
        return 0;
    return m_dictionary->dict_value (nodeID, attribname, type, data);
}



void
ShadingContext::free_dict_resources ()
{
    delete m_dictionary;
}



OSL_SHADEOP int osl_dict_find_iis (void *sg_, int nodeID, void *query)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    return sg->context->dict_find (nodeID, USTR(query));
}



OSL_SHADEOP int osl_dict_find_iss (void *sg_, void *dictionary, void *query)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    return sg->context->dict_find (USTR(dictionary), USTR(query));
}



OSL_SHADEOP int osl_dict_next (void *sg_, int nodeID)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    return sg->context->dict_next (nodeID);
}



OSL_SHADEOP int osl_dict_value (void *sg_, int nodeID, void *attribname,
                               long long type, void *data)
{
    ShaderGlobals *sg = (ShaderGlobals *)sg_;
    return sg->context->dict_value (nodeID, USTR(attribname), TYPEDESC(type), data);
}



OSL_NAMESPACE_EXIT
