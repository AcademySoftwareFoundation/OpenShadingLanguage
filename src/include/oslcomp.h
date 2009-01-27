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

#ifndef OSLCOMP_H
#define OSLCOMP_H


namespace OSL {




class OSLCompiler {
public:
    static OSLCompiler *create ();

    OSLCompiler (void) { }
    virtual ~OSLCompiler (void) { }

    /// Compile the given file, using the list of command-line options.
    /// Return true if ok, false if the compile failed.
    virtual bool compile (const std::string &filename,
                          const std::vector<std::string> &options) = 0;
};



}; // namespace OSL


#endif /* OSLCOMP_PVT_H */
