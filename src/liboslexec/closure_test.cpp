/*
Copyright (c) 2009 Sony Pictures Imageworks, et al.
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
#include <iostream>

#include <OpenImageIO/dassert.h>

#include "oslconfig.h"
#include "oslclosure.h"
using namespace OSL;
//using namespace pvt;


#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>


class MyClosure : public ClosurePrimitive {
public:
    MyClosure () : ClosurePrimitive (ustring("my"), 0, ustring("f")) { }
};

MyClosure myclosure;



BOOST_AUTO_TEST_CASE (closure_test_add)
{
    ClosureColor::register_primitive (myclosure);
    
    // Create a closure with one component
    ClosureColor c;
    ClosureColor::compref_t comp = ClosureColor::primitive (ustring("my"));
    comp->addarg (0.33f);
    c.add (comp, Color3(.1,.1,.1));
    BOOST_CHECK_EQUAL (c.ncomponents(), 1);

    // Add another component same params as the first, should still have
    // one component, just higher weight.
    c.add (comp, Color3(.1,.1,.1));
    BOOST_CHECK_EQUAL (c.ncomponents(), 1);
    BOOST_CHECK_EQUAL (c.weight(0), Color3 (0.2, 0.2, 0.2));
    std::cout << "c = " << c << "\n";

    // Add another component with different params.  It should now look
    // like two components, not combine with the others.
    comp = ClosureColor::primitive (ustring("my"));
    comp->addarg (0.5);
    c.add (comp, Color3(0.4, 0.4, 0.4));
    BOOST_CHECK_EQUAL (c.ncomponents(), 2);
    BOOST_CHECK_EQUAL (c.weight(1), Color3 (0.4, 0.4, 0.4));
    std::cout << "c = " << c << "\n";
}


