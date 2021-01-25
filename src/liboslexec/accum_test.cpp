// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/accum.h>
#include <OSL/oslclosure.h>
#include <OpenImageIO/unittest.h>

using namespace OSL;

#define END_AOV 65535

typedef struct
{
    const char *path[16];
    int expected[8];
} TestPath;


// This is a fake AOV implementation. It will just keep track
// of what test cases wrote to it
class MyAov final : public Aov
{
    public:
        MyAov(const TestPath *test, int id)
        {
            // Init for the test case array. For each test case set a bool
            // in m_expected marking wether this AOV should get some color
            // from that test or not.
            for (int i = 0; test[i].path[0]; ++i) { // iterate test cases
                m_expected.push_back(false); // false by default
                for (const int *expected = test[i].expected; *expected != END_AOV; ++expected)
                    if (*expected == id)
                        // if the AOV id is on the test's expected list, set it to true
                        m_expected[i] = true;
            }
            m_received.resize(m_expected.size());
        }
        virtual ~MyAov() {}

        virtual void write(void *flush_data, Color3 &color, float /*alpha*/,
                                 bool has_color, bool /*has_alpha*/)
        {
            // our custom argument to write is the rule's number so we
            // can check what. But you normally would pass information
            // about the pixel.
            long int testno = (long int)flush_data;
            if (has_color && color.x > 0)
                // only mrk with true if there is a positive color present
                m_received[testno] = true;
            else
                m_received[testno] = false;
        }

        bool check()
        {
            return m_expected == m_received;
        }

    protected:

        std::vector<bool> m_expected;
        std::vector<bool> m_received;
};

// Simulate the tracing of a path with the accumulator
void simulate(Accumulator &accum, const char **events, int testno)
{
    accum.begin();
    accum.pushState();
    // for each ray stop in the path (see test cases) ...
    while (*events) {
        const char *e = *events;
        // for each label in this hit
        while (*e) {
            ustring sym(e, 1);
            // advance our state with the label
            accum.move(sym);
            e++;
        }
        // always finish the hit with a stop label
        accum.move(Labels::STOP);
        events++;
    }
    // Here is were we have reached a light, accumulate color
    accum.accum(Color3(1, 1, 1));
    // Restore state and flush
    accum.popState();
    accum.end((void *)(long int)testno);
}

int main()
{
    // Some constants to avoid refering to AOV's by number
    const int beauty       = 0;
    const int diffuse2_3   = 1;
    const int light3       = 2;
    const int object_1     = 3;
    const int specular     = 4;
    const int diffuse      = 5;
    const int transpshadow = 6;
    const int reflections  = 7;
    const int nocaustic    = 8;
    const int custom       = 9;
    const int naovs        = 10;

    // The actual test cases. Each one is a list of ray hits with some labels.
    // We use 1 char labels for convenience. They will be converted to ustrings
    // by simulate. And then a list of expected AOV's for each.
    TestPath test[] = { { { "C_", "TS",  "TS", "RD", "L_", NULL },             { beauty, specular, nocaustic, END_AOV} },
                              { { "C_", "TS",  "TS", "RD", "RG", "L_", NULL }, { END_AOV } },
                              { { "C_", "TS",  "TS", "RD", "RD", "L_", NULL },     { beauty, specular, diffuse2_3, nocaustic, END_AOV } },
                              { { "C_", "RG",  "RD", "RG", "RD", "RG", "RD", "L_", NULL },    { END_AOV } },
                              { { "C_", "RG",  "RD", "RG", "RD", "L_", NULL }, { nocaustic, END_AOV } },
                              { { "C_", "RD",  "RD", "L_", NULL },             { beauty, diffuse, diffuse2_3, nocaustic, END_AOV } },
                              { { "C_", "RD",  "RS", "RD", "L_", NULL },       { nocaustic, END_AOV } },
                              { { "C_", "RD",  "Ts", "L_", NULL },             { transpshadow, END_AOV } },
                              { { "C_", "TS",  "TS", "RD", "L_3", NULL },      { beauty, specular, light3, nocaustic, END_AOV } },
                              { { "C_", "RD1", "RD", "L_", NULL },             { beauty, diffuse, diffuse2_3, object_1, nocaustic, END_AOV } },
                              { { "C_", "RS",  "RD", "RG", "L_", NULL },       { END_AOV } },
                              { { "C_", "RS",  "RD", "L_", NULL },             { beauty, specular, reflections, nocaustic, END_AOV } },
                              { { "C_", "RD",  "RY", "RD", "U_", NULL },       { custom, END_AOV } },
                              { { NULL }, { END_AOV } } };

    // Create our fake testing AOV's
    std::vector<MyAov> aovs;
    for (int i = 0; i < naovs; ++i)
        aovs.emplace_back(test, i);

    // Create the automata and add the rules
    AccumAutomata automata;

    automata.addEventType(ustring("U"));
    automata.addScatteringType(ustring("Y"));

    OIIO_CHECK_ASSERT(automata.addRule("C[SG]*D*L",        beauty));
    OIIO_CHECK_ASSERT(automata.addRule("C[SG]*D{2,3}L",    diffuse2_3));
    OIIO_CHECK_ASSERT(automata.addRule("C[SG]*D*<L.'3'>",  light3));
    OIIO_CHECK_ASSERT(automata.addRule("C[SG]*<.D'1'>D*L", object_1));
    OIIO_CHECK_ASSERT(automata.addRule("C<.[SG]>+D*L",     specular));
    OIIO_CHECK_ASSERT(automata.addRule("CD+L",             diffuse));
    OIIO_CHECK_ASSERT(automata.addRule("CD+<Ts>L",         transpshadow));
    OIIO_CHECK_ASSERT(automata.addRule("C<R[^D]>+D*L",     reflections));
    OIIO_CHECK_ASSERT(automata.addRule("C([SG]*D){1,2}L",  nocaustic));
    OIIO_CHECK_ASSERT(automata.addRule("CDY+U",            custom));

    automata.compile();

    // now create the accumulator
    Accumulator accum (&automata);

    // and set the AOV's for each id (beauty, diffuse2_3, etc ...)
    for (int i = 0; i < naovs; ++i)
        accum.setAov(i, &aovs[i], false, false);

    // do the simulation for each test case
    for (int i = 0; test[i].path[0]; ++i)
        simulate(accum, test[i].path, i);

    // And check. We unroll this loop for boost to give us a useful
    // error in case they fail
    OIIO_CHECK_ASSERT(aovs[beauty      ].check());
    OIIO_CHECK_ASSERT(aovs[diffuse2_3  ].check());
    OIIO_CHECK_ASSERT(aovs[light3      ].check());
    OIIO_CHECK_ASSERT(aovs[object_1    ].check());
    OIIO_CHECK_ASSERT(aovs[specular    ].check());
    OIIO_CHECK_ASSERT(aovs[diffuse     ].check());
    OIIO_CHECK_ASSERT(aovs[transpshadow].check());
    OIIO_CHECK_ASSERT(aovs[reflections ].check());
    OIIO_CHECK_ASSERT(aovs[nocaustic   ].check());

    std::cout << "Light expressions check OK" << std::endl;
    return unit_test_failures;
}
