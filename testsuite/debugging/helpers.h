#if 0
color helper_2(color c)
{
    color r = cos(c);
    for(int i=0; i < 100; ++i) {
    	r = r + r;
    	r = r/2;
    }
    return r;
}
#endif

float helper_2(float c)
{
    float r = cos(c);
    //if (c < 0.4)
      //  return r;

    for(int i=0; i < 100; ++i) {
        r = r + r;
        r = r/2;
    }
    return r;
}

#define MACRO_HELPER(OUT, IN) \
        OUT = cos(IN); \
        if (IN >= 0.4) {    \
            for(int i=0; i < 100; ++i) { \
                OUT = OUT + OUT; \
                OUT = OUT/2; \
            } \
        } \

