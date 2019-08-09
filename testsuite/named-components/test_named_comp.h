// A test for named components. Use TYPE, X, Y, Z macros to customize for
// different types.


#ifndef TYPE
#  error "Must define TYPE, X, Y, Z"
#endif

// If X hasn't been defined, use X,Y,Z = x,y,z. So it only needs to be
// defined for colors.
#ifndef X
#  define X x
#  define Y y
#  define Z z
#endif

// Helper macro to turn symbols into strings.
#ifndef STRINGIZE
#  define STRINGIZE2(a) #a
#  define STRINGIZE(a) STRINGIZE2(a)
#endif



void
test_named_comp(TYPE dummy)
{
    printf (STRINGIZE(TYPE) ":\n");

    // Test retrieval of components by name
    TYPE R = TYPE (0, 1, 2);
    printf ("  R = " STRINGIZE(TYPE) "(%g) has components %g, %g, %g\n",
            R, R.X, R.Y, R.Z);

    // Test setting of components by name
    R.X = 0.25;
    printf ("  After R." STRINGIZE(X) " = 0.25, R = (%g)\n", R);
    R.Y = 0.5;
    printf ("  After R." STRINGIZE(Y) " = 0.50, R = (%g)\n", R);
    R.Z = 0.75;
    printf ("  After R." STRINGIZE(Z) " = 0.75, R = (%g)\n", R);

    // Make sure we can use components as parameters to functions
    printf ("  Via function params, R comps = %g %g %g\n",
            retrieve(R.X), retrieve(R.Y), retrieve(R.Z));
    set(R.X, 42.0);
    printf ("  After function params, R = %g\n", R);

    // Test array
    TYPE arr[2] = { TYPE(0), TYPE(0) };
    arr[1].X = 1.5;
    arr[1].Y = 2.0;
    arr[1].Z = 2.5;
    printf ("  Setting array components separately, arr[1] = %g %g %g\n",
            arr[1].X, arr[1].Y, arr[1].Z);
}


// Clear the macro definitions so we can include this header multiple times.
#undef TYPE
#undef X
#undef Y
#undef Z
