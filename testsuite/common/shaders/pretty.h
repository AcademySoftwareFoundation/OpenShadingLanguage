// pretty(x) rounds very small values to zero and turns -0 values into +0.
// This is useful for testsuite to eliminate some pesky LSB errors that
// cause reference output to differ between platforms.

float pretty (float x)
{
    float xx = x;
    if (abs(xx) < 5.0e-6)
        xx = 0;
    return xx;
}


color pretty (color x)
{
    return color (pretty(x[0]), pretty(x[1]), pretty(x[2]));
}

point pretty (point x)
{
    return point (pretty(x[0]), pretty(x[1]), pretty(x[2]));
}

vector pretty (vector x)
{
    return vector (pretty(x[0]), pretty(x[1]), pretty(x[2]));
}

normal pretty (normal x)
{
    return normal (pretty(x[0]), pretty(x[1]), pretty(x[2]));
}


matrix pretty (matrix x)
{
    return matrix (pretty(x[0][0]), pretty(x[0][1]), pretty(x[0][2]), pretty(x[0][3]), 
                   pretty(x[1][0]), pretty(x[1][1]), pretty(x[1][2]), pretty(x[1][3]), 
                   pretty(x[2][0]), pretty(x[2][1]), pretty(x[2][2]), pretty(x[2][3]), 
                   pretty(x[3][0]), pretty(x[3][1]), pretty(x[3][2]), pretty(x[3][3]));
}
