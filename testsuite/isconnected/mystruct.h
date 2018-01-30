struct MyStruct {
    float x;
    float y;
};

void status (float variable, string name)
{
    printf ("%s connected: %d  (value=%g)\n",
            name, isconnected(variable), variable);
}


void status (MyStruct variable, string name)
{
    printf ("%s connected: %d  (value={%g, %g})\n",
            name, isconnected(variable), variable.x, variable.y);
    status (variable.x, concat(name, ".x"));
    status (variable.y, concat(name, ".y"));
}
