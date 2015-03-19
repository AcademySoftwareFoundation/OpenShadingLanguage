#!/usr/bin/env python

command = testshade("-g 1 1 --layer upstream upstream --layer downstream test " +
                    "--connect upstream out downstream a " +
                    "--connect upstream struct1 downstream mystruct1")
