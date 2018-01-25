#!/usr/bin/env python


# This test verifies the execution order for both the default lazy
# execution, and the explicit entry points.
#
# The group looks like this:
#
#     layer A   (sets global Ci, no connections)
#
#
#     layer B   (connected downstream to E)
#   ____|
#  |
#  |  layer C   (no connections)
#  |
#  |
#  |  layer D   (no connections, but sets a renderer output)
#  |    |____>
#  |
#  +----+
#       V
#     layer E   (connected upstream to B, downstream to F and G)
#  +----|
#  |    V
#  |  layer F   (connected upstream to E)
#  |
#  +----v
#     layer G   (connected upstream to E, and it's the last layer)
#
#
# So by the default execution rules, the execution sequence will be:
#   A   (because it sets a global) [presuming lazyglobls == 0]
#   D   (because it sets a renderer output)
#   G   (because it's the last layer -- presumed group entry point)
#   E   (because F pulls its output)
#   B   (because E pulls its output)
#
# But if we give explicit entry points and ask to execute layers B, F, E,
# then the call sequence will be:
#   B   (because we ask for it)
#   F   (because we ask for it)
#         (note: then B does NOT get called gain, because it already ran)
#   E   (because F asks for it)
#   (then we should see F end)
#   (then E does NOT get called again, because it already ran)
#
# Finally, if we merely ask it to run whatever layer is necessary for
# output E.out, then we should see:
#   E   (because that's what makes E.out)
#   B   (because E pulls its output)
#

groupsetup = ("-layer A -param name A -param id 1 -param in 1.0 -param set_Ci 1 node " +
              "-layer B -param name B -param id 2 -param in 2.0 node " +
              "-layer C -param name C -param id 3 -param in 3.0 node " +
              "-layer D -param name D -param id 4 -param in 4.0 node " +
              "-layer E -param name E -param id 5 -param in 5.0 node -connect B out E in " +
              "-layer F -param name F -param id 6 -param in 6.0 node -connect E out F in " +
              "-layer G -param name G -param id 7 -param in 7.0 node -connect E out G in " +
              "--options llvm_debug_layers=1,lazyglobals=0 "
              )

def echoCmd(msg) :
    if (platform.system () == 'Windows'):
        return 'echo %s>> out.txt 2>&1 ;\n' % (msg)
    return 'echo "%s" >> out.txt 2>&1 ;\n' % (msg)

command += echoCmd('---') + echoCmd('default execution:')
command += testshade(groupsetup +
                     "-groupoutputs -o D.out out.exr ")

command += echoCmd('---') + echoCmd('explicit execution by layer (BFE):')
command += testshade(groupsetup +
                     "-O2 -groupoutputs -o D.out out.exr " +
                     "-entry B -entry F -entry E "
                     )

command += echoCmd('---') + echoCmd('explicit execution by output (E.out):')
command += testshade(groupsetup +
                     "-O2 -groupoutputs -o D.out out.exr " +
                     "-entry B -entry F -entry E --entryoutput E.out "
                     )
