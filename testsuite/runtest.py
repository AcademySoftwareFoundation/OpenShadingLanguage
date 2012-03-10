#!/usr/bin/python 

import os
import glob
import sys
import platform
import subprocess
from optparse import OptionParser

# Handy functions...

def osl_app (app):
    # when we use Visual Studio, built applications are stored
    # in the app/$(OutDir)/ directory, e.g., Release or Debug.
    # In that case the special token "$<CONFIGURATION>" which is replaced by
    # the actual configuration if one is specified. "$<CONFIGURATION>" works
    # because on Windows it is a forbidden filename due to the "<>" chars.
    if (platform.system () == 'Windows'):
        return app + "/$<CONFIGURATION>/" + app + " "
    return path + "/" + app + "/" + app + " "



# Construct a command that will compile the shader file, appending output to
# the file "out.txt".
def oslc (args) :
    return (osl_app("oslc") + args + " >> out.txt 2>&1 ;\n")



# Construct a command that will compile the shader file, appending output to
# the file "out.txt".
def testshade (args) :
    return (osl_app("testshade") + args + " >> out.txt 2>&1 ;\n")



def runtest (command, outputs, failureok=0, failthresh=0, failpercent=0) :
    parser = OptionParser()
    parser.add_option("-p", "--path", help="add to executable path",
                      action="store", type="string", dest="path", default="")
    parser.add_option("--devenv-config", help="use a MS Visual Studio configuration",
                      action="store", type="string", dest="devenv_config", default="")
    parser.add_option("--solution-path", help="MS Visual Studio solution path",
                      action="store", type="string", dest="solution_path", default="")
    (options, args) = parser.parse_args()

#    print ("working dir = " + tmpdir)
    os.chdir (srcdir)
    open ("out.txt", "w").close()    # truncate out.txt

    if options.path != "" :
        sys.path = [options.path] + sys.path
    print "command = " + command

    if (platform.system () == 'Windows'):
        # Replace the /$<CONFIGURATION>/ component added in oiio_app
        oiio_app_replace_str = "/"
        if options.devenv_config != "":
            oiio_app_replace_str = '/' + options.devenv_config + '/'
        command = command.replace ("/$<CONFIGURATION>/", oiio_app_replace_str)

    test_environ = None
    if (platform.system () == 'Windows') and (options.solution_path != "") and \
       (os.path.isdir (options.solution_path)):
        test_environ = os.environ
        libOIIO_path = options.solution_path + "\\libOpenImageIO\\"
        if options.devenv_config != "":
            libOIIO_path = libOIIO_path + '\\' + options.devenv_config
        test_environ["PATH"] = libOIIO_path + ';' + test_environ["PATH"]

    for sub_command in command.split(';'):
        cmdret = subprocess.call (sub_command, shell=True, env=test_environ)
        if cmdret != 0 and failureok == 0 :
            print "#### Error: this command failed: ", sub_command
            print "FAIL"
            return (1)

    err = 0
    for out in outputs :
        ok = 0
        # We will first compare out to ref/out, and if that fails, we will
        # compare it to everything else in the ref directory.  That allows us
        # to have multiple matching variants for different platforms, etc.
        for testfile in (["ref/"+out] + glob.glob (os.path.join ("ref", "*"))) :
            #print ("comparing " + out + " to " + testfile)
            extension = os.path.splitext(out)[1]
            if extension == ".tif" or extension == ".exr" :
                # images -- use idiff
                cmpcommand = (os.path.join (os.environ['OPENIMAGEIOHOME'], "bin", "idiff")
                              + " -fail 0" 
                              + " -failpercent " + str(failpercent)
                              + " -hardfail " + str(failthresh)
                              + " -warn " + str(2*failthresh)
                              + " " + out + " " + testfile)
            else :
                # anything else, mainly text files
                if (platform.system () == 'Windows'):
                    diff_cmd = "fc "
                else:
                    diff_cmd = "diff "
                cmpcommand = (diff_cmd + out + " " + testfile)

            print "cmpcommand = " + cmpcommand
            cmpresult = os.system (cmpcommand)
            if cmpresult == 0 :
                print ("PASS: " + out + " matches " + testfile)
                ok = 1
                break      # we're done
        
        if ok == 0:
            err = 1
            print "NO MATCH for " + out
            print "FAIL " + out

    return (err)



##########################################################################

#
# Get standard testsuite test arguments: srcdir exepath
#

srcdir = "."
tmpdir = "."
path = "../.."

if len(sys.argv) > 1 :
    srcdir = sys.argv[1]
    srcdir = os.path.abspath (srcdir) + "/"
    os.chdir (srcdir)
if len(sys.argv) > 2 :
    path = sys.argv[2]

tmpdir = "."
tmpdir = os.path.abspath (tmpdir)

refdir = "ref/"
parent = "../../../../../"

outputs = [ "out.txt" ]    # default

command = ""
failureok = 0
failthresh = 0.004
failpercent = 0.02


#print ("srcdir = " + srcdir)
#print ("tmpdir = " + tmpdir)
#print ("path = " + path)
#print ("refdir = " + refdir)



#
# Read the individual run.py file for this test, which will define 
# command and outputs.
#
if os.path.exists("run.py") :
    execfile ("run.py")

# Force out.txt to be in the outputs
if "out.txt" not in outputs :
    outputs.append ("out.txt")

# Force any local shaders to compile automatically, prepending the
# compilation onto whatever else the individual run.py file requested.
compiles = ""
for testfile in glob.glob ("*.osl") :
    compiles += oslc (testfile)
command = compiles + command

# If either out.exr or out.tif is in the reference directory but somehow
# is not in the outputs list, put it there anyway!
if (os.path.exists("ref/out.exr") and ("out.exr" not in outputs)) :
    outputs.append ("out.exr")
if (os.path.exists("ref/out.tif") and ("out.tif" not in outputs)) :
    outputs.append ("out.tif")

# Run the test and check the outputs
ret = runtest (command, outputs, failureok=failureok,
               failthresh=failthresh, failpercent=failpercent)
sys.exit (ret)
