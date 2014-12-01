#!/usr/bin/python 

import os
import glob
import sys
import platform
import subprocess
import difflib
import filecmp

from optparse import OptionParser


#
# Get standard testsuite test arguments: srcdir exepath
#

srcdir = "."
tmpdir = "."
path = "../.."

# Options for the command line
parser = OptionParser()
parser.add_option("-p", "--path", help="add to executable path",
                  action="store", type="string", dest="path", default="")
parser.add_option("--devenv-config", help="use a MS Visual Studio configuration",
                  action="store", type="string", dest="devenv_config", default="")
parser.add_option("--solution-path", help="MS Visual Studio solution path",
                  action="store", type="string", dest="solution_path", default="")
(options, args) = parser.parse_args()

if args and len(args) > 0 :
    srcdir = args[0]
    srcdir = os.path.abspath (srcdir) + "/"
    os.chdir (srcdir)
if args and len(args) > 1 :
    path = args[1]
path = os.path.normpath (path)

tmpdir = "."
tmpdir = os.path.abspath (tmpdir)

refdir = "ref/"
parent = "../../../../../"

command = ""
outputs = [ "out.txt" ]    # default
failureok = 0
failthresh = 0.004
failpercent = 0.03

#print ("srcdir = " + srcdir)
#print ("tmpdir = " + tmpdir)
#print ("path = " + path)
#print ("refdir = " + refdir)

###########################################################################

# Handy functions...

# Compare two text files. Returns 0 if they are equal otherwise returns
# a non-zero value and writes the differences to "diff_file".
# Based on the command-line interface to difflib example from the Python
# documentation
def text_diff (fromfile, tofile, diff_file=None):
    import time
    try:
        fromdate = time.ctime (os.stat (fromfile).st_mtime)
        todate = time.ctime (os.stat (tofile).st_mtime)
        fromlines = open (fromfile, 'rU').readlines()
        tolines   = open (tofile, 'rU').readlines()
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        return -1
        
    diff = difflib.unified_diff(fromlines, tolines, fromfile, tofile,
                                fromdate, todate)
    # Diff is a generator, but since we need a way to tell if it is
    # empty we just store all the text in advance
    diff_lines = [l for l in diff]
    if not diff_lines:
        return 0
    if diff_file:
        try:
            open (diff_file, 'w').writelines (diff_lines)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
    return 1



def osl_app (app):
    # when we use Visual Studio, built applications are stored
    # in the app/$(OutDir)/ directory, e.g., Release or Debug.
    # In that case the special token "$<CONFIGURATION>" which is replaced by
    # the actual configuration if one is specified. "$<CONFIGURATION>" works
    # because on Windows it is a forbidden filename due to the "<>" chars.
    if (platform.system () == 'Windows'):
        return app + "/$<CONFIGURATION>/" + app + " "
    return path + "/src/" + app + "/" + app + " "



# Construct a command that will compile the shader file, appending output to
# the file "out.txt".
def oslc (args) :
    return (osl_app("oslc") + args + " >> out.txt 2>&1 ;\n")


# Construct a command that will run oslinfo, appending output to
# the file "out.txt".
def oslinfo (args) :
    return (osl_app("oslinfo") + args + " >> out.txt 2>&1 ;\n")


# Construct a command that runs oiiotool, appending console output
# to the file "out.txt".
def oiiotool (args) :
    return (os.path.join (os.environ['OPENIMAGEIOHOME'], "bin", "oiiotool") +
            " " + args + " >> out.txt 2>&1 ;\n")


# Construct a command that runs maketx, appending console output
# to the file "out.txt".
def maketx (args) :
    return (os.path.join (os.environ['OPENIMAGEIOHOME'], "bin", "maketx") +
            " " + args + " >> out.txt 2>&1 ;\n")


# Construct a command that run testshade with the specified arguments,
# appending output to the file "out.txt".
def testshade (args) :
    if os.environ.__contains__('OSL_TESTSHADE_NAME') :
        testshadename = os.environ['OSL_TESTSHADE_NAME'] + " "
    else :
        testshadename = osl_app("testshade")
    return (testshadename + args + " >> out.txt 2>&1 ;\n")


# Construct a command that run testrender with the specified arguments,
# appending output to the file "out.txt".
def testrender (args) :
    return (osl_app("testrender") + " -v --stats " + args + " >> out.txt 2>&1 ;\n")


# Run 'command'.  For each file in 'outputs', compare it to the copy
# in 'ref/'.  If all outputs match their reference copies, return 0
# to pass.  If any outputs do not match their references return 1 to
# fail.
def runtest (command, outputs, failureok=0, failthresh=0, failpercent=0) :
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
        extension = os.path.splitext(out)[1]
        ok = 0
        # We will first compare out to ref/out, and if that fails, we
        # will compare it to everything else with the same extension in
        # the ref directory.  That allows us to have multiple matching
        # variants for different platforms, etc.
        for testfile in (["ref/"+out] + glob.glob (os.path.join ("ref", "*"+extension))) :
            # print ("comparing " + out + " to " + testfile)
            if extension == ".tif" or extension == ".exr" :
                # images -- use idiff
                cmpcommand = (os.path.join (os.environ['OPENIMAGEIOHOME'], "bin", "idiff")
                              + " -fail 0" 
                              + " -failpercent " + str(failpercent)
                              + " -hardfail " + str(failthresh)
                              + " -warn " + str(2*failthresh)
                              + " " + out + " " + testfile)
                # print "cmpcommand = " + cmpcommand
                cmpresult = os.system (cmpcommand)
            elif extension == ".txt" :
                cmpresult = text_diff (out, testfile, out + ".diff")
            else :
                # anything else
                cmpresult = 0 if filecmp.cmp (out, testfile) else 1
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
# Read the individual run.py file for this test, which will define 
# command and outputs.
#
if os.path.exists("run.py") :
    execfile ("run.py")

# Force out.txt to be in the outputs
##if "out.txt" not in outputs :
##    outputs.append ("out.txt")

# Force any local shaders to compile automatically, prepending the
# compilation onto whatever else the individual run.py file requested.
compiles = ""
oslfiles = glob.glob ("*.osl")
oslfiles.sort() ## sort the shaders to compile so that they always compile in the same order
for testfile in oslfiles :
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
