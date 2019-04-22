#!/usr/bin/env python

from __future__ import print_function
import os
import glob
import sys
import platform
import subprocess
import difflib
import filecmp
import shutil

from optparse import OptionParser


#
# Get standard testsuite test arguments: srcdir exepath
#

srcdir = "."
tmpdir = "."

OSL_BUILD_DIR = os.environ.get("OSL_BUILD_DIR", "../..")
OSL_SOURCE_DIR = os.environ.get("OSL_SOURCE_DIR", "../../../../testsuite")
OSL_TESTSUITE_DIR = os.path.join(OSL_SOURCE_DIR, "testsuite")
OPENIMAGEIO_ROOT_DIR = os.environ.get("OPENIMAGEIO_ROOT_DIR", None)

os.environ['OSLHOME'] = os.path.join(OSL_SOURCE_DIR, "src")

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
    OSL_BUILD_DIR = args[1]
OSL_BUILD_DIR = os.path.normpath (OSL_BUILD_DIR)

tmpdir = "."
tmpdir = os.path.abspath (tmpdir)

refdir = "ref/"
test_source_dir = os.path.join(OSL_TESTSUITE_DIR,
                               os.path.basename(os.path.abspath(srcdir)))

command = ""
outputs = [ "out.txt" ]    # default
failureok = 0
failthresh = 0.004
hardfail = 0.01
failpercent = 0.02
oslcargs = "-Wall"

image_extensions = [ ".tif", ".tx", ".exr", ".jpg", ".png", ".rla",
                     ".dpx", ".iff", ".psd" ]

compile_osl_files = True
splitsymbol = ';'

#print ("srcdir = " + srcdir)
#print ("tmpdir = " + tmpdir)
#print ("path = " + path)
#print ("refdir = " + refdir)
print ("test source dir = ", test_source_dir)

if platform.system() == 'Windows' :
    if not os.path.exists("./ref") :
        shutil.copytree (os.path.join (test_source_dir, "ref"), "./ref")
    if os.path.exists (os.path.join (test_source_dir, "src")) and not os.path.exists("./src") :
        shutil.copytree (os.path.join (test_source_dir, "src"), "./src")
    if not os.path.exists(os.path.abspath("data")) :
        shutil.copytree (test_source_dir, os.path.abspath("data"))
    if not os.path.exists(os.path.abspath("../common")) :
        shutil.copytree (os.path.abspath(os.path.join(OSL_TESTSUITE_DIR, "common")),
                         os.path.abspath("../common"))
else :
    if not os.path.exists("./ref") :
        os.symlink (os.path.join (test_source_dir, "ref"), "./ref")
    if os.path.exists (os.path.join (test_source_dir, "src")) and not os.path.exists("./src") :
        os.symlink (os.path.join (test_source_dir, "src"), "./src")
    if not os.path.exists("./data") :
        os.symlink (test_source_dir, "./data")
    if not os.path.exists("../common") :
        os.symlink (os.path.join(OSL_TESTSUITE_DIR, "common"), "../common")

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
            print ("Diff " + fromfile + " vs " + tofile + " was:\n-------")
#            print (diff)
            print ("".join(diff_lines))
        except:
            print ("Unexpected error:", sys.exc_info()[0])
    return 1



def osl_app (app):
    apath = os.path.join(OSL_BUILD_DIR, "src", app)
    if (platform.system () == 'Windows'):
        # when we use Visual Studio, built applications are stored
        # in the app/$(OutDir)/ directory, e.g., Release or Debug.
        apath = os.path.join(apath, options.devenv_config)
    return os.path.join(apath, app) + " "


def oiio_relpath (path, start=os.curdir):
    "Wrapper around os.path.relpath which always uses '/' as the separator."
    p = os.path.relpath (path, start)
    return p if sys.platform != "win32" else p.replace ('\\', '/')


def oiio_app (app):
    if OPENIMAGEIO_ROOT_DIR :
        return os.path.join (OPENIMAGEIO_ROOT_DIR, "bin", app) + " "
    else :
        return app + " "


# Construct a command that will compile the shader file, appending output to
# the file "out.txt".
def oslc (args) :
    return (osl_app("oslc") + oslcargs + " " + args + " >> out.txt 2>&1 ;\n")


# Construct a command that will run oslinfo, appending output to
# the file "out.txt".
def oslinfo (args) :
    return (osl_app("oslinfo") + args + " >> out.txt 2>&1 ;\n")


# Construct a command that runs oiiotool, appending console output
# to the file "out.txt".
def oiiotool (args) :
    return (oiio_app("oiiotool") + args + " >> out.txt 2>&1 ;\n")


# Construct a command that runs maketx, appending console output
# to the file "out.txt".
def maketx (args) :
    return (oiio_app("maketx") + args + " >> out.txt 2>&1 ;\n")

# Construct a command that will compare two images, appending output to
# the file "out.txt".  We allow a small number of pixels to have up to
# 1 LSB (8 bit) error, it's very hard to make different platforms and
# compilers always match to every last floating point bit.
def oiiodiff (fileA, fileB, extraargs="", silent=True, concat=True) :
    command = (oiio_app("idiff") + "-a"
               + " -fail " + str(failthresh)
               + " -failpercent " + str(failpercent)
               + " -hardfail " + str(hardfail)
               + " -warn " + str(2*failthresh)
               + " -warnpercent " + str(failpercent)
               + " " + extraargs + " " + oiio_relpath(fileA,tmpdir)
               + " " + oiio_relpath(fileB,tmpdir))
    if not silent :
        command += " >> out.txt 2>&1 "
    if concat:
        command += " ;\n"
    return command


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
    os.environ["optix_log_level"] = "0"
    return (osl_app("testrender") + " " + args + " >> out.txt 2>&1 ;\n")


# Construct a command that run testoptix with the specified arguments,
# appending output to the file "out.txt".
def testoptix (args) :
    # Disable OptiX logging to prevent messages from the library from
    # appearing in the program output.
    os.environ["optix_log_level"] = "0"
    return (osl_app("testoptix") + " " + args + " >> out.txt 2>&1 ;\n")


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

    test_environ = None
    if (platform.system () == 'Windows') and (options.solution_path != "") and \
       (os.path.isdir (options.solution_path)):
        test_environ = os.environ
        libOIIO_path = options.solution_path + "\\libOpenImageIO\\"
        if options.devenv_config != "":
            libOIIO_path = libOIIO_path + '\\' + options.devenv_config
        test_environ["PATH"] = libOIIO_path + ';' + test_environ["PATH"]

    print ("command = ", command)
    for sub_command in command.split(splitsymbol):
        sub_command = sub_command.lstrip().rstrip()
        #print ("running = ", sub_command)
        cmdret = subprocess.call (sub_command, shell=True, env=test_environ)
        if cmdret != 0 and failureok == 0 :
            print ("#### Error: this command failed: ", sub_command)
            print ("FAIL")
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
                cmpcommand = oiiodiff (out, testfile, concat=False, silent=True)
                # print ("cmpcommand = ", cmpcommand)
                cmpresult = os.system (cmpcommand)
            elif extension == ".txt" :
                cmpresult = text_diff (out, testfile, out + ".diff")
            else :
                # anything else
                cmpresult = 0 if filecmp.cmp (out, testfile) else 1
            if cmpresult == 0 :
                ok = 1
                break      # we're done

        if ok :
            # if extension == ".tif" or extension == ".exr" or extension == ".jpg" or extension == ".png":
            #     # If we got a match for an image, save the idiff results
            #     os.system (oiiodiff (out, testfile, silent=False))
            print ("PASS: ", out, " matches ", testfile)
        else :
            err = 1
            print ("NO MATCH for ", out)
            print ("FAIL ", out)
            if extension == ".txt" :
                # If we failed to get a match for a text file, print the
                # file and the diff, for easy debugging.
                print ("-----" + out + "----->")
                print (open(out,'r').read() + "<----------")
                print ("Diff was:\n-------")
                print (open (out+".diff", 'rU').read())
            if extension == ".tif" or extension == ".exr" or extension == ".jpg" or extension == ".png":
                # If we failed to get a match for an image, send the idiff
                # results to the console
                os.system (oiiodiff (out, os.path.join (refdir, out), silent=False))

    return (err)


##########################################################################



#
# Read the individual run.py file for this test, which will define 
# command and outputs.
#
with open(os.path.join(test_source_dir,"run.py")) as f:
    code = compile(f.read(), "run.py", 'exec')
    exec (code)
# if os.path.exists("run.py") :
#     execfile ("run.py")

# Allow a little more slop for slight pixel differences when in DEBUG
# mode or when running on remote Travis-CI machines.
if (("TRAVIS" in os.environ and os.environ["TRAVIS"]) or
    ("DEBUG" in os.environ and os.environ["DEBUG"])) :
    failthresh *= 2.0
    hardfail *= 2.0
    failpercent *= 2.0

# Force out.txt to be in the outputs
##if "out.txt" not in outputs :
##    outputs.append ("out.txt")

# Force any local shaders to compile automatically, prepending the
# compilation onto whatever else the individual run.py file requested.
for filetype in [ "*.osl", "*.h", "*.oslgroup", "*.xml" ] :
    for testfile in glob.glob (os.path.join (test_source_dir, filetype)) :
        shutil.copyfile (testfile, os.path.basename(testfile))
if compile_osl_files :
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
