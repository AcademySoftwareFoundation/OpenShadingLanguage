#!/usr/bin/env python

# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

import os
import glob
import shlex
import sys
import platform
import subprocess
import difflib
import filecmp
import shutil
import re
from itertools import chain
from collections.abc import Iterator
from typing import Union
import argparse


def make_relpath (path, start=os.curdir):
    "Wrapper around os.path.relpath which always uses '/' as the separator."
    p = os.path.relpath (path, start)
    return p if sys.platform != "Windows" else p.replace ('\\', '/')


#
# Get standard testsuite test arguments: srcdir exepath
#

srcdir = "."
tmpdir = "."

OSL_BUILD_DIR = os.environ.get("OSL_BUILD_DIR", "..")
OSL_SOURCE_DIR = os.environ.get("OSL_SOURCE_DIR", "../../..")
OSL_TESTSUITE_DIR = os.path.join(OSL_SOURCE_DIR, "testsuite")
OpenImageIO_ROOT = os.environ.get("OpenImageIO_ROOT", None)
OSL_TESTSUITE_ROOT = make_relpath(os.getenv('OSL_TESTSUITE_ROOT',
                                             '../../../testsuite'))
os.environ['OSLHOME'] = os.path.join(OSL_SOURCE_DIR, "src")
OSL_REGRESSION_TEST = os.environ.get("OSL_REGRESSION_TEST", None)

# Options for the command line
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="add to executable path",
                    action="store", dest="path", default="")
parser.add_argument("--devenv-config", help="use a MS Visual Studio configuration",
                    action="store", dest="devenv_config", default="")
parser.add_argument("--solution-path", help="MS Visual Studio solution path",
                    action="store", dest="solution_path", default="")
parser.add_argument("--backend", help="use this backend to execute test",
                    action="store", default="OSO")
parser.add_argument("--baseline", help="backend baseline to compare to, by default same as the backend being tested",
                    action='store', default=None)
parser.add_argument("--lazy_oslc", help="only compile shader if the source or the compiler are younger than the corresponding artifact",
                    action='store_true')

class AppendEdit(argparse.Action):
    """ Convert filter_re argument into the corresponding --edit argument """
    def __call__(self, parser, namespace, values, option_string=None):
        modifiers = ''
        if values[0] == '~':
            values = values[1:]
            modifiers += 'v'
        setattr(namespace, self.dest, f"m/{values}/{modifiers}")

parser.add_argument("--filter_re", help="only compare lines matching these python re's, and not matching the ones where the first character is '~'",
                    action=AppendEdit, dest='edits', default=[])
parser.add_argument("--edit", help="apply these edits to the lines before comparing. s/// is used for regsub and m// is used for matching",
                    action='append', dest='edits', default=[])
parser.add_argument("args", nargs='*')
options = parser.parse_args()

args = options.args
if args and len(args) > 0 :
    srcdir = args[0]
    srcdir = os.path.abspath (srcdir) + "/"
    os.chdir (srcdir)
if args and len(args) > 1 :
    OSL_BUILD_DIR = args[1]
OSL_BUILD_DIR = os.path.normpath (OSL_BUILD_DIR)

options.backend = options.backend.upper()
if options.backend not in ["OSO", "MSB"]:
    print("WARNING: requested unknown backend '%s', using 'OSO'" % options.backend)
    options.backend = "OSO"

if options.baseline:
    options.baseline = options.baseline.upper()
    if options.baseline not in ["OSO", "MSB"]:
        print("WARNING: requested unknown baseline backend '%s', using '%s' (same as backend)" % (options.baseline, options.backend))
        options.baseline = options.backend

tmpdir = "."
tmpdir = os.path.abspath (tmpdir)
out_txt = "out.txt"

if platform.system() == 'Windows' :
    redirect = " >> %s 2>&1 " % out_txt
else :
    redirect = " >> %s 2>>%s " % (out_txt, out_txt)

refbasedir = "ref"       # base directory for references
refdir = refbasedir      # this is usually <refbasedir>/<backend>
refdir_fallback = refdir # this is what we use if <refdir>/myfile does not exists
baselinedir = "baseline"
mytest = os.path.split(os.path.abspath(os.getcwd()))[-1]
if str(mytest).endswith('.opt') or str(mytest).endswith('.optix') :
    mytest = mytest.split('.')[0]
test_source_dir = os.getenv('OSL_TESTSUITE_SRC',
                            os.path.join(OSL_TESTSUITE_ROOT, mytest))
#test_source_dir = os.path.join(OSL_TESTSUITE_DIR,
#                               os.path.basename(os.path.abspath(srcdir)))

command = ""
outputs = [ out_txt ]    # default

# The image comparison thresholds are tricky to remember. Here's the key:
# A test fails if more than `failpercent` of pixel values differ by more
# than `failthresh` AND the difference is more than `failrelative` times the
# correct pixel value, or if even one pixel differs by more than `hardfail`.
failthresh = 0.004         # "Failure" threshold for any pixel value
hardfail = 0.01            # Even one pixel this wrong => hard failure
failpercent = 0.02         # Ok fo this percentage of pixels to "fail"
failrelative = 0.001       # Ok to fail up to this amount vs the pixel value
allowfailures = 0          # Freebie failures

# Some tests are designed for the app running to "fail" (in the sense of
# terminating with an error return code), for example, a test that is designed
# to present an error condition to check that it issues the right error. That
# "failure" is a success of the test! For those cases, set `failureok = 1` to
# indicate that the app having an error is fine, and the full test will pass
# or fail based on comparing the output files.
failureok = 0

idiff_program = "oiiotool"
idiff_postfilecmd = ""
skip_diff = int(os.environ.get("OSL_TESTSUITE_SKIP_DIFF", "0"))

cleanup_on_success = False
if int(os.getenv('TESTSUITE_CLEANUP_ON_SUCCESS', '0')) :
    cleanup_on_success = True
oslcargs = "-Wall"
oslinfoargs = ""
testshadeargs = ""
testrenderargs = ""
testoptixargs = ""

if options.backend != "OSO":
    if options.backend == "MSB":
        oslcargs += " -g -msb -save-temps"
    else:
        print("ERROR: requested backend '%s', but runtest.py doesn't know the corresponding option for 'oslc'. Abort." % options.backend)
        exit(1)
    #oslinfoargs += "" # no change needed
    testshadeargs += " --backend " + options.backend
    testrenderargs += " --backend " + options.backend
    testoptixargs += " --backend " + options.backend
    refdir = os.path.join(refdir, options.backend)

if not options.baseline:
    baselinedir = "baseline/" + options.backend
elif options.baseline != "OSO":
    baselinedir = "baseline/" + options.baseline

image_extensions = [ ".tif", ".tx", ".exr", ".jpg", ".png", ".rla",
                     ".dpx", ".iff", ".psd" ]

compile_osl_files = True
splitsymbol = ';'

#print("OSL_BUILD_DIR=%s" % (OSL_BUILD_DIR if OSL_BUILD_DIR else '(None)'))
#print("OSL_SOURCE_DIR=%s" % (OSL_SOURCE_DIR if OSL_SOURCE_DIR else '(None)'))
#print("OSL_TESTSUITE_DIR=%s" % (OSL_TESTSUITE_DIR if OSL_TESTSUITE_DIR else '(None)'))
#print("OpenImageIO_ROOT=%s" % (OpenImageIO_ROOT if OpenImageIO_ROOT else '(None)'))
#print("OSL_TESTSUITE_ROOT=%s" % (OSL_TESTSUITE_ROOT if OSL_TESTSUITE_ROOT else '(None)'))
#print("OSL_REGRESSION_TEST=%s" % (OSL_REGRESSION_TEST if OSL_REGRESSION_TEST else '(None)'))

#print ("srcdir = " + srcdir)
#print ("tmpdir = " + tmpdir)
#print ("path = " + path) # path variable seems to be undefined
print ("baselinedir = " + baselinedir)
print ("refdir = " + refdir)
print ("refdir_fallback = " + refdir_fallback)
print ("test source dir = ", test_source_dir)
print ("backend = ", options.backend )
print ("baseline = ", options.baseline if options.baseline else "(None)" )
print (f"edit = {options.edits}")
if platform.system() == 'Windows' :
    if not os.path.exists(os.path.join(".", refbasedir)) :
        test_source_ref_dir = os.path.join (test_source_dir, refbasedir)
        if os.path.exists(test_source_ref_dir) :
            shutil.copytree (test_source_ref_dir, os.path.join(".", refbasedir))
    if os.path.exists (os.path.join (test_source_dir, "src")) and not os.path.exists("./src") :
        shutil.copytree (os.path.join (test_source_dir, "src"), "./src")
    if not os.path.exists(os.path.abspath("data")) :
        shutil.copytree (test_source_dir, os.path.abspath("data"))
else :
    if not os.path.exists(os.path.join(".", refbasedir)) :
        test_source_ref_dir = os.path.join (test_source_dir, refbasedir)
        if os.path.exists(test_source_ref_dir) :
            os.symlink (test_source_ref_dir, os.path.join(".", refbasedir))
    if os.path.exists (os.path.join (test_source_dir, "src")) and not os.path.exists("./src") :
        os.symlink (os.path.join (test_source_dir, "src"), "./src")
    if not os.path.exists("./data") :
        os.symlink (test_source_dir, "./data")

pythonbin = sys.executable
#print ("pythonbin = ", pythonbin)

###########################################################################

# Handy functions...

def s_edit(sub_pat: str, input: Iterator[str]):
    """
    A generator that returns a regsubbed version of the stream passed as input
    This is more or less sed's or PERL's 's/pat/sub/modifiers;'
    The character after the leading 's' can be anything, and exactly three must be present
    in the string, after escaping with backslash. Unlike PERL, we don't implement
    matching delimiters (in PERL you'd have s{pat}{sub}modifiers, for all delimiters
    that come in open/close pairs, but this is not implemented here)
    Currently accepted modifiers are:
      - i - case-insensitive matching
      - g - global replace
    """

    # absorb leading 's'
    if sub_pat[0] == 's':
        sub_pat = sub_pat[1:]

    # split and manage escaping
    toks = []
    for tok in sub_pat.split(sub_pat[0])[1:]:
        if len(toks) and toks[-1][-1] == '\\':
            toks[-1] = toks[-1][:-1] + tok
        else:
            toks.append(tok)

    try:
        pat, sub, modifiers = toks
    except:
        raise ValueError(f"Malformed substitution expression '{sub_pat}'")

    flags = re.IGNORECASE if 'i' in modifiers else re.NOFLAG
    count = 0 if 'g' in modifiers else 1
    expr = re.compile(pat, flags)
    for line in input:
        yield expr.sub(sub, line, count=count)


def m_edit(match_pat: str, input: Iterator[str]):
    """
    A generator that returns a grepped version of the stream passed as input
    This is more or less PERL's 'print if m/pat/modifiers;'
    The character after the leading 'm' can be anything, and exactly two must be present
    in the string, after escaping with backslash. Unlike PERL, we don't implement
    matching delimiters (in PERL you'd have m{pat}modifiers, for all delimiters
    that come in open/close pairs, but this is not implemented here)
    Currently accepted modifiers are:
      - i - case-insensitive matching
      - v - invert the meaning of the match (like `grep -v`)
      - a<num> - also match num lines after an actual match
      - b<num> - also match num lines before an actual match
    """

    # absorb leading 'm'
    if match_pat[0] == 'm':
        match_pat = match_pat[1:]

    # split and manage escaping
    toks = []
    for tok in match_pat.split(match_pat[0])[1:]:
        if len(toks) and toks[-1][-1] == '\\':
            toks[-1] = toks[-1][:-1] + tok
        else:
            toks.append(tok)

    try:
        pat, modifiers = toks
    except:
        raise ValueError(f"Malformed match expression '{match_pat}'")

    flags = re.IGNORECASE if 'i' in modifiers else re.NOFLAG
    invert = 'v' in modifiers
    after = 0
    before = 0
    if 'a' in modifiers:
        start = modifiers.index('a') + 1
        end = start
        while end < len(modifiers) and modifiers[end].isdigit():
            end += 1
        after = int(modifiers[start:end])

    if 'b' in modifiers:
        start = modifiers.index('b') + 1
        end = start
        while end < len(modifiers) and modifiers[end].isdigit():
            end += 1
        before = int(modifiers[start:end])

    beforebuf = []
    aftercount = 0
    expr = re.compile(pat, flags)
    for line in input:
        yielded = False
        if aftercount:
            aftercount -= 1
            yielded = True
            yield line
            # when we re-enter the generator we
            # go to check the match before looping around
        elif before:
            # don't capture a line during the "after" phase,
            # otherwise it would flow out twice
            beforebuf.append(line)
            while len(beforebuf) > before:
                beforebuf.pop(0)

        ismatch = expr.search(line) is not None
        if invert:
            ismatch = not ismatch
        if not ismatch:
            continue

        # if we're here, the line is a match
        aftercount = after
        # spool out the before buffer
        yielded = yielded or len(beforebuf) > 0
        while len(beforebuf):
            yield beforebuf.pop(0)


        # as well as our matching line, unless it's already gone out
        if not yielded:
            yield line

def edit_file(fname: str, edits: list[str]) -> list[str]:
    with open(fname, 'r') as f:
        gen = f
        # build our edit chain
        for edit in edits:
            try:
                if edit[0] == 's':
                    gen = s_edit(edit, gen)
                elif edit[0] == 'm':
                    gen = m_edit(edit, gen)
                else:
                    print(f"Unknown edit command '{edit[0]}' in expression '{edit}'")
            except:
                print(f"Can't add edit command '{edit}'")
        # spool it all back out
        return list(gen)

# Compare two text files. Returns 0 if they are equal otherwise returns
# a non-zero value and writes the differences to "diff_file".
# Based on the command-line interface to difflib example from the Python
# documentation, with the added twist that we apply the edits before the diff
def text_diff (fromfile: str, tofile: str, diff_file: str =None, edits: list[str] =[]):
    import time
    try:
        fromdate = time.ctime (os.stat (fromfile).st_mtime)
        todate = time.ctime (os.stat (tofile).st_mtime)

        fromlines = edit_file(fromfile, edits)
        tolines = edit_file(tofile, edits)
    except FileNotFoundError as e:
        print ("File not found:", e.filename)
        return -1
    except:
        print ("Unexpected error:", sys.exc_info()[0])
        return -1
        
    diff = difflib.unified_diff(fromlines, tolines,
                                fromfile, tofile,
                                fromdate, todate)
    # Diff is a generator, but since we need a way to tell if it is
    # empty we just store all the text in advance
    diff_lines = list(diff)
    if not diff_lines:
        return 0
    if diff_file:
        try:
            open (diff_file, 'w').writelines (diff_lines)
            print ("Diff " + fromfile + " vs " + tofile + " was:\n-------")
            for l in diff_lines:
                print(l, end='')
        except:
            print ("Unexpected error:", sys.exc_info()[0])
    return 1



def run_app (app, silent=False, concat=True) :
    command = app
    if not silent :
        command += redirect
    if concat:
        command += " ;\n"
    return command


def osl_app (app):
    apath = os.path.join(OSL_BUILD_DIR, "bin")
    if (platform.system () == 'Windows'):
        # when we use Visual Studio, built applications are stored
        # in the app/$(OutDir)/ directory, e.g., Release or Debug.
        apath = os.path.join(apath, options.devenv_config)
    return os.path.join(apath, app) + " "


def oiio_app (app):
    if OpenImageIO_ROOT :
        return os.path.join (OpenImageIO_ROOT, "bin", app) + " "
    else :
        return app + " "


def get_artifacts(source_or_cmd: str) -> list[str]:
    """
    Try to divine the resulting artifacts from a job.
    If unable, return empty list
    (Work in progress)
    """
    toks = shlex.split(source_or_cmd)
    source = toks[-1]  # FIXME: this is horrible
    if source.endswith(".osl"):
        if options.backend.upper() == "OSO":
            return [source[:-1] + 'o']
        elif options.backend.upper() == "MSB":
            return [source[:-1] + 'm']
    return []

oslc_mtime = None
def needsrecompile(source: str) -> bool:
    """
    Determine if the given shader needs to be recompiled
    """
    global oslc_mtime
    if not options.lazy_oslc:
        return True
    artifact = get_artifacts(source)
    if not len(artifact):
        return True
    artifact = artifact[0]
    artifact_mtime = os.path.getmtime(artifact) if os.path.exists(artifact) else 0
    if not oslc_mtime:
        oslc_bin = osl_app("oslc").strip()
        oslc_mtime = os.path.getmtime(oslc_bin)
    src_mtime = os.path.getmtime(source)
    if src_mtime > artifact_mtime or oslc_mtime > artifact_mtime:
        return True
    # "cache hit"
    return False


# Construct a command that will compile the shader file, appending output to
# the file "out.txt".
def oslc (args) :
    return (osl_app("oslc") + oslcargs + " " + args + redirect + " ;\n")


# Construct a command that will run oslinfo, appending output to
# the file "out.txt".
def oslinfo (args) :
    return (osl_app("oslinfo") + oslinfoargs + " "+ args + redirect + " ;\n")


# Construct a command that runs oiiotool, appending console output
# to the file "out.txt".
def oiiotool (args, silent=False) :
    oiiotool_cmd = (oiio_app("oiiotool") + args)
    if not silent :
        oiiotool_cmd += redirect
    oiiotool_cmd += " ;\n"
    return oiiotool_cmd

# Construct a command that runs maketx, appending console output
# to the file "out.txt".
def maketx (args) :
    return (oiio_app("maketx") + args + redirect + " ;\n")

# Construct a command that will compare two images, appending output to
# the file "out.txt".  We allow a small number of pixels to have up to
# 1 LSB (8 bit) error, it's very hard to make different platforms and
# compilers always match to every last floating point bit.
def oiiodiff (fileA, fileB, extraargs="", silent=True, concat=True) :
    threshargs = (" -fail " + str(failthresh)
               + " -failpercent " + str(failpercent)
               + " -hardfail " + str(hardfail)
               + " -warn " + str(2*failthresh)
               + " -warnpercent " + str(failpercent))
    if idiff_program == "idiff" :
        threshargs += (" -failrelative " + str(failrelative)
                     + " -allowfailures " + str(allowfailures))
    command = (oiio_app(idiff_program) + "-a"
               + " " + threshargs
               + " " + extraargs
               + " " + make_relpath(fileA,tmpdir) + idiff_postfilecmd
               + " " + make_relpath(fileB,tmpdir) + idiff_postfilecmd
               + (" --diff" if idiff_program == "oiiotool" else ""))
    if not silent :
        command += redirect
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
    return (testshadename + " " + testshadeargs + " " + args + redirect + " ;\n")


# Construct a command that run testrender with the specified arguments,
# appending output to the file "out.txt".
def testrender (args) :
    os.environ["optix_log_level"] = "0"
    return (osl_app("testrender") + " " + testrenderargs + " " + args + redirect + " ;\n")


# Construct a command that run testoptix with the specified arguments,
# appending output to the file "out.txt".
def testoptix (args) :
    # Disable OptiX logging to prevent messages from the library from
    # appearing in the program output.
    os.environ["optix_log_level"] = "0"
    return (osl_app("testoptix") + " " + testoptixargs + " " + args + redirect + " ;\n")


# Run 'command'.  For each file in 'outputs', compare it to the copy
# in refdir.  If all outputs match their reference copies, return 0
# to pass.  If any outputs do not match their references return 1 to
# fail.
def runtest (command, outputs, failureok=0, failthresh=0, failpercent=0, regression=None, edits=[]) :
#    print ("working dir = " + tmpdir)
    os.chdir (srcdir)
    open (out_txt, "w").close()    # truncate out_txt

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

    if regression == "BATCHED" :
        if test_environ == None :
            test_environ = os.environ
        test_environ["TESTSHADE_BATCHED"] = "1"

    if regression == "RS_BITCODE" :
        if test_environ == None :
            test_environ = os.environ
        test_environ["TESTSHADE_RS_BITCODE"] = "1"

    print ("command = ", command)

    for sub_command in command.split(splitsymbol):
        sub_command = sub_command.strip()
        if not sub_command:
            continue
        print ("running = ", sub_command)
        cmdret = subprocess.call (sub_command, shell=True, env=test_environ)
        if cmdret != 0 and failureok == 0 :
            print ("#### Error: this command failed: ", sub_command)
            print ("FAIL")
            print ("Output was:\n--------")
            print (open (out_txt, 'r').read())
            print ("--------")
            return (1)

    if skip_diff :
        return 0

    err = 0
    if regression == "BASELINE" :
        if not os.path.exists(os.path.join(".", baselinedir)) :
            os.makedirs(os.path.join(".", baselinedir))
        for out in outputs :
            shutil.move(out, os.path.join(".", baselinedir, out)) 
    else :
        for out in outputs :
            basefname,extension = os.path.splitext(out)
            ok = 0
            # We will first compare out to <refdir>/out, and if that fails, we
            # will compare it to everything else with the same basefilename and extension in
            # the ref directory.  That allows us to have multiple matching
            # variants for different platforms, etc.
            if regression != None:
                testfiles = [os.path.join(baselinedir, out)]
            else:
                if os.path.exists(refdir):
                    testfiles = [os.path.join (refdir,out)] + glob.glob (os.path.join (refdir, basefname+"*"+extension))
                else:
                    testfiles = [os.path.join(refdir_fallback, out)] + glob.glob(os.path.join(refdir_fallback, basefname+"*" + extension))
                # requires python 3.7+ to preserve ordering
                dedup = lambda lst: list(dict.fromkeys(lst))
                testfiles = dedup(testfiles)
            cmpresult = None
            for testfile in (testfiles) :
                # print ("comparing " + out + " to " + testfile)
                if extension == ".tif" or extension == ".exr" :
                    # images -- use idiff
                    cmpcommand = oiiodiff (out, testfile, concat=False, silent=True)
                    # print ("cmpcommand = ", cmpcommand)
                    cmpresult = os.system (cmpcommand)
                elif extension == ".txt" :
                    cmpresult = text_diff (out, testfile, out + ".diff", edits=edits)
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
                if cmpresult is None:
                    print("NO REFERENCE FOUND for ", out)
                else:
                    print ("NO MATCH for ", out)
                print ("FAIL ", out)
                if extension == ".txt" :
                    # If we failed to get a match for a text file, print the
                    # file and the diff, for easy debugging.
                    print ("-----" + out + "----->")
                    print (open(out,'r').read() + "<----------")
                    print ("Diff was:\n-------")
                    try:
                        print (open (out+".diff", 'r').read())
                        pass
                    except FileNotFoundError as e:
                        print ("File not found:", e.filename)
                if extension == ".tif" or extension == ".exr" or extension == ".jpg" or extension == ".png":
                    # If we failed to get a match for an image, send the idiff
                    # results to the console
                    testfile = None
                    if regression != None:
                        testfile = os.path.join (baselinedir, out)
                    else :
                        testfile = os.path.join (refdir, out)
                    os.system (oiiodiff (out, testfile, silent=False))

    return (err)


##########################################################################

# flush every line, even when we're going into a pipe
sys.stdout.reconfigure(line_buffering=True, write_through=True)

#
# Read the individual run.py file for this test, which will define 
# command and outputs.
#
with open(os.path.join(test_source_dir,"run.py")) as f:
    code = compile(f.read(), "run.py", 'exec')
    exec (code)
# if os.path.exists("run.py") :
#     execfile ("run.py")


# Allow a little more slop for slight pixel differences when in DEBUG mode.
if "DEBUG" in os.environ and os.environ["DEBUG"] :
    failthresh *= 2.0
    hardfail *= 2.0
    failpercent *= 2.0

# Allow an environment variable to scale the testsuite image comparison
# thresholds:
if 'OSL_TESTSUITE_THRESH_SCALE' in os.environ :
    thresh_scale = float(os.getenv('OSL_TESTSUITE_THRESH_SCALE', '1.0'))
    failthresh *= thresh_scale
    hardfail *= thresh_scale
    failpercent *= thresh_scale
    failrelative *= thresh_scale
    allowfailures = int(allowfailures * thresh_scale)

# Force out_txt to be in the outputs
##if out_txt not in outputs :
##    outputs.append (out_txt)

# Force any local shaders to compile automatically, prepending the
# compilation onto whatever else the individual run.py file requested.
for filetype in [ "*.osl", "*.h", "*.oslgroup", "*.xml" ] :
    for testfile in glob.glob (os.path.join (test_source_dir, filetype)) :
        dest = os.path.basename(testfile)
        if options.lazy_oslc and (not os.path.exists(dest) or os.path.getmtime(testfile) > os.path.getmtime(dest)):
            shutil.copyfile (testfile, dest)
if compile_osl_files :
    compiles = ""
    oslfiles = glob.glob ("*.osl")
    oslfiles.sort() ## sort the shaders to compile so that they always compile in the same order
    for testfile in oslfiles :
        if needsrecompile(testfile):
            compiles += oslc (testfile)
        else:
            # pretend we did compile
            compiles += 'echo "Compiled %s -> %s" %s ;\n' % (testfile, get_artifacts(testfile)[0], redirect)
    command = compiles + command

# If either out.exr or out.tif is in the reference directory but somehow
# is not in the outputs list, put it there anyway!
if (os.path.exists(os.path.join(refdir, "out.exr")) and ("out.exr" not in outputs)) :
    outputs.append ("out.exr")
if (os.path.exists(os.path.join(refdir, "out.tif")) and ("out.tif" not in outputs)) :
    outputs.append ("out.tif")

# Run the test and check the outputs
if OSL_REGRESSION_TEST != None :
    # need to produce baseline images, but we only know how to do this if the backend and the baseline are the same
    ret = 0
    if options.baseline == options.backend:
        print(" -- Building baseline data for backend %s" % options.backend)
        ret = runtest (command, outputs, failureok=failureok,
                   failthresh=failthresh, failpercent=failpercent, regression="BASELINE", edits=options.edits)
    if ret == 0 :
        # run again comparing against baseline, not ref
        ret = runtest (command, outputs, failureok=failureok,
                       failthresh=failthresh, failpercent=failpercent, regression=OSL_REGRESSION_TEST, edits=options.edits)
else :                   
    ret = runtest (command, outputs, failureok=failureok,
                   failthresh=failthresh, failpercent=failpercent, edits=options.edits)
    
if ret == 0 and cleanup_on_success :
    for ext in image_extensions + [ ".txt", ".diff", ".oso" ] :
        files = glob.iglob (srcdir + '/*' + ext)
        baselineFiles = glob.iglob (os.path.join(srcdir, baselinedir, '*' + ext)) 
        for f in chain(files,baselineFiles) :
            os.remove(f)
            #print('REMOVED ', f)

if ret != 0:
    print("Reproduce with:")
    test_build_dir = os.path.join(OSL_BUILD_DIR, make_relpath(test_source_dir, OSL_SOURCE_DIR))
    print("cd", test_build_dir)
    print(command)

sys.exit (ret)
