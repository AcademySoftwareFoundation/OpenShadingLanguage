#
# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: Apache License
# Coyright Notice From Pixar USD (Apache License)
# 90% of the content is changed to be compatible for installing OSL and its dependencies
# 
# Based on USD build_scripts by Pixar Animation Studio
# 
# ---------------------------------------------------------------------------
#

from __future__ import print_function

from distutils.spawn import find_executable

import argparse
import codecs
import contextlib
import ctypes
import datetime
import distutils
import fnmatch
import glob
import locale
import multiprocessing
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import zipfile

if sys.version_info.major >= 3:
    from urllib.request import urlopen
else:
    from urllib2 import urlopen

# Helpers for printing output
verbosity = 1


def Print(msg):
    if verbosity > 0:
        print(msg)


def PrintWarning(warning):
    if verbosity > 0:
        print("WARNING:", warning)


def PrintStatus(status):
    if verbosity >= 1:
        print("STATUS:", status)


def PrintInfo(info):
    if verbosity >= 2:
        print("INFO:", info)


def PrintCommandOutput(output):
    if verbosity >= 3:
        sys.stdout.write(output)


def PrintError(error):
    if verbosity >= 3 and sys.exc_info()[1] is not None:
        import traceback

        traceback.print_exc()
    print("ERROR:", error)


# Helpers for determining platform
def Windows():
    return platform.system() == "Windows"


def Linux():
    return platform.system() == "Linux"


def MacOS():
    return platform.system() == "Darwin"


def Python3():
    return sys.version_info.major == 3


def GetLocale():
    return sys.stdout.encoding or locale.getdefaultlocale()[1] or "UTF-8"


def GetCommandOutput(command):
    """Executes the specified command and returns output or None."""
    try:
        return (
            subprocess.check_output(shlex.split(command), stderr=subprocess.STDOUT)
            .decode(GetLocale(), "replace")
            .strip()
        )
    except subprocess.CalledProcessError:
        pass
    return None


def GetXcodeDeveloperDirectory():
    """Returns the active developer directory as reported by 'xcode-select -p'.
    Returns None if none is set."""
    if not MacOS():
        return None

    return GetCommandOutput("xcode-select -p")


def GetVisualStudioCompilerAndVersion():
    """Returns a tuple containing the path to the Visual Studio compiler
    and a tuple for its version, e.g. (14, 0). If the compiler is not found
    or version number cannot be determined, returns None."""
    if not Windows():
        return None

    msvcCompiler = find_executable("cl")
    if msvcCompiler:
        # VisualStudioVersion environment variable should be set by the
        # Visual Studio Command Prompt.
        match = re.search(r"(\d+)\.(\d+)", os.environ.get("VisualStudioVersion", ""))
        if match:
            return (msvcCompiler, tuple(int(v) for v in match.groups()))
    return None


def IsVisualStudioVersionOrGreater(desiredVersion):
    if not Windows():
        return False

    msvcCompilerAndVersion = GetVisualStudioCompilerAndVersion()
    if msvcCompilerAndVersion:
        _, version = msvcCompilerAndVersion
        return version >= desiredVersion
    return False


def IsVisualStudio2019OrGreater():
    VISUAL_STUDIO_2019_VERSION = (16, 0)
    return IsVisualStudioVersionOrGreater(VISUAL_STUDIO_2019_VERSION)


def IsVisualStudio2017OrGreater():
    VISUAL_STUDIO_2017_VERSION = (15, 0)
    return IsVisualStudioVersionOrGreater(VISUAL_STUDIO_2017_VERSION)


def IsVisualStudio2015OrGreater():
    VISUAL_STUDIO_2015_VERSION = (14, 0)
    return IsVisualStudioVersionOrGreater(VISUAL_STUDIO_2015_VERSION)


def IsMayaPython():
    """Determine whether we're running in Maya's version of Python. When
    building against Maya's Python, there are some additional restrictions
    on what we're able to build."""
    try:
        import maya

        return True
    except:
        pass

    return False


def GetPythonInfo():
    """Returns a tuple containing the path to the Python executable, shared
    library, and include directory corresponding to the version of Python
    currently running. Returns None if any path could not be determined.

    This function is used to extract build information from the Python
    interpreter used to launch this script. This information is used
    in the Boost and OSL builds. By taking this approach we can support
    having OSL builds for different Python versions built on the same
    machine. This is very useful, especially when developers have multiple
    versions installed on their machine, which is quite common now with
    Python2 and Python3 co-existing.
    """
    # First we extract the information that can be uniformly dealt with across
    # the platforms:
    pythonExecPath = sys.executable
    pythonVersion = sysconfig.get_config_var("py_version_short")  # "2.7"
    pythonVersionNoDot = sysconfig.get_config_var("py_version_nodot")  # "27"

    # Lib path is unfortunately special for each platform and there is no
    # config_var for it. But we can deduce it for each platform, and this
    # logic works for any Python version.
    def _GetPythonLibraryFilename():
        if Windows():
            return "python" + pythonVersionNoDot + ".lib"
        elif Linux():
            return sysconfig.get_config_var("LDLIBRARY")
        elif MacOS():
            return "libpython" + pythonVersion + ".dylib"
        else:
            raise RuntimeError("Platform not supported")

    # XXX: Handle the case where this script is being called using Maya's
    # Python since the sysconfig variables are set up differently in Maya.
    # Ideally we would not have any special Maya knowledge in here at all.
    if IsMayaPython():
        pythonBaseDir = sysconfig.get_config_var("base")

        # On Windows, the "base" path points to a "Python\" subdirectory
        # that contains the DLLs for site-package modules but not the
        # directories for the headers and .lib file we need -- those
        # are one level up.
        if Windows():
            pythonBaseDir = os.path.dirname(pythonBaseDir)

        pythonIncludeDir = os.path.join(
            pythonBaseDir, "include", "python" + pythonVersion
        )
        pythonLibPath = os.path.join(pythonBaseDir, "lib", _GetPythonLibraryFilename())
    else:
        pythonIncludeDir = sysconfig.get_config_var("INCLUDEPY")
        if Windows():
            pythonBaseDir = sysconfig.get_config_var("base")
            pythonLibPath = os.path.join(
                pythonBaseDir, "libs", _GetPythonLibraryFilename()
            )
        elif Linux():
            pythonLibDir = sysconfig.get_config_var("LIBDIR")
            pythonMultiarchSubdir = sysconfig.get_config_var("multiarchsubdir")
            if pythonMultiarchSubdir:
                pythonLibDir = pythonLibDir + pythonMultiarchSubdir
            pythonLibPath = os.path.join(pythonLibDir, _GetPythonLibraryFilename())
        elif MacOS():
            pythonBaseDir = sysconfig.get_config_var("base")
            pythonLibPath = os.path.join(
                pythonBaseDir, "lib", _GetPythonLibraryFilename()
            )
        else:
            raise RuntimeError("Platform not supported")

    return (pythonExecPath, pythonLibPath, pythonIncludeDir, pythonVersion)


def GetCPUCount():
    try:
        return multiprocessing.cpu_count()
    except NotImplementedError:
        return 1


def Run(cmd, logCommandOutput=True):
    """Run the specified command in a subprocess."""
    PrintInfo('Running "{cmd}"'.format(cmd=cmd))

    with codecs.open("log.txt", "a", "utf-8") as logfile:
        logfile.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        logfile.write("\n")
        logfile.write(cmd)
        logfile.write("\n")

        # Let exceptions escape from subprocess calls -- higher level
        # code will handle them.
        if logCommandOutput:
            p = subprocess.Popen(
                shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            while True:
                l = p.stdout.readline().decode(GetLocale(), "replace")
                if l:
                    logfile.write(l)
                    PrintCommandOutput(l)
                elif p.poll() is not None:
                    break
        else:
            p = subprocess.Popen(shlex.split(cmd))
            p.wait()

    if p.returncode != 0:
        # If verbosity >= 3, we'll have already been printing out command output
        # so no reason to print the log file again.
        if verbosity < 3:
            with open("log.txt", "r") as logfile:
                Print(logfile.read())
        raise RuntimeError(
            "Failed to run '{cmd}'\nSee {log} for more details.".format(
                cmd=cmd, log=os.path.abspath("log.txt")
            )
        )


@contextlib.contextmanager
def CurrentWorkingDirectory(dir):
    """Context manager that sets the current working directory to the given
    directory and resets it to the original directory when closed."""
    curdir = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(curdir)


def CopyFiles(context, src, dest):
    """Copy files like shutil.copy, but src may be a glob pattern."""
    filesToCopy = glob.glob(src)
    if not filesToCopy:
        raise RuntimeError("File(s) to copy {src} not found".format(src=src))

    instDestDir = os.path.join(context.instDir, dest)
    for f in filesToCopy:
        PrintCommandOutput(
            "Copying {file} to {destDir}\n".format(file=f, destDir=instDestDir)
        )
        shutil.copy(f, instDestDir)


def CopyDirectory(context, srcDir, destDir):
    """Copy directory like shutil.copytree."""
    instDestDir = os.path.join(context.instDir, destDir)
    if os.path.isdir(instDestDir):
        shutil.rmtree(instDestDir)

    PrintCommandOutput(
        "Copying {srcDir} to {destDir}\n".format(srcDir=srcDir, destDir=instDestDir)
    )
    shutil.copytree(srcDir, instDestDir)


def FormatMultiProcs(numJobs, generator):
    tag = "-j"
    if generator:
        if "Visual Studio" in generator:
            tag = "/M:"
        elif "Xcode" in generator:
            tag = "-j "

    return "{tag}{procs}".format(tag=tag, procs=numJobs)


def RunCMake(context, force, extraArgs=None, extraSrcDir="", extraInstDir=""):
    """Invoke CMake to configure, build, and install a library whose
    source code is located in the current working directory."""
    # Create a directory for out-of-source builds in the build directory
    # using the name of the current working directory.

    srcDir = os.getcwd()
    if extraSrcDir != "":
        srcDir = os.path.join(srcDir, extraSrcDir)
    instDir = context.oslInstDir if srcDir == context.oslSrcDir else context.instDir
    if extraInstDir != "":
        instDir = os.path.join(instDir, extraInstDir)
    buildDir = os.path.join(context.buildDir, os.path.split(srcDir)[1])
    if force and os.path.isdir(buildDir):
        shutil.rmtree(buildDir)

    if not os.path.isdir(buildDir):
        os.makedirs(buildDir)

    generator = context.cmakeGenerator

    # On Windows, we need to explicitly specify the generator to ensure we're
    # building a 64-bit project. (Surely there is a better way to do this?)
    # TODO: figure out exactly what "vcvarsall.bat x64" sets to force x64

    if generator is None and Windows():
        if IsVisualStudio2019OrGreater():
            generator = "Visual Studio 16 2019"
        elif IsVisualStudio2017OrGreater():
            generator = "Visual Studio 15 2017 Win64"
        else:
            generator = "Visual Studio 14 2015 Win64"

    if generator is not None:
        generator = '-G "{gen}"'.format(gen=generator)

    if generator != '-G "NMake Makefiles"':
        if IsVisualStudio2019OrGreater():
            generator = generator + " -A x64"

    toolset = context.cmakeToolset
    if toolset is not None:
        toolset = '-T "{toolset}"'.format(toolset=toolset)

    # On MacOS, enable the use of @rpath for relocatable builds.
    osx_rpath = None
    if MacOS():
        osx_rpath = "-DCMAKE_MACOSX_RPATH=ON"

    # We use -DCMAKE_BUILD_TYPE for single-configuration generators
    # (Ninja, make), and --config for multi-configuration generators
    # (Visual Studio); technically we don't need BOTH at the same
    # time, but specifying both is simpler than branching
    config = "Debug" if context.buildDebug else "Release"

    with CurrentWorkingDirectory(buildDir):
        Run(
            "cmake "
            '-DCMAKE_INSTALL_PREFIX="{instDir}" '
            '-DCMAKE_PREFIX_PATH="{depsInstDir}" '
            "-DCMAKE_BUILD_TYPE={config} "
            "{osx_rpath} "
            "{generator} "
            "{toolset} "
            "{extraArgs} "
            '"{srcDir}"'.format(
                instDir=instDir,
                depsInstDir=context.instDir,
                config=config,
                srcDir=srcDir,
                osx_rpath=(osx_rpath or ""),
                generator=(generator or ""),
                toolset=(toolset or ""),
                extraArgs=(" ".join(extraArgs) if extraArgs else ""),
            )
        )
        if generator == '-G "NMake Makefiles"':
            Run("nmake")
            # Run("nmake /I /K") # for ignoring exit with code 0 error
            Run("cmake --install .")
        else:
            Run(
                "cmake --build . --config {config} --target install -- {multiproc}".format(
                    config=config,
                    multiproc=FormatMultiProcs(context.numJobs, generator),
                )
            )


def GetCMakeVersion():
    """
    Returns the CMake version as tuple of integers (major, minor) or
    (major, minor, patch) or None if an error occured while launching cmake and
    parsing its output.
    """

    output_string = GetCommandOutput("cmake --version")
    if not output_string:
        PrintWarning(
            "Could not determine cmake version -- please install it "
            "and adjust your PATH"
        )
        return None

    # cmake reports, e.g., "... version 3.14.3"
    match = re.search(r"version (\d+)\.(\d+)(\.(\d+))?", output_string)
    if not match:
        PrintWarning("Could not determine cmake version")
        return None

    major, minor, patch_group, patch = match.groups()
    if patch_group is None:
        return (int(major), int(minor))
    else:
        return (int(major), int(minor), int(patch))


def PatchFile(filename, patches, multiLineMatches=False):
    """Applies patches to the specified file. patches is a list of tuples
    (old string, new string)."""
    if multiLineMatches:
        oldLines = [open(filename, "r").read()]
    else:
        oldLines = open(filename, "r").readlines()
    newLines = oldLines
    for (oldString, newString) in patches:
        newLines = [s.replace(oldString, newString) for s in newLines]
    if newLines != oldLines:
        PrintInfo(
            "Patching file {filename} (original in {oldFilename})...".format(
                filename=filename, oldFilename=filename + ".old"
            )
        )
        shutil.copy(filename, filename + ".old")
        open(filename, "w").writelines(newLines)


def DownloadFileWithCurl(url, outputFilename):
    # Don't log command output so that curl's progress
    # meter doesn't get written to the log file.
    Run(
        "curl {progress} -L -o {filename} {url}".format(
            progress="-#" if verbosity >= 2 else "-s", filename=outputFilename, url=url
        ),
        logCommandOutput=False,
    )


def DownloadFileWithPowershell(url, outputFilename):
    # It's important that we specify to use TLS v1.2 at least or some
    # of the downloads will fail.
    cmd = "powershell [Net.ServicePointManager]::SecurityProtocol = \
            [Net.SecurityProtocolType]::Tls12; \"(new-object \
            System.Net.WebClient).DownloadFile('{url}', '{filename}')\"".format(
        filename=outputFilename, url=url
    )

    Run(cmd, logCommandOutput=False)


def DownloadFileWithUrllib(url, outputFilename):
    r = urlopen(url)
    with open(outputFilename, "wb") as outfile:
        outfile.write(r.read())


def DownloadURL(url, context, force, dontExtract=None):
    """Download and extract the archive file at given URL to the
    source directory specified in the context.

    dontExtract may be a sequence of path prefixes that will
    be excluded when extracting the archive.

    Returns the absolute path to the directory where files have
    been extracted."""
    with CurrentWorkingDirectory(context.srcDir):
        # Extract filename from URL and see if file already exists.
        filename = url.split("/")[-1]
        if force and os.path.exists(filename):
            os.remove(filename)

        if os.path.exists(filename):
            PrintInfo(
                "{0} already exists, skipping download".format(
                    os.path.abspath(filename)
                )
            )
        else:
            PrintInfo("Downloading {0} to {1}".format(url, os.path.abspath(filename)))

            # To work around occasional hiccups with downloading from websites
            # (SSL validation errors, etc.), retry a few times if we don't
            # succeed in downloading the file.
            maxRetries = 5
            lastError = None

            # Download to a temporary file and rename it to the expected
            # filename when complete. This ensures that incomplete downloads
            # will be retried if the script is run again.
            tmpFilename = filename + ".tmp"
            if os.path.exists(tmpFilename):
                os.remove(tmpFilename)

            for i in range(maxRetries):
                try:
                    context.downloader(url, tmpFilename)
                    break
                except Exception as e:
                    PrintCommandOutput(
                        "Retrying download due to error: {err}\n".format(err=e)
                    )
                    lastError = e
            else:
                errorMsg = str(lastError)
                if "SSL: TLSV1_ALERT_PROTOCOL_VERSION" in errorMsg:
                    errorMsg += (
                        "\n\n"
                        "Your OS or version of Python may not support "
                        "TLS v1.2+, which is required for downloading "
                        "files from certain websites. This support "
                        "was added in Python 2.7.9."
                        "\n\n"
                        "You can use curl to download dependencies "
                        "by installing it in your PATH and re-running "
                        "this script."
                    )
                raise RuntimeError(
                    "Failed to download {url}: {err}".format(url=url, err=errorMsg)
                )

            shutil.move(tmpFilename, filename)

        # Open the archive and retrieve the name of the top-most directory.
        # This assumes the archive contains a single directory with all
        # of the contents beneath it.
        archive = None
        rootDir = None
        members = None
        try:
            if tarfile.is_tarfile(filename):
                archive = tarfile.open(filename)
                rootDir = archive.getnames()[0].split("/")[0]
                if dontExtract != None:
                    members = (
                        m
                        for m in archive.getmembers()
                        if not any((fnmatch.fnmatch(m.name, p) for p in dontExtract))
                    )
            elif zipfile.is_zipfile(filename):
                archive = zipfile.ZipFile(filename)
                rootDir = archive.namelist()[0].split("/")[0]
                if dontExtract != None:
                    members = (
                        m
                        for m in archive.getnames()
                        if not any((fnmatch.fnmatch(m, p) for p in dontExtract))
                    )
            else:
                raise RuntimeError("unrecognized archive file type")

            with archive:
                extractedPath = os.path.abspath(rootDir)
                if force and os.path.isdir(extractedPath):
                    shutil.rmtree(extractedPath)

                if os.path.isdir(extractedPath):
                    PrintInfo(
                        "Directory {0} already exists, skipping extract".format(
                            extractedPath
                        )
                    )
                else:
                    PrintInfo("Extracting archive to {0}".format(extractedPath))

                    # Extract to a temporary directory then move the contents
                    # to the expected location when complete. This ensures that
                    # incomplete extracts will be retried if the script is run
                    # again.
                    tmpExtractedPath = os.path.abspath("extract_dir")
                    if os.path.isdir(tmpExtractedPath):
                        shutil.rmtree(tmpExtractedPath)

                    archive.extractall(tmpExtractedPath, members=members)

                    shutil.move(os.path.join(tmpExtractedPath, rootDir), extractedPath)
                    shutil.rmtree(tmpExtractedPath)

                return extractedPath
        except Exception as e:
            # If extraction failed for whatever reason, assume the
            # archive file was bad and move it aside so that re-running
            # the script will try downloading and extracting again.
            shutil.move(filename, filename + ".bad")
            raise RuntimeError(
                "Failed to extract archive {filename}: {err}".format(
                    filename=filename, err=e
                )
            )


############################################################
# 3rd-Party Dependencies

AllDependencies = list()
AllDependenciesByName = dict()


class Dependency(object):
    def __init__(self, name, installer, *files):
        self.name = name
        self.installer = installer
        self.filesToCheck = files

        AllDependencies.append(self)
        AllDependenciesByName.setdefault(name.lower(), self)

    def Exists(self, context):
        return all(
            [
                os.path.isfile(os.path.join(context.instDir, f))
                for f in self.filesToCheck
            ]
        )


class PythonDependency(object):
    def __init__(self, name, getInstructions, moduleNames):
        self.name = name
        self.getInstructions = getInstructions
        self.moduleNames = moduleNames

    def Exists(self, context):
        # If one of the modules in our list imports successfully, we are good.
        for moduleName in self.moduleNames:
            try:
                pyModule = __import__(moduleName)
                return True
            except:
                pass

        return False


def AnyPythonDependencies(deps):
    return any([type(d) is PythonDependency for d in deps])


############################################################
# zlib

ZLIB_URL = "https://github.com/madler/zlib/archive/v1.2.11.zip"


def InstallZlib(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(ZLIB_URL, context, force)):
        RunCMake(context, force, buildArgs)


ZLIB = Dependency("zlib", InstallZlib, "include/zlib.h")

############################################################
# boost

if Linux() or MacOS():
    if Python3():
        BOOST_URL = "https://downloads.sourceforge.net/project/boost/boost/1.70.0/boost_1_70_0.tar.gz"
    else:
        BOOST_URL = "https://downloads.sourceforge.net/project/boost/boost/1.61.0/boost_1_61_0.tar.gz"
    BOOST_VERSION_FILE = "include/boost/version.hpp"
elif Windows():
    # The default installation of boost on Windows puts headers in a versioned
    # subdirectory, which we have to account for here. In theory, specifying
    # "layout=system" would make the Windows install match Linux/MacOS, but that
    # causes problems for other dependencies that look for boost.
    #
    # boost 1.70 is required for Visual Studio 2019. For simplicity, we use
    # this version for all older Visual Studio versions as well.
    BOOST_URL = "https://downloads.sourceforge.net/project/boost/boost/1.70.0/boost_1_70_0.tar.gz"
    BOOST_VERSION_FILE = "include/boost-1_70/boost/version.hpp"


def InstallBoost_Helper(context, force, buildArgs):
    # Documentation files in the boost archive can have exceptionally
    # long paths. This can lead to errors when extracting boost on Windows,
    # since paths are limited to 260 characters by default on that platform.
    # To avoid this, we skip extracting all documentation.
    #
    # For some examples, see: https://svn.boost.org/trac10/ticket/11677
    dontExtract = ["*/doc/*", "*/libs/*/doc/*"]

    with CurrentWorkingDirectory(DownloadURL(BOOST_URL, context, force, dontExtract)):
        bootstrap = "bootstrap.bat" if Windows() else "./bootstrap.sh"
        Run(
            '{bootstrap} --prefix="{instDir}"'.format(
                bootstrap=bootstrap, instDir=context.instDir
            )
        )

        # b2 supports at most -j64 and will error if given a higher value.
        num_procs = min(64, context.numJobs)

        b2_settings = [
            '--prefix="{instDir}"'.format(instDir=context.instDir),
            '--build-dir="{buildDir}"'.format(buildDir=context.buildDir),
            "-j{procs}".format(procs=num_procs),
            "address-model=64",
            "link=shared",
            "runtime-link=shared",
            "threading=multi",
            "variant={variant}".format(
                variant="debug" if context.buildDebug else "release"
            ),
            "--with-atomic",
            "--with-program_options",
            "--with-regex",
        ]

        if context.buildPython:
            b2_settings.append("--with-python")
            pythonInfo = GetPythonInfo()
            if Windows():
                # Unfortunately Boost build scripts require the Python folder
                # that contains the executable on Windows
                pythonPath = os.path.dirname(pythonInfo[0])
            else:
                # While other platforms want the complete executable path
                pythonPath = pythonInfo[0]
            # This is the only platform-independent way to configure these
            # settings correctly and robustly for the Boost jam build system.
            # There are Python config arguments that can be passed to bootstrap
            # but those are not available in boostrap.bat (Windows) so we must
            # take the following approach:
            projectPath = "python-config.jam"
            with open(projectPath, "w") as projectFile:
                # Note that we must escape any special characters, like
                # backslashes for jam, hence the mods below for the path
                # arguments. Also, if the path contains spaces jam will not
                # handle them well. Surround the path parameters in quotes.
                line = 'using python : %s : "%s" : "%s" ;\n' % (
                    pythonInfo[3],
                    pythonPath.replace("\\", "\\\\"),
                    pythonInfo[2].replace("\\", "\\\\"),
                )
                projectFile.write(line)
            b2_settings.append("--user-config=python-config.jam")

        if context.buildOIIO:
            b2_settings.append("--with-date_time")

        if context.buildOIIO:
            b2_settings.append("--with-system")
            b2_settings.append("--with-thread")
        # if Linux():
        #     b2_settings.append("toolset=gcc")

        if context.buildOPENVDB:
            b2_settings.append("--with-iostreams")

            # b2 with -sNO_COMPRESSION=1 fails with the following error message:
            #     error: at [...]/boost_1_61_0/tools/build/src/kernel/modules.jam:107
            #     error: Unable to find file or target named
            #     error:     '/zlib//zlib'
            #     error: referred to from project at
            #     error:     'libs/iostreams/build'
            #     error: could not resolve project reference '/zlib'

            # But to avoid an extra library dependency, we can still explicitly
            # exclude the bzip2 compression from boost_iostreams (note that
            # OpenVDB uses blosc compression).
            b2_settings.append("-sNO_BZIP2=1")

        if context.buildOIIO:
            b2_settings.append("--with-filesystem")

        if force:
            b2_settings.append("-a")

        if Windows():
            # toolset parameter for Visual Studio documented here:
            # https://github.com/boostorg/build/blob/develop/src/tools/msvc.jam
            if context.cmakeToolset == "v142":
                b2_settings.append("toolset=msvc-14.2")
            elif context.cmakeToolset == "v141":
                b2_settings.append("toolset=msvc-14.1")
            elif context.cmakeToolset == "v140":
                b2_settings.append("toolset=msvc-14.0")
            elif IsVisualStudio2019OrGreater():
                b2_settings.append("toolset=msvc-14.2")
            elif IsVisualStudio2017OrGreater():
                b2_settings.append("toolset=msvc-14.1")
            else:
                b2_settings.append("toolset=msvc-14.0")

        if MacOS():
            # Must specify toolset=clang to ensure install_name for boost
            # libraries includes @rpath
            b2_settings.append("toolset=clang")

        # Add on any user-specified extra arguments.
        b2_settings += buildArgs

        b2 = "b2" if Windows() else "./b2"
        Run("{b2} {options} install".format(b2=b2, options=" ".join(b2_settings)))


def InstallBoost(context, force, buildArgs):
    # Boost's build system will install the version.hpp header before
    # building its libraries. We make sure to remove it in case of
    # any failure to ensure that the build script detects boost as a
    # dependency to build the next time it's run.
    try:
        InstallBoost_Helper(context, force, buildArgs)
    except:
        versionHeader = os.path.join(context.instDir, BOOST_VERSION_FILE)
        if os.path.isfile(versionHeader):
            try:
                os.remove(versionHeader)
            except:
                pass
        raise


BOOST = Dependency("boost", InstallBoost, BOOST_VERSION_FILE)

############################################################
# JPEGTurbo

JPEGTurbo_URL = "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.zip"
if Windows():
    JPEGTurbo_URL = "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.5.zip"


def InstallJPEGTurbo(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(JPEGTurbo_URL, context, force)):
        RunCMake(context, force, buildArgs)


JPEGTURBO = Dependency("JPEGTurbo", InstallJPEGTurbo, "include/jpeglib.h")

############################################################
# TIFF

TIFF_URL = "http://download.osgeo.org/libtiff/tiff-4.1.0.zip"


def InstallTIFF(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(TIFF_URL, context, force)):
        # libTIFF has a build issue on Windows where tools/tiffgt.c
        # unconditionally includes unistd.h, which does not exist.
        # To avoid this, we patch the CMakeLists.txt to skip building
        # the tools entirely. We do this on Linux and MacOS as well
        # to avoid requiring some GL and X dependencies.
        #
        # We also need to skip building tests, since they rely on
        # the tools we've just elided.
        PatchFile(
            "CMakeLists.txt",
            [
                ("add_subdirectory(tools)", "# add_subdirectory(tools)"),
                ("add_subdirectory(test)", "# add_subdirectory(test)"),
            ],
        )

        # The libTIFF CMakeScript says the ld-version-script
        # functionality is only for compilers using GNU ld on
        # ELF systems or systems which provide an emulation; therefore
        # skipping it completely on mac and windows.
        if MacOS() or Windows():
            extraArgs = ["-Dld-version-script=OFF"]
        else:
            extraArgs = []
        extraArgs += buildArgs
        RunCMake(context, force, extraArgs)


TIFF = Dependency("TIFF", InstallTIFF, "include/tiff.h")

############################################################
# PNG

# PNG_URL = "https://downloads.sourceforge.net/project/libpng/libpng16/older-releases/1.6.29/libpng-1.6.29.tar.gz"
PNG_URL = "https://github.com/glennrp/libpng/archive/v1.6.35.zip"


def InstallPNG(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(PNG_URL, context, force)):
        RunCMake(context, force, buildArgs)


PNG = Dependency("PNG", InstallPNG, "include/png.h")

############################################################
# IlmBase/OpenEXR

# OPENEXR_URL = "https://github.com/openexr/openexr/archive/v2.2.0.zip"
OPENEXR_URL = "https://github.com/AcademySoftwareFoundation/openexr/archive/v2.5.3.zip"


def InstallOpenEXR(context, force, buildArgs):
    srcDir = DownloadURL(OPENEXR_URL, context, force)

    ilmbaseSrcDir = os.path.join(srcDir, "IlmBase")
    with CurrentWorkingDirectory(ilmbaseSrcDir):
        # openexr 2.2 has a bug with Ninja:
        # https://github.com/openexr/openexr/issues/94
        # https://github.com/openexr/openexr/pull/142
        # Fix commit here:
        # https://github.com/openexr/openexr/commit/8eed7012c10f1a835385d750fd55f228d1d35df9
        # Merged here:
        # https://github.com/openexr/openexr/commit/b206a243a03724650b04efcdf863c7761d5d5d5b
        if context.cmakeGenerator == "Ninja":
            PatchFile(
                os.path.join("Half", "CMakeLists.txt"),
                [
                    (
                        "TARGET eLut POST_BUILD",
                        "OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/eLut.h",
                    ),
                    (
                        "  COMMAND eLut > ${CMAKE_CURRENT_BINARY_DIR}/eLut.h",
                        "  COMMAND eLut ARGS > ${CMAKE_CURRENT_BINARY_DIR}/eLut.h\n"
                        "  DEPENDS eLut",
                    ),
                    (
                        "TARGET toFloat POST_BUILD",
                        "OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/toFloat.h",
                    ),
                    (
                        "  COMMAND toFloat > ${CMAKE_CURRENT_BINARY_DIR}/toFloat.h",
                        "  COMMAND toFloat ARGS > ${CMAKE_CURRENT_BINARY_DIR}/toFloat.h\n"
                        "  DEPENDS toFloat",
                    ),
                    (
                        "  ${CMAKE_CURRENT_BINARY_DIR}/eLut.h\n"
                        "  OBJECT_DEPENDS\n"
                        "  ${CMAKE_CURRENT_BINARY_DIR}/toFloat.h\n",
                        '  "${CMAKE_CURRENT_BINARY_DIR}/eLut.h;${CMAKE_CURRENT_BINARY_DIR}/toFloat.h"\n',
                    ),
                ],
                multiLineMatches=True,
            )
        RunCMake(context, force, buildArgs)

    openexrSrcDir = os.path.join(srcDir, "OpenEXR")
    with CurrentWorkingDirectory(openexrSrcDir):
        RunCMake(
            context,
            force,
            ['-DILMBASE_PACKAGE_PREFIX="{instDir}"'.format(instDir=context.instDir)]
            + buildArgs,
        )


OPENEXR = Dependency("OpenEXR", InstallOpenEXR, "include/OpenEXR/ImfVersion.h")

############################################################
# Ptex

PTEX_URL = "https://github.com/wdas/ptex/archive/v2.1.28.zip"


def InstallPtex(context, force, buildArgs):
    if Windows():
        InstallPtex_Windows(context, force, buildArgs)
    else:
        InstallPtex_LinuxOrMacOS(context, force, buildArgs)


def InstallPtex_Windows(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(PTEX_URL, context, force)):
        # Ptex has a bug where the import library for the dynamic library and
        # the static library both get the same name, Ptex.lib, and as a
        # result one clobbers the other. We hack the appropriate CMake
        # file to prevent that. Since we don't need the static library we'll
        # rename that.
        #
        # In addition src\tests\CMakeLists.txt adds -DPTEX_STATIC to the
        # compiler but links tests against the dynamic library, causing the
        # links to fail. We patch the file to not add the -DPTEX_STATIC
        PatchFile(
            "src\\ptex\\CMakeLists.txt",
            [
                (
                    "set_target_properties(Ptex_static PROPERTIES OUTPUT_NAME Ptex)",
                    "set_target_properties(Ptex_static PROPERTIES OUTPUT_NAME Ptexs)",
                )
            ],
        )
        PatchFile(
            "src\\tests\\CMakeLists.txt",
            [("add_definitions(-DPTEX_STATIC)", "# add_definitions(-DPTEX_STATIC)")],
        )

        # Patch Ptex::String to export symbol for operator<<
        # This is required for newer versions of OIIO, which make use of the
        # this operator on Windows platform specifically.
        PatchFile(
            "src\\ptex\\Ptexture.h",
            [
                (
                    "std::ostream& operator << (std::ostream& stream, const Ptex::String& str);",
                    "PTEXAPI std::ostream& operator << (std::ostream& stream, const Ptex::String& str);",
                )
            ],
        )

        RunCMake(context, force, buildArgs)


def InstallPtex_LinuxOrMacOS(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(PTEX_URL, context, force)):
        RunCMake(context, force, buildArgs)


PTEX = Dependency("Ptex", InstallPtex, "include/PtexVersion.h")

############################################################
# GLUT (for Partio)

GLUT_URL = "http://prdownloads.sourceforge.net/freeglut/freeglut-3.2.1.tar.gz"


def InstallGLUT(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(GLUT_URL, context, force)):
        RunCMake(context, force, buildArgs)


GLUT = Dependency("GLUT", InstallGLUT, "include/GL/freeglut.h")

############################################################
# Partio (for OSL)

# Partio_URL = "https://github.com/wdas/partio/archive/v1.13.0.zip"
Partio_URL = "https://github.com/wdas/partio/archive/b1163a94261cb43d05966c5075edcdacfe4af52d.zip"


def InstallPartio(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(Partio_URL, context, force)):
        extraArgs = []
        extraArgs.append(
            '-DGLUT_INCLUDE_DIR="{instDir}/include"'.format(instDir=context.instDir)
        )
        extraArgs.append(
            '-DGLUT_glut_LIBRARY="{instDir}/lib"'.format(instDir=context.instDir)
        )
        extraArgs += buildArgs
        RunCMake(context, force, extraArgs)


PARTIO = Dependency("Partio", InstallPartio, "include/Partio.h")

############################################################
# PugiXML (for OSL)

PugiXML_URL = "https://github.com/zeux/pugixml/releases/download/v1.10/pugixml-1.10.zip"


def InstallPugiXML(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(PugiXML_URL, context, force)):
        extraArgs = []

        if Linux():
            extraArgs.append('-DCMAKE_CXX_FLAGS="-fPIC"')
        extraArgs += buildArgs
        RunCMake(context, force, extraArgs)


PUGIXML = Dependency("PugiXML", InstallPugiXML, "include/pugixml.hpp")

############################################################
# LLVM (for OSL)

LLVM_URL = "https://github.com/llvm/llvm-project/archive/llvmorg-11.0.0.zip"
# LLVM_URL = "https://github.com/llvm/llvm-project/releases/tag/llvmorg-10.0.1"
# LLVM_URL = "https://github.com/llvm/llvm-project/archive/llvmorg-8.0.1.zip"
# LLVM_URL = "https://github.com/llvm/llvm-project/archive/llvmorg-9.0.1.zip"
# LLVM_URL = "https://github.com/llvm/llvm-project/archive/llvmorg-7.1.0.zip"
# LLVM_URL = "https://github.com/llvm/llvm-project/archive/llvmorg-7.0.1.zip"


def InstallLLVM(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(LLVM_URL, context, force)):
        extraArgs = []
        if Windows():
            extraArgs.append("-Thost=x64")
        extraArgs += buildArgs
        RunCMake(context, force, extraArgs, extraSrcDir="llvm")


LLVM = Dependency("LLVM", InstallLLVM, "include/llvm/LTO/LTO.h")

############################################################
# CLANG (for OSL)

# URL is same as llvm url


def InstallCLANG(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(LLVM_URL, context, force)):
        RunCMake(context, force, buildArgs, extraSrcDir="clang")


CLANG = Dependency("CLANG", InstallCLANG, "include/clang/Basic/Version.h")

############################################################
# WinFlexBison (for OSL)

WinFlexBison_URL = "https://github.com/lexxmark/winflexbison/archive/v2.5.22.zip"


def InstallWinFlexBison(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(WinFlexBison_URL, context, force)):
        RunCMake(context, force, buildArgs, extraInstDir="bin")


WINFLEXBISON = Dependency("WinFlexBison", InstallWinFlexBison, "bin/win_flex.exe")

############################################################
# PyBind11 (for OSL)

PyBind11_URL = "https://github.com/pybind/pybind11/archive/v2.6.0.zip"


def InstallPyBind11(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(PyBind11_URL, context, force)):
        RunCMake(context, force, buildArgs)


PYBIND11 = Dependency("PyBind11", InstallPyBind11, "include/pybind11/pybind11.h")
############################################################
# OSL


def InstallOSL(context, force, buildArgs):
    with CurrentWorkingDirectory(context.oslSrcDir):
        extraArgs = []

        if context.buildPython:
            if Python3():
                extraArgs.append("-DUSE_PYTHON=0")

        extraArgs.append("-DOSL_BUILD_TESTS=0")
        # if you are using LLVM 10 or higher C++ should be set on 14
        extraArgs.append("-DCMAKE_CXX_STANDARD=14")

        # if you are using LLVM 10 or higher C++ should be set on 11
        # extraArgs.append("-DCMAKE_CXX_STANDARD=11")

        # if you used windows installer for llvm you should add the path like this
        delimeter = ":"
        if Windows():
            delimeter = ";"
        if Windows():
            extraArgs.append('-DLLVM_ROOT="{instDir}"'.format(instDir=context.instDir))

        # if Linux():
        #     extraArgs.append(
        #         '-DLLVM_ROOT="/opt/rh/llvm-toolset-7.0/root/usr"'
        #     )  # does not work, should set an env var befor running this py file
        #     context.instDir += delimeter + "/opt/rh/llvm-toolset-7.0/root/usr"

        # if Linux():
        #     extraArgs.append('-DCMAKE_CXX_FLAGS="-fPIC"')

        if Windows():
            # for now have to use manual boost build
            extraArgs.append("-DBoost_NO_BOOST_CMAKE=On")
            extraArgs.append("-DBoost_NO_SYSTEM_PATHS=True")

            # Increase the precompiled header buffer limit.
            extraArgs.append('-DCMAKE_CXX_FLAGS="/Zm150"')
            if context.buildDebug:
                extraArgs.append('-DCMAKE_CXX_FLAGS="/NODEFAULTLIB /MD /EHsc"')
                # extraArgs.append('-DCMAKE_PREFIX_PATH="{instDir}";"{buildDir}/OpenShadingLanguage/bin"'.format(instDir=context.instDir,buildDir=context.buildDir))

            extraArgs.append("-DENABLE_PRECOMPILED_HEADERS=OFF")
            extraArgs.append("-DUSE_Package=OFF")

        extraArgs += buildArgs

        # there is an error related to make or gmake in linux
        # Run("gmake --version") # debug
        # context.cmakeGenerator = "NMake Makefiles"
        RunCMake(context, force, extraArgs)


OSL = Dependency("OSL", InstallOSL, "include/OSL/oslversion.h")

# ############################################################
# LibRaw (for OpenImageIO)

# LibRaw_URL = "https://www.libraw.org/data/LibRaw-0.20.2-Win64.zip"
LibRaw_URL = "https://www.libraw.org/data/LibRaw-0.20.2.zip"

if Linux():
    LibRaw_URL = "https://www.libraw.org/data/LibRaw-0.20.2.tar.gz"


def InstallLibRaw(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(LibRaw_URL, context, force)):
        if Linux():
            # print("--=", os.getcwd())
            # Run("autoreconf --install")
            Run('./configure --prefix="{instDir}"'.format(instDir=context.instDir))
            # Run("automake --version > {0}/output.log".format(instDir))
            Run("make")
            Run("make install")
        if Windows():
            Run("nmake Makefile.msvc")
            # Run("mkdir {instDir}\\bin".format(instDir=context.instDir))
            # Run("mkdir {instDir}\\include\\libraw".format(instDir=context.instDir))
            # Run("mkdir {instDir}\\lib".format(instDir=context.instDir))
            Run('xcopy /E /I /Y bin "{instDir}\\bin"'.format(instDir=context.instDir))
            Run(
                'xcopy /E /I /Y libraw "{instDir}\\include\\libraw"'.format(
                    instDir=context.instDir
                )
            )
            Run('xcopy /E /I /Y lib "{instDir}\\lib"'.format(instDir=context.instDir))


# if Windows():
LIBRAW = Dependency("LibRaw", InstallLibRaw, "include/libraw/libraw.h")

############################################################
# OpenVDB

# Using version 6.1.0 since it has reworked its CMake files so that
# there are better options to not compile the OpenVDB binaries and to
# not require additional dependencies such as GLFW. Note that version
# 6.1.0 does require CMake 3.3 though.

OPENVDB_URL = "https://github.com/AcademySoftwareFoundation/openvdb/archive/v6.1.0.zip"


def InstallOpenVDB(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(OPENVDB_URL, context, force)):
        extraArgs = [
            "-DOPENVDB_BUILD_PYTHON_MODULE=OFF",
            "-DOPENVDB_BUILD_BINARIES=OFF",
            "-DOPENVDB_BUILD_UNITTESTS=OFF",
        ]

        # Make sure to use boost installed by the build script and not any
        # system installed boost
        extraArgs.append("-DBoost_NO_BOOST_CMAKE=On")
        extraArgs.append("-DBoost_NO_SYSTEM_PATHS=True")

        extraArgs.append('-DBLOSC_ROOT="{instDir}"'.format(instDir=context.instDir))
        extraArgs.append('-DTBB_ROOT="{instDir}"'.format(instDir=context.instDir))
        # OpenVDB needs Half type from IlmBase
        extraArgs.append('-DILMBASE_ROOT="{instDir}"'.format(instDir=context.instDir))

        RunCMake(context, force, extraArgs)


OPENVDB = Dependency("OpenVDB", InstallOpenVDB, "include/openvdb/openvdb.h")

############################################################
# OpenImageIO

# OIIO_URL = "https://github.com/OpenImageIO/oiio/archive/Release-2.2.7.0.zip"
OIIO_URL = "https://github.com/OpenImageIO/oiio/archive/Release-2.1.20.0.zip"


def InstallOpenImageIO(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(OIIO_URL, context, force)):
        extraArgs = [
            "-DOIIO_BUILD_TOOLS=OFF",
            "-DOIIO_BUILD_TESTS=OFF",
            "-DUSE_PYTHON=OFF",
            "-DSTOP_ON_WARNING=OFF",
        ]

        # OIIO's FindOpenEXR module circumvents CMake's normal library
        # search order, which causes versions of OpenEXR installed in
        # /usr/local or other hard-coded locations in the module to
        # take precedence over the version we've built, which would
        # normally be picked up when we specify CMAKE_PREFIX_PATH.
        # This may lead to undefined symbol errors at build or runtime.
        # So, we explicitly specify the OpenEXR we want to use here.
        extraArgs.append('-DOPENEXR_HOME="{instDir}"'.format(instDir=context.instDir))

        # If Ptex support is disabled in OSL, disable support in OpenImageIO
        # as well. This ensures OIIO doesn't accidentally pick up a Ptex
        # library outside of our build.
        if not context.buildPTEX:
            extraArgs.append("-DUSE_PTEX=OFF")

        # Make sure to use boost installed by the build script and not any
        # system installed boost
        if Windows():
            extraArgs.append("-DBoost_NO_BOOST_CMAKE=On")
            extraArgs.append("-DBoost_NO_SYSTEM_PATHS=True")

        # Add on any user-specified extra arguments.
        extraArgs += buildArgs

        RunCMake(context, force, extraArgs)


OPENIMAGEIO = Dependency(
    "OpenImageIO", InstallOpenImageIO, "include/OpenImageIO/oiioversion.h"
)

############################################################
# OpenColorIO

# Use v1.1.0 on MacOS and Windows since v1.0.9 doesn't build properly on
# those platforms.
if Linux():
    OCIO_URL = "https://github.com/imageworks/OpenColorIO/archive/v1.0.9.zip"
else:
    OCIO_URL = "https://github.com/imageworks/OpenColorIO/archive/v1.1.0.zip"


def InstallOpenColorIO(context, force, buildArgs):
    with CurrentWorkingDirectory(DownloadURL(OCIO_URL, context, force)):
        extraArgs = [
            "-DOCIO_BUILD_TRUELIGHT=OFF",
            "-DOCIO_BUILD_APPS=OFF",
            "-DOCIO_BUILD_NUKE=OFF",
            "-DOCIO_BUILD_DOCS=OFF",
            "-DOCIO_BUILD_TESTS=OFF",
            "-DOCIO_BUILD_PYGLUE=OFF",
            "-DOCIO_BUILD_JNIGLUE=OFF",
            "-DOCIO_STATIC_JNIGLUE=OFF",
        ]

        # The OCIO build treats all warnings as errors but several come up
        # on various platforms, including:
        # - On gcc6, v1.1.0 emits many -Wdeprecated-declaration warnings for
        #   std::auto_ptr
        # - On clang, v1.1.0 emits a -Wself-assign-field warning. This is fixed
        #   in https://github.com/AcademySoftwareFoundation/OpenColorIO/commit/0be465feb9ac2d34bd8171f30909b276c1efa996
        #
        # To avoid build failures we force all warnings off for this build.
        if GetVisualStudioCompilerAndVersion():
            # This doesn't work because CMake stores default flags for
            # MSVC in CMAKE_CXX_FLAGS and this would overwrite them.
            # However, we don't seem to get any warnings on Windows
            # (at least with VS2015 and 2017).
            # extraArgs.append('-DCMAKE_CXX_FLAGS=/w')
            pass
        else:
            extraArgs.append("-DCMAKE_CXX_FLAGS=-w")

        # Add on any user-specified extra arguments.
        extraArgs += buildArgs

        RunCMake(context, force, extraArgs)


OPENCOLORIO = Dependency(
    "OpenColorIO", InstallOpenColorIO, "include/OpenColorIO/OpenColorABI.h"
)

############################################################
# Install script

programDescription = """\
Installation Script for OSL

Builds and installs OSL and 3rd-party dependencies to specified location.

- Libraries:
The following is a list of libraries that this script will download and build
as needed. These names can be used to identify libraries for various script
options, like --force or --build-args.

{libraryList}

- Downloading Libraries:
If curl or powershell (on Windows) are installed and located in PATH, they
will be used to download dependencies. Otherwise, a built-in downloader will 
be used.

- Specifying Custom Build Arguments:
Users may specify custom build arguments for libraries using the --build-args
option. This values for this option must take the form <library name>,<option>. 
For example:

%(prog)s --build-args boost,cxxflags=... OSL,-DCMAKE_CXX_STANDARD=14 ...

These arguments will be passed directly to the build system for the specified 
library. Multiple quotes may be needed to ensure arguments are passed on 
exactly as desired. Users must ensure these arguments are suitable for the
specified library and do not conflict with other options, otherwise build 
errors may occur.

""".format(
    libraryList=" ".join(sorted([d.name for d in AllDependencies]))
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter, description=programDescription
)

parser.add_argument(
    "install_dir", type=str, help="Directory where OSL will be installed"
)
parser.add_argument(
    "-n",
    "--dry_run",
    dest="dry_run",
    action="store_true",
    help="Only summarize what would happen",
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "-v",
    "--verbose",
    action="count",
    default=1,
    dest="verbosity",
    help="Increase verbosity level (1-3)",
)
group.add_argument(
    "-q",
    "--quiet",
    action="store_const",
    const=0,
    dest="verbosity",
    help="Suppress all output except for error messages",
)

group = parser.add_argument_group(title="Build Options")
group.add_argument(
    "-j",
    "--jobs",
    type=int,
    default=GetCPUCount(),
    help=(
        "Number of build jobs to run in parallel. "
        "(default: # of processors [{0}])".format(GetCPUCount())
    ),
)
group.add_argument(
    "--build",
    type=str,
    help=(
        "Build directory for OSL and 3rd-party dependencies "
        "(default: <install_dir>/build)"
    ),
)
group.add_argument(
    "--build-args",
    type=str,
    nargs="*",
    default=[],
    help=(
        "Custom arguments to pass to build system when "
        "building libraries (see docs above)"
    ),
)
group.add_argument(
    "--force",
    type=str,
    action="append",
    dest="force_build",
    default=[],
    help=("Force download and build of specified library " "(see docs above)"),
)
group.add_argument(
    "--force-all", action="store_true", help="Force download and build of all libraries"
)
group.add_argument(
    "--generator",
    type=str,
    help=("CMake generator to use when building libraries with " "cmake"),
)
group.add_argument(
    "--toolset",
    type=str,
    help=("CMake toolset to use when building libraries with " "cmake"),
)

group = parser.add_argument_group(title="3rd Party Dependency Build Options")
group.add_argument(
    "--src",
    type=str,
    help=(
        "Directory where dependencies will be downloaded "
        "(default: <install_dir>/src)"
    ),
)
group.add_argument(
    "--inst",
    type=str,
    help=("Directory where dependencies will be installed " "(default: <install_dir>)"),
)

group = parser.add_argument_group(title="OSL Options")

(SHARED_LIBS, MONOLITHIC_LIB) = (0, 1)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--build-shared",
    dest="build_type",
    action="store_const",
    const=SHARED_LIBS,
    default=SHARED_LIBS,
    help="Build individual shared libraries (default)",
)
subgroup.add_argument(
    "--build-monolithic",
    dest="build_type",
    action="store_const",
    const=MONOLITHIC_LIB,
    help="Build a single monolithic shared library",
)

group.add_argument(
    "--debug",
    dest="build_debug",
    action="store_true",
    help="Build with debugging information",
)

# subgroup = group.add_mutually_exclusive_group()
# subgroup.add_argument("--tests", dest="build_tests", action="store_true",
#                       default=False, help="Build unit tests")

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--python",
    dest="build_python",
    action="store_true",
    default=False,
    help="Build python based components " "(default)",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--prefer-safety-over-speed",
    dest="safety_first",
    action="store_true",
    default=True,
    help="Enable extra safety checks (which may negatively "
    "impact performance) against malformed input files "
    "(default)",
)
subgroup.add_argument(
    "--prefer-speed-over-safety",
    dest="safety_first",
    action="store_false",
    help="Disable performance-impacting safety checks against " "malformed input files",
)

#
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--zlib",
    dest="build_zlib",
    action="store_true",
    default=False,
    help="Build zlib for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--boost",
    dest="build_boost",
    action="store_true",
    default=False,
    help="Build boost for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--llvm",
    dest="build_llvm",
    action="store_true",
    default=False,
    help="Build llvm for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--clang",
    dest="build_clang",
    action="store_true",
    default=False,
    help="Build clang for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--pugixml",
    dest="build_pugixml",
    action="store_true",
    default=False,
    help="Build pugixml for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--openexr",
    dest="build_openexr",
    action="store_true",
    default=False,
    help="Build openexr for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--tiff",
    dest="build_tiff",
    action="store_true",
    default=False,
    help="Build tiff for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--jpeg",
    dest="build_jpeg",
    action="store_true",
    default=False,
    help="Build jpeg for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--png",
    dest="build_png",
    action="store_true",
    default=False,
    help="Build png for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--flex",
    dest="build_flex",
    action="store_true",
    default=False,
    help="Build flex for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--bison",
    dest="build_bison",
    action="store_true",
    default=False,
    help="Build bison for OSL",
)

#

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--ptex",
    dest="enable_ptex",
    action="store_true",
    default=False,
    help="Enable Ptex support in imaging",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--openvdb",
    dest="enable_openvdb",
    action="store_true",
    default=False,
    help="Enable OpenVDB support in imaging",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--libraw",
    dest="build_libraw",
    action="store_true",
    default=False,
    help="Build libraw for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--openimageio",
    dest="build_oiio",
    action="store_true",
    default=False,
    help="Build OpenImageIO for OSL",
)

subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--opencolorio",
    dest="build_ocio",
    action="store_true",
    default=False,
    help="Build OpenColorIO for OSL",
)
# --------------------------------------
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--partio",
    dest="build_partio",
    action="store_true",
    default=False,
    help="Build partio for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--pybind11",
    dest="build_pybind11",
    action="store_true",
    default=False,
    help="Build pybind11 for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--ffmpeg",
    dest="build_ffmpeg",
    action="store_true",
    default=False,
    help="Build ffmpeg for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--field3d",
    dest="build_field3d",
    action="store_true",
    default=False,
    help="Build field3d for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--opencv",
    dest="build_opencv",
    action="store_true",
    default=False,
    help="Build opencv for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--gif",
    dest="build_gif",
    action="store_true",
    default=False,
    help="Build gif for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--heif",
    dest="build_heif",
    action="store_true",
    default=False,
    help="Build heif for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--squish",
    dest="build_squish",
    action="store_true",
    default=False,
    help="Build squish for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--dcmtk",
    dest="build_dcmtk",
    action="store_true",
    default=False,
    help="Build dcmtk for OSL",
)
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--webp",
    dest="build_webp",
    action="store_true",
    default=False,
    help="Build webp for OSL",
)
# --------------------------------------

#
subgroup = group.add_mutually_exclusive_group()
subgroup.add_argument(
    "--osl",
    dest="build_osl",
    action="store_true",
    default=False,
    help="Build OpenShadingLanguage for OSL",
)

args = parser.parse_args()


class InstallContext:
    def __init__(self, args):

        # Assume the OSL source directory is in the parent directory
        self.oslSrcDir = os.path.normpath(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..")
        )

        # Directory where OSL will be installed
        self.oslInstDir = os.path.abspath(args.install_dir)

        # Directory where dependencies will be installed
        self.instDir = os.path.abspath(args.inst) if args.inst else self.oslInstDir

        # Directory where dependencies will be downloaded and extracted
        self.srcDir = (
            os.path.abspath(args.src)
            if args.src
            else os.path.join(self.oslInstDir, "src")
        )

        # Directory where OSL and dependencies will be built
        self.buildDir = (
            os.path.abspath(args.build)
            if args.build
            else os.path.join(self.oslInstDir, "build")
        )

        # Prerequisites
        self.pythonInstallDir = os.path.normpath(os.environ["PYTHON_LOCATION"])
        self.qtInstallDir = os.path.normpath(os.environ["QT_LOCATION"])
        self.nasmInstallDir = os.path.normpath(os.environ["NASM_LOCATION"])
        self.gitInstallDir = os.path.normpath(os.environ["GIT_LOCATION"])
        self.cmakeInstallDir = os.path.normpath(os.environ["CMAKE_LOCATION"])
        self.vcInstallDir = os.path.normpath(os.environ["VCVARS_LOCATION"])

        # Determine which downloader to use.  The reason we don't simply
        # use urllib2 all the time is that some older versions of Python
        # don't support TLS v1.2, which is required for downloading some
        # dependencies.
        if find_executable("curl"):
            self.downloader = DownloadFileWithCurl
            self.downloaderName = "curl"
        elif Windows() and find_executable("powershell"):
            self.downloader = DownloadFileWithPowershell
            self.downloaderName = "powershell"
        else:
            self.downloader = DownloadFileWithUrllib
            self.downloaderName = "built-in"

        # CMake generator and toolset
        self.cmakeGenerator = args.generator
        self.cmakeToolset = args.toolset

        # Number of jobs
        self.numJobs = args.jobs
        if self.numJobs <= 0:
            raise ValueError("Number of jobs must be greater than 0")

        # Build arguments
        self.buildArgs = dict()
        for a in args.build_args:
            (depName, _, arg) = a.partition(",")
            if not depName or not arg:
                raise ValueError("Invalid argument for --build-args: {}".format(a))
            if depName.lower() not in AllDependenciesByName:
                raise ValueError("Invalid library for --build-args: {}".format(depName))

            self.buildArgs.setdefault(depName.lower(), []).append(arg)

        # Build type
        self.buildDebug = args.build_debug
        self.buildShared = args.build_type == SHARED_LIBS
        self.buildMonolithic = args.build_type == MONOLITHIC_LIB

        # Build options
        self.safetyFirst = args.safety_first

        # Dependencies that are forced to be built
        self.forceBuildAll = args.force_all
        self.forceBuild = [dep.lower() for dep in args.force_build]

        # Optional components
        self.buildPython = args.build_python

        self.buildOIIO = args.build_oiio
        self.buildZLIB = args.build_zlib
        self.buildBOOST = args.build_boost
        self.buildLLVM = args.build_llvm
        self.buildCLANG = args.build_clang
        self.buildPUGIXML = args.build_pugixml
        self.buildOPENEXR = args.build_openexr
        self.buildTIFF = args.build_tiff
        self.buildJPEG = args.build_jpeg
        self.buildPNG = args.build_png
        self.buildFLEX = args.build_flex
        self.buildBISON = args.build_bison

        self.buildLIBRAW = args.build_libraw
        self.buildPTEX = args.enable_ptex
        self.buildOPENVDB = args.enable_openvdb
        self.buildOCIO = args.build_ocio
        self.buildPARTIO = args.build_partio
        self.buildPYBIND11 = args.build_pybind11
        self.buildFFMPEG = args.build_ffmpeg
        self.buildFIELD3D = args.build_field3d
        self.buildOPENCV = args.build_opencv
        self.buildGIF = args.build_gif
        self.buildHEIF = args.build_heif
        self.buildSQUISH = args.build_squish
        self.buildDCMTK = args.build_dcmtk
        self.buildWEBP = args.build_webp

        self.buildOSL = args.build_osl

    def GetBuildArguments(self, dep):
        return self.buildArgs.get(dep.name.lower(), [])

    def ForceBuildDependency(self, dep):
        # Never force building a Python dependency, since users are required
        # to build these dependencies themselves.
        if type(dep) is PythonDependency:
            return False
        return self.forceBuildAll or dep.name.lower() in self.forceBuild


try:
    context = InstallContext(args)
except Exception as e:
    PrintError(str(e))
    sys.exit(1)

verbosity = args.verbosity

# Augment PATH on Windows so that 3rd-party dependencies can find libraries
# they depend on. In particular, this is needed for building IlmBase/OpenEXR.
extraPaths = []
extraPythonPaths = []
if Windows():
    extraPaths.append(os.path.join(context.instDir, "lib"))
    extraPaths.append(os.path.join(context.instDir, "bin"))

if extraPaths:
    paths = os.environ.get("PATH", "").split(os.pathsep) + extraPaths
    os.environ["PATH"] = os.pathsep.join(paths)

if extraPythonPaths:
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep) + extraPythonPaths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


requiredDependencies = []

# Determine list of dependencies that are required based on options
# user has selected.
# BOOST is deleted becauseI want to use system installed boost
if Windows():
    if context.buildZLIB:
        requiredDependencies += [ZLIB]
    if context.buildBOOST:
        requiredDependencies += [BOOST]

if Windows():
    if context.buildLLVM:
        requiredDependencies += [LLVM]
    if context.buildCLANG:
        requiredDependencies += [CLANG]


if context.buildOSL:
    if context.buildPUGIXML:
        requiredDependencies += [PUGIXML]
    if context.buildPYBIND11:
        requiredDependencies += [PYBIND11]
    if Windows():
        if context.buildPARTIO:
            requiredDependencies += [GLUT, PARTIO]
        if context.buildFLEX:  # this is for both flex and bison
            requiredDependencies += [WINFLEXBISON]
    if context.buildOPENEXR:
        requiredDependencies += [OPENEXR]

if context.buildOIIO:
    if context.buildJPEG:
        requiredDependencies += [JPEGTURBO]
    if context.buildTIFF:
        requiredDependencies += [TIFF]
    if context.buildPNG:
        requiredDependencies += [PNG]
    if context.buildOPENEXR:
        requiredDependencies += [OPENEXR]
    requiredDependencies += [OPENIMAGEIO]

if context.buildPTEX:
    requiredDependencies += [PTEX]

if context.buildOCIO:
    requiredDependencies += [OPENCOLORIO]

if context.buildOPENVDB:
    requiredDependencies += [
        OPENEXR,
        OPENVDB,
    ]

if Windows():
    if context.buildLIBRAW:
        requiredDependencies += [LIBRAW]
    if context.buildFFMPEG:
        requiredDependencies += [FFMPEG]
    if context.buildFIELD3D:
        requiredDependencies += [FIELD3D]
    if context.buildOPENCV:
        requiredDependencies += [OPENCV]
    if context.buildGIF:
        requiredDependencies += [GIF]
    if context.buildHEIF:
        requiredDependencies += [HEIF]
    if context.buildHEIF:
        requiredDependencies += [HEIF]
    if context.buildSQUISH:
        requiredDependencies += [SQUISH]
    if context.buildDCMTK:
        requiredDependencies += [DCMTK]
    if context.buildWEBP:
        requiredDependencies += [WEBP]


dependenciesToBuild = []
for dep in requiredDependencies:
    if context.ForceBuildDependency(dep) or not dep.Exists(context):
        if dep not in dependenciesToBuild:
            dependenciesToBuild.append(dep)


# Verify toolchain needed to build required dependencies
if (
    not find_executable("g++")
    and not find_executable("clang")
    and not GetXcodeDeveloperDirectory()
    and not GetVisualStudioCompilerAndVersion()
):
    PrintError("C++ compiler not found -- please install a compiler")
    sys.exit(1)

if find_executable("python"):
    # Error out if a 64bit version of python interpreter is not found
    # Note: Ideally we should be checking the python binary found above, but
    # there is an assumption (for very valid reasons) at other places in the
    # script that the python process used to run this script will be found.
    isPython64Bit = ctypes.sizeof(ctypes.c_voidp) == 8
    if not isPython64Bit:
        PrintError("64bit python not found -- please install it and adjust your" "PATH")
        sys.exit(1)

    # Error out on Windows with Python 3.8+. OSL currently does not support
    # these versions due to:
    # https://docs.python.org/3.8/whatsnew/3.8.html#bpo-36085-whatsnew
    isPython38 = sys.version_info.major >= 3 and sys.version_info.minor >= 8
    if Windows() and isPython38:
        PrintError("Python 3.8+ is not supported on Windows")
        sys.exit(1)

else:
    PrintError("python not found -- please ensure python is included in your " "PATH")
    sys.exit(1)

if find_executable("cmake"):
    # Check cmake requirements
    if Windows():
        # Windows build depend on boost 1.70, which is not supported before
        # cmake version 3.14
        cmake_required_version = (3, 14)
    else:
        cmake_required_version = (3, 12)
    cmake_version = GetCMakeVersion()
    if not cmake_version:
        PrintError("Failed to determine CMake version")
        sys.exit(1)

    if cmake_version < cmake_required_version:

        def _JoinVersion(v):
            return ".".join(str(n) for n in v)

        PrintError(
            "CMake version {req} or later required to build OSL, "
            "but version found was {found}".format(
                req=_JoinVersion(cmake_required_version),
                found=_JoinVersion(cmake_version),
            )
        )
        sys.exit(1)
else:
    PrintError("CMake not found -- please install it and adjust your PATH")
    sys.exit(1)

if JPEGTURBO in requiredDependencies:
    # NASM is required to build libjpeg-turbo
    if Windows() and not find_executable("nasm"):
        PrintError("nasm not found -- please install it and adjust your PATH")
        sys.exit(1)

if context.buildOSL:
    dependenciesToBuild.append(OSL)

# Summarize
summaryMsg = """
Building with settings:         ------------
  Python Install directory      {pythonInstallDir}
  Qt Install Directory          {qtInstallDir}
  Nasm Install directory        {nasmInstallDir}
  Git Install directory         {gitInstallDir}
  Cmake Install directory       {cmakeInstallDir}
  VC Install directory          {vcInstallDir}
                                ------------
  OSL source directory          {oslSrcDir}
  OSL install directory         {oslInstDir}
  3rd-party source directory    {srcDir}
  3rd-party install directory   {instDir}
  Build directory               {buildDir}
  CMake generator               {cmakeGenerator}
  CMake toolset                 {cmakeToolset}
  Downloader                    {downloader}

  Python support                {buildPython}
    Python 3:                   {enablePython3}

  Building                      {buildType}
    Config                      {buildConfig}

  Mandatory Dependencies:       ------------ 
    Zlib:                       {buildZLIB}
    Boost:                      {buildBOOST}
    LLVM:                       {buildLLVM}
    CLANG:                      {buildCLANG}
    PugiXML:                    {buildPUGIXML}
    OpenEXR/IlmBase:            {buildOPENEXR}
    OpenImageIO:                {buildOIIO}
    TIFF:                       {buildTIFF}
    JPEG:                       {buildJPEG}
    PNG:                        {buildPNG}
    Flex:                       {buildFLEX}
    Bison:                      {buildBISON}

  Optional Dependencies:        ------------     
    Ptex:                       {buildPTEX}
    OpenColorIO:                {buildOCIO}
    LibRaw:                     {buildLIBRAW}
    OpenvVDB:                   {buildOPENVDB}
    Partio:                     {buildPARTIO}
    PyBind11:                   {buildPYBIND11}
    ffmpeg:                     {buildFFMPEG}
    Field3D:                    {buildFIELD3D}
    OpenCV:                     {buildOPENCV}
    Gif:                        {buildGIF}
    Heif:                       {buildHEIF}
    Squish:                     {buildSQUISH}
    DCMTK:                      {buildDCMTK}
    Webp:                       {buildWEBP}

                                ------------
  OSL                           {buildOSL}
  ------------------------------------------
  Dependencies                  {dependencies}
  ------------------------------------------
  """

if context.buildArgs:
    summaryMsg += """
  Build arguments               {buildArgs}"""


def FormatBuildArguments(buildArgs):
    s = ""
    for depName in sorted(buildArgs.keys()):
        args = buildArgs[depName]
        s += """
                                {name}: {args}""".format(
            name=AllDependenciesByName[depName].name, args=" ".join(args)
        )
    return s.lstrip()


summaryMsg = summaryMsg.format(
    oslSrcDir=context.oslSrcDir,
    oslInstDir=context.oslInstDir,
    srcDir=context.srcDir,
    buildDir=context.buildDir,
    instDir=context.instDir,
    cmakeGenerator=(
        "Default" if not context.cmakeGenerator else context.cmakeGenerator
    ),
    cmakeToolset=("Default" if not context.cmakeToolset else context.cmakeToolset),
    downloader=(context.downloaderName),
    dependencies=(
        "None"
        if not dependenciesToBuild
        else ", ".join([d.name for d in dependenciesToBuild])
    ),
    buildArgs=FormatBuildArguments(context.buildArgs),
    buildType=(
        "Shared libraries"
        if context.buildShared
        else "Monolithic shared library"
        if context.buildMonolithic
        else ""
    ),
    pythonInstallDir=context.pythonInstallDir,
    qtInstallDir=context.qtInstallDir,
    nasmInstallDir=context.nasmInstallDir,
    gitInstallDir=context.gitInstallDir,
    cmakeInstallDir=context.cmakeInstallDir,
    vcInstallDir=context.vcInstallDir,
    buildConfig=("Debug" if context.buildDebug else "Release"),
    buildPTEX=("On" if context.buildPTEX else "Off"),
    buildOIIO=("On" if context.buildOIIO else "Off"),
    buildZLIB=("On" if context.buildZLIB else "Off"),
    buildBOOST=("On" if context.buildBOOST else "Off"),
    buildLLVM=("On" if context.buildLLVM else "Off"),
    buildCLANG=("On" if context.buildCLANG else "Off"),
    buildPUGIXML=("On" if context.buildPUGIXML else "Off"),
    buildOPENEXR=("On" if context.buildOPENEXR else "Off"),
    buildTIFF=("On" if context.buildTIFF else "Off"),
    buildJPEG=("On" if context.buildJPEG else "Off"),
    buildPNG=("On" if context.buildPNG else "Off"),
    buildFLEX=("On" if context.buildFLEX else "Off"),
    buildBISON=("On" if context.buildBISON else "Off"),
    buildLIBRAW=("On" if context.buildLIBRAW else "Off"),
    buildOSL=("On" if context.buildOSL else "On"),
    buildOCIO=("On" if context.buildOCIO else "Off"),
    buildOPENVDB=("On" if context.buildOPENVDB else "Off"),
    buildPARTIO=("On" if context.buildPARTIO else "Off"),
    buildPYBIND11=("On" if context.buildPYBIND11 else "Off"),
    buildFFMPEG=("On" if context.buildFFMPEG else "Off"),
    buildFIELD3D=("On" if context.buildFIELD3D else "Off"),
    buildOPENCV=("On" if context.buildOPENCV else "Off"),
    buildGIF=("On" if context.buildGIF else "Off"),
    buildHEIF=("On" if context.buildHEIF else "Off"),
    buildSQUISH=("On" if context.buildSQUISH else "Off"),
    buildDCMTK=("On" if context.buildDCMTK else "Off"),
    buildWEBP=("On" if context.buildWEBP else "Off"),
    buildPython=("On" if context.buildPython else "Off"),
    enablePython3=("On" if Python3() else "Off"),
)

Print(summaryMsg)

if args.dry_run:
    sys.exit(0)

# Scan for any dependencies that the user is required to install themselves
# and print those instructions first.
pythonDependencies = [
    dep for dep in dependenciesToBuild if type(dep) is PythonDependency
]
if pythonDependencies:
    for dep in pythonDependencies:
        Print(dep.getInstructions())
    sys.exit(1)

# Ensure directory structure is created and is writable.
for dir in [context.oslInstDir, context.instDir, context.srcDir, context.buildDir]:
    try:
        if os.path.isdir(dir):
            testFile = os.path.join(dir, "canwrite")
            open(testFile, "w").close()
            os.remove(testFile)
        else:
            os.makedirs(dir)
    except Exception as e:
        PrintError(
            "Could not write to directory {dir}. Change permissions "
            "or choose a different location to install to.".format(dir=dir)
        )
        sys.exit(1)

try:
    # Download and install 3rd-party dependencies, followed by OSL.
    for dep in dependenciesToBuild:
        PrintStatus("\nInstalling {dep}...\n".format(dep=dep.name))
        dep.installer(
            context,
            buildArgs=context.GetBuildArguments(dep),
            force=context.ForceBuildDependency(dep),
        )
except Exception as e:
    PrintError(str(e))
    sys.exit(1)

# Done. Print out a final status message.
# requiredInPythonPath = set([os.path.join(context.oslInstDir, "lib", "python")])
# requiredInPythonPath.update(extraPythonPaths)

requiredInPath = set([os.path.join(context.oslInstDir, "bin")])
requiredInPath.update(extraPaths)

if Windows():
    requiredInPath.update(
        [
            os.path.join(context.oslInstDir, "lib"),
            os.path.join(context.instDir, "bin"),
            os.path.join(context.instDir, "lib"),
        ]
    )

Print(
    """
Success! To use OSL, please ensure that you have:"""
)

# if context.buildPython:
#     Print(
#         """
#     The following in your PYTHONPATH environment variable:
#     {requiredInPythonPath}""".format(
#             requiredInPythonPath="\n    ".join(sorted(requiredInPythonPath))
#         )
#     )

Print(
    """
    The following in your PATH environment variable:
    {requiredInPath}
""".format(
        requiredInPath="\n    ".join(sorted(requiredInPath))
    )
)

# if context.buildPrman:
#     Print("See documentation at http://openusd.org/docs/RenderMan-OSL-Imaging-Plugin.html "
#           "for setting up the RenderMan plugin.\n")
