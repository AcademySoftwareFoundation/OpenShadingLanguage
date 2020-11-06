## **Building, Using and Debugging OSL in Windows**
---

### **Building**

There are bunch od prerequisites for building OSL that I explained in following sections

### **Cleaning Up the PATH**

Be sure you don't have any custom item added to PATH in your global Environment Variables of Windows

you can reach it:

Start Menu > Edit the system environmment variables > Environment Variables > User Variables/System Variables

so you should not have any path that refer to previous osl installation folders

Recommendation: remove added path for cmake, git, python, Qt, llvm

it should be clean now.

### **Set-Up needed Environment Variables**

For running batch scripts without any problem I recommend to create some base environment variables I listed below

**Needed Environment Variables: (Key/Value based on my installation folder)**

Name | Lunch order
------- | ----------------
BASE_LOCATION | D:\madoodia\sdks
PYTHON_LOCATION | D:\madoodia\sdks
QT_LOCATION | C:\Qt\5.15.0\msvc2019_64
NASM_LOCATION | C:\NASM
GIT_LOCATION | "C:\Program Files\Git"
CMAKE_LOCATION | "C:\Program Files\CMake"
VCVARS_LOCATION | "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build"
|

Open **build_scripts/osl_env_vars_setup.bat** in your editor and edit paths' values based on your installation folders.


### **Set Flags based on your needs**

The main build launcher is here **build_scripts/build_osl.bat**

open it and add or remove your flags based on the table below

**Table of flags we can assign to the command**

Flag | State       | Can Use 
------- | -------- | -----
--osl  | Implemented | Yes
--python  | Implemented | Yes
--zlib  | Implemented | Yes
--boost  | Implemented | Yes
--llvm  | Implemented | Yes
--clang  | Implemented | Yes
--pugixml  | Implemented | Yes
--openexr  | Implemented | Yes
--tiff  | Implemented | Yes
--jpeg  | Implemented | Yes
--png  | Implemented | Yes
--flex  | Implemented | Yes
--bison  | Implemented | Yes
--opencolorio  | Implemented | Yes
--openimageio  | Implemented | Yes
--libraw  | Implemented | Yes
--pybind11 | Implemented | Yes
--Ptex | Implemented | No
--openvdb | Implemented | No
--partio | Implemented | No
--ffmpeg | Not-Implemented | No
--field3d | Not-Implemented | No
--opencv | Not-Implemented | No
--gif | Not-Implemented | No
--heif | Not-Implemented | No
--squish | Not-Implemented | No
--dcmtk | Not-Implemented | No
--webp | Not-Implemented | No


---
### **Building (Release, Debug)**

Based on the config (Release or Debug) that you send for installer it will create directory in **BASE_LOCATION** for you

**BASE_LOCATION/osl_release** for Release mode
**BASE_LOCATION/osl_debug** for Debug mode

**IMPORTANT**: `Good to know you can not have Release and Debug files in one directory at the same time, it makes conflict and give you weired errors. So keep them separate.`

### **Be Patient**

You should be patient because the whole process (based on your PC config) takes time between 30 minutes to 1 hour

And it take:

**Release**: 8Gb of your hard disk (after installation) (Download size is ~350Mb)
**Release**: ~100Gb of your hard disk (after installation) (Download size is ~400Mb)

---
### **Running**

**Recommendation**: use **Powershell**

**TIP**: If you add your osl installation folders (both release and debug) in global Windows Environment Variable, you will get error.

So you can add release path or debug path in your global environment variables.
Or create and use batch scripts.

For example, you can open osltoy in releae mode with running **osltoy_release_launcher.bat**

For adding path you should add these three paths to your global env variables:
- **PATH** : <osl install dir>\bin;
- **PATH** : <osl install dir>\lib;
- **PYTHONPATH** : <osl install dir>\osl_<config>/lib/python<version>;

or create your own (for example for VSCode) and run it in your config mode.

if you have question, feel free to ask please.

---
### **Development in Windows**


After editing your code, when you want to see your changes you should build osl again. But this time, it is so faster than before, (if you keep build folders)
So just need to rebuild it in release mode.

**Release mode**:
- open powershell
- cd OpenShadingLanguage repo
- cd build_script
- run build_osl.bat release
- run osl_release_sln.bat
- example: set osltoy as default project
    - run it in release x64 mode

**Debug mode**:
- open powershell
- cd OpenShadingLanguage repo
- cd build_script
- run build_osl.bat debug
- run osl_debug_sln.bat
- example: set osltoy as default project
    - run it in debug x64 mode

If you open Visual Studio solution (sln) from release or debug, you can build it from inside it after your code editing

I recommend using: (in **powershell**)
- **osl_release_sln.bat** or **osl_debug_sln.bat**

now you can use Visual Studio for debugging.

---
Reza Aarabi

(C) madoodia.com
