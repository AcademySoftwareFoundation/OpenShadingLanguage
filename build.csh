# use newer version of cmake
setenv PATH /opt/intel/cmake/bin:$PATH

# use gcc, not icc 
unsetenv CC
unsetenv CXX
source /opt/intel/compilers_and_libraries_2017.2.174/linux/bin/compilervars.csh intel64
setenv CC icc
setenv CXX icpc


setenv PKG_CONFIG_PATH /nfs/pdx/home/amwells/Pixar/OSL/install/lib/pkgconfig/
setenv THIRD_PARTY_TOOLS_HOME /nfs/pdx/home/amwells/Pixar/OSL/install/
setenv BOOST_ROOT /nfs/pdx/home/amwells/Pixar/boost/
setenv OPENIMAGEIOHOME /nfs/pdx/home/amwells/Pixar/OSL/oiio/dist/linux64
setenv LD_LIBRARY_PATH /nfs/pdx/home/amwells/Pixar/OSL/oiio/dist/linux64/lib:$LD_LIBRARY_PATH
setenv LD_LIBRARY_PATH /nfs/pdx/home/amwells/Pixar/boost/lib:$LD_LIBRARY_PATH
setenv LLVM_DIRECTORY /nfs/pdx/home/amwells/Pixar/OSL/install/
setenv LLVM_STATIC 1
setenv LDFLAGS "-L/nfs/pdx/home/amwells/Pixar/boost/lib $LDFLAGS"
#setenv CXXFLAGS "-std=c++11 -I/nfs/pdx/home/amwells/Pixar/boost/include -I/usr/include/c++/4.8.2 -Wno-sign-compare -Wno-unused-local-typedefs"
#setenv CXXFLAGS "-std=c++11 -I/nfs/pdx/home/amwells/Pixar/boost/include -Wno-sign-compare -Wno-unused-local-typedefs"
setenv SPECIAL_COMPILE_FLAGS "-Wno-sign-compare -Wno-unused-local-typedefs -Werror=unknown-pragmas"

#Couldn't figure out how to pass these in externally and had to embedd the following compilation flags into src/liboslexec/CMakeLists.txt
#setenv LLVM_COMPILE_FLAGS "-I/usr/include/c++/4.8.2 -I/usr/include/c++/4.8.2/x86_64-redhat-linux"

#Have to link with -lirc to fix 
#liboslexec.so: undefined reference to `__intel_sse2_strrchr'

#Have to link with -ltinfo to fix 
#liboslexec.so: undefined reference to `set_curterm'


#Have to link with -lboost_thread to fix
#liboslexec.so: undefined reference to `boost::thread_detail::rollback_once_region(boost::once_flag&)'

setenv LDFLAGS "-lirc $LDFLAGS"
#setenv LDFLAGS "-ltinfo $LDFLAGS"
#setenv LDFLAGS "-lboost_thread $LDFLAGS"
#setenv LDFLAGS "-ltinfo"


#make
#make -j28 USE_CPP11=1 VERBOSE=1
make profile VERBOSE=1 USE_CCACHE=0 LLVM_VERSION=3.9.1 BOOST_HOME=/nfs/pdx/home/amwells/Pixar/boost OPENEXR_HOME=/nfs/pdx/home/amwells/Pixar/OSL/install ILMBASE_HOME=/nfs/pdx/home/amwells/Pixar/OSL/install -j 48

# after  building add 
source /opt/intel/compilers_and_libraries_2017.2.174/linux/bin/compilervars.csh intel64
setenv PATH /nfs/pdx/home/amwells/Pixar/OSL/OSL_Dev/OpenShadingLanguage/dist/linux64.profile/bin:$PATH
setenv LD_LIBRARY_PATH /nfs/pdx/home/amwells/Pixar/OSL/OSL_Dev/OpenShadingLanguage/dist/linux64.profile/lib:$LD_LIBRARY_PATH
setenv LD_LIBRARY_PATH ~/Pixar/boost/lib/:$LD_LIBRARY_PATH
setenv LD_LIBRARY_PATH ~/Pixar/OSL/install/lib/:$LD_LIBRARY_PATH


#LDFLAGS=-lboost_thread -ltinfo -lirc -L/nfs/pdx/home/amwells/Pixar/boost/lib
