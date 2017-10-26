#!/bin/bash

# Install OpenImageIO

OIIOREPO=${OIIOREPO:=OpenImageIO/oiio}
OIIOBRANCH=${OIIOBRANCH:=master}
OIIOINSTALLDIR=${OIIOINSTALLDIR:=${PWD}/OpenImageIO}

if [ ! -e $OIIOINSTALLDIR ] ; then
    git clone https://github.com/${OIIOREPO} $OIIOINSTALLDIR
fi

cd $OIIOINSTALLDIR
git fetch --all -p
git checkout $OIIOBRANCH --force
make nuke
make ${OIIOMAKEFLAGS} VERBOSE=1 cmakesetup
make ${OIIOMAKEFLAGS}

echo "OIIOINSTALLDIR $OIIOINSTALLDIR"
ls -R $OIIOINSTALLDIR/dist
