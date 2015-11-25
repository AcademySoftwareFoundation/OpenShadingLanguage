#!/bin/bash

# Install OpenImageIO

OIIOBRANCH=${OIIOBRANCH:=master}
OIIOINSTALLDIR=${OIIOINSTALLDIR:=${PWD}/OpenImageIO}

if [ ! -e $OIIOINSTALLDIR ] ; then
    git clone https://github.com/OpenImageIO/oiio.git $OIIOINSTALLDIR
fi

( cd $OIIOINSTALLDIR ; git fetch --all -p && git checkout $OIIOBRANCH --force ; make nuke )
( cd $OIIOINSTALLDIR ; make ${OIIOMAKEFLAGS} VERBOSE=1 cmakesetup )
( cd $OIIOINSTALLDIR ; make ${OIIOMAKEFLAGS} )

echo "OIIOINSTALLDIR $OIIOINSTALLDIR"
ls -R $OIIOINSTALLDIR/dist
