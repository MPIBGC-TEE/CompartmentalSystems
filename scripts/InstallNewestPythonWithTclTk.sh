# This script downloads the newest versions of tcl/tk and python3 interpreter src code 
# It also builds and installs them 
# under the users home directory in ~/opt 
# after a successfull installation you can switch to the newest interpreter by
# adding the ~/opt/bin to your PATH variable via your .profile or .bashrc files


# chose a place in your home directory where the python interpretor should live
prefix="${HOME}/opt"
mkdir -p $prefix
declare -A urls 
declare -A tarNames
declare -A dirNames
declare -A prefixes

# Go to the tcl/tk download site and find the newest version
# Also go to the python  download site and find the newest version
urls[tcl]="https://prdownloads.sourceforge.net/tcl/tcl8.6.8-src.tar.gz"
urls[tk]="https://prdownloads.sourceforge.net/tcl/tk8.6.8-src.tar.gz"
#urls[tcl]="https://prdownloads.sourceforge.net/tcl/tcl8.7a1-src.tar.gz"
#urls[tk]="https://prdownloads.sourceforge.net/tcl/tk8.7a1-src.tar.gz"
urls[python]="https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tgz"

for key in tcl tk python
	do 
		# downlaod
		url=${urls[$key]}
		echo $url
		tarN=$(basename $url)
		tarNames[$key]=$tarN
		dirN=$(echo $tarN|sed 's/-src\.tar\.gz$\|\.tgz$//')
		rm -rf $dirN
		#rm $tarN*
		#wget $url
		
		# untar
		tar xfz $tarN
		
		#remember srcdir dirNames array
		echo $dirN
		dirNames[$key]=$dirN
		prefixes[$key]=${prefix}/$dirN
done

# we start by configuring, making and installing the tcl sources 
tclBuildDir=${dirNames[tcl]}/unix
cd $tclBuildDir
	pwd
	./configure --prefix=${prefixes[tcl]}
	make
	make install
cd -

# now we configure make and install the tk sources
cd ${dirNames[tk]}/unix
	pwd
	./configure --prefix=${prefixes[tk]} --with-tcl="../../$tclBuildDir"
	make
	make install
cd -

# finally we configure make and install the python interpreter
cd ${dirNames[python]}
	Libs="-L${prefixes[tcl]}/lib -L${prefixes[tk]}/lib"
	Includes="-I${prefixes[tcl]}/include -I${prefixes[tk]}/include"
	echo ${Libs}
	echo ${Includes}
	command="./configure --prefix=${prefixes[python]} --with-tcltk-includes=\"$Includes\" --with-tcltk-libs=\"$Libs\""
	echo $command
	eval $command
	#--exec-prefix=${prefixes[python]} 
	make
	make install
cd -

