with import <nixpkgs> {};
#(python36.withPackages (ps: [ ps.pip ps.setuptools ps.numpy ps.matplotlib ])).env
( let 
  my_toolz =python35.pkgs.buildPythonPackage rec {
		pname="toolz";
		version = "0.7.4";

    src = python35.pkgs.fetchPypi {
      inherit pname version;
      sha256 =  "43c2c9e5e7a16b6c88ba3088a9bfc82f7db8e13378be7c78d6c14a5f8ed05afd";
    };

    doCheck = false;

    meta = {
      homepage="http://github.com/pytoolz/toolz/";
      description = "List processing tools and functional utilities";
    };
  };
in python35.withPackages (ps:[ps.numpy my_toolz ])
).env 
