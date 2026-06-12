install_from_github_if_not_available(){
  pip show ${1};ret=$? 
  if [ $ret -eq 0 ]; then
      echo "allredy installed"
  else 
      pip install git+https://github.com/MPIBGC-TEE/${1}.git#egg=${1}
  fi 
}

install_from_github_if_not_available testinfrastructure
install_from_github_if_not_available LAPM

pip install -e .  
