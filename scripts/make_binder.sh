#!/bin/bash
branch_name="binder"

# remove old remote and local versions of the branch  if they exist



if git ls-remote --quiet --exit-code --heads origin ${branch_name}  
then 
  echo "found remote branch ${branch_name} and will remove it"  ;  git push -d origin ${branch_name}
else
  echo "remote branch ${branch_name} does not exist."
fi

if git show-ref  --quiet refs/heads/binder 
then 
  echo "erasing local branch ${branchname}."
  git branch -D ${branch_name}
else
  echo "local branch ${branchname} does not exist."
fi

# create new branch
git checkout -b ${branch_name}
# remove the large directories
git rm -r ../notebooks/nonl_gcm_3p* 
git rm -r ../notebooks/PNAS/
git commit -m "automatically created by ${0}  branch ${branch_name} from branch ${parent}. This branch has large directories removed from it and is intended to be checked out by binder in depth 1 which reduces it's size to a few MB. 
Never merge it back, since this will remove the files for the PNAS notebooks from the branch it gets merged or rebased to, which will force you to restore them manually."
git push --set-upstream origin ${branch_name}

# go back to the original branch
git checkout ${parent}
