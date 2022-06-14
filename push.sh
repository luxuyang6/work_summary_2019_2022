# auto push to github
git add -A
git commit -m "`date '+%Y-%m-%d %H:%M:%S'`"
git push origin master
echo 'finished'
cmd /k