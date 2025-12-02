@echo off
echo Running git status... > git_debug.log
git status >> git_debug.log 2>&1
echo Done. >> git_debug.log
