#!/bin/bash
echo "======================================"
echo "  检查 backend/app.py 内容"
echo "======================================"
echo "当前目录: $(pwd)"
ls -l backend/app.py
echo "--------------------------------------"
echo "搜索 'pdf_url':"
grep -n "pdf_url" backend/app.py
echo "--------------------------------------"
echo "搜索 'total_chars':"
grep -n "total_chars" backend/app.py
echo "--------------------------------------"
echo "文件最后 20 行:"
tail -n 20 backend/app.py
echo "======================================"
