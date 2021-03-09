#!/bin/sh

DIR=build/html/ja

if [ $1 = "ja" ]; then
    echo -n Reforming html...
    find build/html/ja/ -name "*.html" | xargs sed -i 's/\(<p class="admonition-title">\)Restriction/\1制限事項/g'
    echo done
fi
