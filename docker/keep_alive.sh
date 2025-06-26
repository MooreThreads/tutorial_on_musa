#!/bin/bash
/workspace/check_status.sh  # 执行原任务
exec tail -f /dev/null      # 永久阻塞（exec 确保成为主进程）
