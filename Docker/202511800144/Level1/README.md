# Level1 实践日志

在 `编写对应的 Dockerfile` 步骤中注意到 L8: `COPY requirements.txt .` 因为该步骤中并为给出相应的 `requirements.txt` 文件。因此，将 L8 部分代码删除，并将 L11 部分代码修改为 `RUN pip install flask`

# 任务目标
运行成功日志 见`L1log.png`
运行成功截图 见`L1result.png`