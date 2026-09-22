# Nature Skills 项目级安装记录

日期：2026-09-18。

## 用户范围约束

用户明确要求：skills 仅用于本项目；除了本项目外，不修改电脑中的其他文件，尤其 C 盘。后续安装依赖、缓存、临时文件、研究产物均须遵守此限制。不要改用户级 Codex 配置、技能目录或 Git 安装/配置。若某操作无法满足此约束，先说明并等待用户决定。

## 安装结果

- 用户下载的 ZIP 源码：`nature-skills-main/nature-skills-main/`。
- 项目安装位置：`.agents/skills/`。
- 安装 20 个完整目录（含 nature-shared），使用 PowerShell Copy-Item -Recurse，未仅复制 SKILL.md。
- 原脚本 `--pull --dest <项目>/.agents/skills` 已尝试：ZIP 无 Git 元数据，自动跳过 pull；随后由于本机没有 rsync 而中止，没有用此步骤完成安装。
- 随后完整复制目录，并执行未经修改的原脚本 `--check --dest <项目>/.agents/skills`。
- 验证退出码 0，20 项全部 MATCH。`Verified against unknown` 来自 ZIP 缺少 Git 提交元数据，不表示目录比较失败。
- Git Bash 使用 D 盘已有程序，以 --noprofile --norc 启动；仅为子进程设置 PATH 和 TMPDIR，未修改持久环境变量。
- 验证临时目录限定为项目下 `.skill-install-tmp/`，校验脚本已清理自身临时文件。
- 未安装 Python/Node 依赖，未配置 MCP、API 密钥或外部服务；本次验证只证明技能目录与下载源码一致，不代表每个附带程序的运行依赖已具备。

后续更新仍需使用用户指定的项目范围；不要直接执行仓库脚本的默认全局安装路径。
