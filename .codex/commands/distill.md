Run a standalone Academic Distillation (Agent C) on the specified paper: $ARGUMENTS

## Input Validation
- `$ARGUMENTS` 必须是目标论文的文件名（不含后缀）。例如：`rmsnorm`。
- 确认目标文件存在于全局文献池 `E:\allwork\cs336\paper\$ARGUMENTS\$ARGUMENTS.md`（或 `.tex`）中。

## Execution Instructions
严格执行以下步骤进行学术文献降维：

### Step 1: Agent C (Academic Decryptor) Initialization
- **System Profile & Rules:** 读取并严格遵循 `.codex/shared/agents/agentC.md` 中的所有约束与解码 Codebook。
- **Writing Style:** 遵守 `.codex/shared/writing_style.md` 中的高信息密度与 LaTeX 公式规范。

### Step 2: Context Hydration
- 读取 `E:\allwork\cs336\paper\$ARGUMENTS.md` 的完整内容进入上下文。

### Step 3: Distillation & Output
- 根据 Agent C 的框架生成该论文的“概念信息卡 (Concept Card)”。
- **Output:** 将结果严格输出至全局卡片池 `paperAfterC/$ARGUMENTS_ConceptCard.md`。

### Step 4: Completion
- 在终端打印提示：*"[Agent C] Global distillation complete for $ARGUMENTS. Concept Card added to the global pool."*