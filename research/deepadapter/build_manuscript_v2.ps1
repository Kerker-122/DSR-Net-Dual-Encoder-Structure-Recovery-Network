$ErrorActionPreference = 'Stop'
$draftRoot = $PSScriptRoot
$chapterNames = @('01_Introduction.md','02_Related_Work.md','03_Method.md','04_Experiments.md','05_Conclusion.md')
$parts = @(
    '# DeepAdapter：面向低照度海洋动物分割的互补表征适配与渐进形状细化',
    'DeepAdapter: Complementary Representation Adaptation and Progressive Shape Refinement for Low-Light Marine Animal Segmentation',
    '> 中文工作初稿 v2：依据用户提供的 SPMNet、NightAdapter 与 UNeXT 全文完善。公式来源与实现差异见独立核对表；正文的〔待补〕与〔待核〕仅保留当前模型配置和真实实验仍未确认的内容。SPMNet 的实验数值未转用为本研究结果。'
)
foreach ($name in $chapterNames) {
    $source = Join-Path $draftRoot ('versions/v2/chapters/' + $name)
    Copy-Item -LiteralPath $source -Destination (Join-Path $draftRoot ('chapters/' + $name)) -Force
    $parts += Get-Content -LiteralPath $source -Raw -Encoding UTF8
}
$parts += Get-Content -LiteralPath (Join-Path $draftRoot 'refs/references.md') -Raw -Encoding UTF8
$parts -join "`n`n" | Set-Content -LiteralPath (Join-Path $draftRoot 'manuscript_zh_v2.md') -Encoding UTF8
Write-Output 'Built manuscript_zh_v2.md and synchronized five current chapter files; v1 preserved.'
