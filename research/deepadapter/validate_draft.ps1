param([string]$ManuscriptName = 'manuscript_zh_v2.md')
$ErrorActionPreference = 'Stop'
$draftRoot = $PSScriptRoot
$expected = [ordered]@{
    '01_Introduction.md' = 1300
    '02_Related_Work.md' = 1100
    '03_Method.md' = 2200
    '04_Experiments.md' = 1500
    '05_Conclusion.md' = 350
}
$combinedPath = Join-Path $draftRoot $ManuscriptName
$combined = Get-Content -LiteralPath $combinedPath -Raw -Encoding UTF8
$records = @()
$allCitations = @()
foreach ($chapterName in $expected.Keys) {
    $chapterPath = Join-Path $draftRoot ('chapters/' + $chapterName)
    $chapterText = Get-Content -LiteralPath $chapterPath -Raw -Encoding UTF8
    $characters = ($chapterText -replace '\s','').Length
    if ($characters -lt $expected[$chapterName]) { throw "Too short: $chapterName" }
    if (-not $combined.Contains($chapterText.Trim())) { throw "Combined document mismatch: $chapterName" }
    $mathCount = [regex]::Matches($chapterText, '(?m)^\$\$\s*$').Count
    if ($mathCount % 2 -ne 0) { throw "Unbalanced display math: $chapterName" }
    $proseText = [regex]::Replace($chapterText, '(?s)\$\$.*?\$\$', '')
    $proseText = [regex]::Replace($proseText, '\$[^$\r\n]*\$', '')
    $citations = @([regex]::Matches($proseText, '\[(\d+(?:,\s*\d+)*)\]') | ForEach-Object { $_.Groups[1].Value -split ',\s*' })
    foreach ($citation in $citations) {
        if ([int]$citation -lt 1) { throw "Undefined citation: $citation" }
    }
    $allCitations += $citations
    $records += [ordered]@{
        file = $chapterName
        non_whitespace_characters = $characters
        chinese_characters = [regex]::Matches($chapterText, '[\u4e00-\u9fff]').Count
        pending_items = [regex]::Matches($chapterText, '〔待(?:补|核)：').Count
        display_equations = $mathCount / 2
        citations = $citations
    }
}
$actual = @(Get-ChildItem -LiteralPath (Join-Path $draftRoot 'chapters') -Filter *.md)
if ($actual.Count -ne 5) { throw 'Chapter count must be 5' }
$refs = Get-Content -LiteralPath (Join-Path $draftRoot 'refs/references.md') -Raw -Encoding UTF8
foreach ($citation in ($allCitations | Select-Object -Unique)) {
    if ($refs -notmatch ('(?m)^\[' + $citation + '\]')) { throw "Missing bibliography entry: $citation" }
}
$tags = @([regex]::Matches($combined, '\\tag\{(\d+)\}') | ForEach-Object { [int]$_.Groups[1].Value })
if (@($tags | Select-Object -Unique).Count -ne $tags.Count) { throw 'Duplicate equation number' }
if ($tags.Count -gt 0) {
    for ($j = 0; $j -lt $tags.Count; $j++) {
        if ($tags[$j] -ne ($j + 1)) { throw 'Nonsequential equation numbering' }
    }
}
$result = [ordered]@{
    check = 'first-draft structure, citation resolution and assembly'
    result = 'PASS'
    manuscript = $ManuscriptName
    numbered_equations = $tags.Count
    scope = 'Does not validate empirical results, numerical loss definitions, or submission readiness; original skill gate did not pass.'
    chapters = $records
    total_non_whitespace_characters = ($records | ForEach-Object { $_['non_whitespace_characters'] } | Measure-Object -Sum).Sum
    total_chinese_characters = ($records | ForEach-Object { $_['chinese_characters'] } | Measure-Object -Sum).Sum
    pending_body_items = ($records | ForEach-Object { $_['pending_items'] } | Measure-Object -Sum).Sum
    sha256 = (Get-FileHash -LiteralPath $combinedPath -Algorithm SHA256).Hash
}
$reportName = [System.IO.Path]::GetFileNameWithoutExtension($ManuscriptName) + '-validation.json'
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $draftRoot ('plan/review/' + $reportName)) -Encoding UTF8
$result | ConvertTo-Json -Depth 8
