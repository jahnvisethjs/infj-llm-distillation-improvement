$ErrorActionPreference = 'Stop'
$orgWorkspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$orgPrefix = $orgWorkspace.TrimEnd('\') + '\'
$orgProject = Join-Path $orgWorkspace 'capstone_person1'
$orgEvidence = Join-Path $orgProject 'evidence'

function Assert-WorkspacePath([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($orgPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes the intended workspace: $resolved"
    }
    return $resolved
}

$orgMoves = [Collections.Generic.List[object]]::new()
function Add-FileMove([string]$OldRelative, [string]$NewRelative) {
    $source = Assert-WorkspacePath (Join-Path $orgWorkspace $OldRelative)
    $destination = Assert-WorkspacePath (Join-Path $orgWorkspace $NewRelative)
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing source: $source" }
    if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination" }
    $orgMoves.Add([pscustomobject]@{old=$OldRelative.Replace('\','/'); new=$NewRelative.Replace('\','/'); source=$source; destination=$destination; sha256=(Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()})
}

foreach ($name in @('test_benchmark_manifests.py','test_data_contract.py','test_handoff_gate.py','test_pipeline_controls.py','test_snapshot_catalog.py','test_snapshot_stream.py','test_target_regressions.py')) {
    Add-FileMove "capstone_person1\$name" "capstone_person1\tests\$name"
}
foreach ($name in @('prepare_tokenizer.py','sync_notebook_controls.py','validate_notebook.py','validate_tokenizer_adapter.py','verify_reference_contract.py','verify_smoke_run.py')) {
    Add-FileMove "capstone_person1\$name" "capstone_person1\scripts\$name"
}
foreach ($name in @('initialize_v2.py','align_v2_notebook.py','align_benchmark_stage.py','record_alignment_batch.py','record_benchmark_batch.py','record_milestone.py')) {
    Add-FileMove "capstone_person1\$name" "capstone_person1\scripts\history\$name"
}
foreach ($name in @('REFERENCE_ENCODER.md','BENCHMARK_INPUT_CONTRACT.md')) {
    Add-FileMove "capstone_person1\$name" "capstone_person1\docs\contracts\$name"
}
foreach ($name in @('FIRST_ALIGNMENT_BATCH.md','BENCHMARK_INTERFACE_BATCH.md','FIXES_AND_HANDOFF.md')) {
    Add-FileMove "capstone_person1\$name" "capstone_person1\docs\milestones\$name"
}
foreach ($name in @('FIRST_BATCH_VERIFICATION.json','IMPLEMENTATION_STATUS.json','NEXT_STEP_VERIFICATION.json','REFERENCE_CONTRACT_VERIFICATION.json','SNAPSHOT_CONTRACT_SAMPLE.json')) {
    Add-FileMove "capstone_person1\$name" "capstone_person1\evidence\$name"
}
Add-FileMove 'capstone_person1\rebuild_v2.log' 'capstone_person1\logs\rebuild_v2.log'
foreach ($name in @('data_audit_full.md','FSE 570 Capstone Project Proposal Rubric.pdf','FSE_570_Proposal_Evidence_Grounded_QA.docx','FSE_570_Proposal_Mathematical_Reasoning.docx','FSE_570_Proposal_Python_Code_Generation.docx','INFJ team charter.pdf','INFJ_capstone_proposal.pdf','Project_Proposal.pdf')) {
    Add-FileMove $name "project_documents\$name"
}

# Capture the stable code/notebooks and all saved run/input artifacts before mutation.
$orgFixedHashes = @{}
foreach ($name in @('Person1_Full_Run.ipynb','Person1_Full_Run_v2.ipynb')) {
    $orgFixedHashes[$name] = (Get-FileHash -LiteralPath (Join-Path $orgWorkspace $name) -Algorithm SHA256).Hash.ToLower()
}
foreach ($name in @('pipeline_controls.py','person1_data_contract.py','benchmark_manifests.py','configuration.py','snapshot_catalog.py','snapshot_stream.py','streaming_rebuild.py','tokenizer_adapter.py','rebuild_from_snapshot.py','import_person3_manifests.py')) {
    $relative = "capstone_person1/$name"
    $orgFixedHashes[$relative] = (Get-FileHash -LiteralPath (Join-Path $orgWorkspace $relative) -Algorithm SHA256).Hash.ToLower()
}
$orgArtifacts = [Collections.Generic.List[object]]::new()
foreach ($folder in @('runs','outputs','preserved_inputs','inputs_from_team','configs','diagnostics')) {
    foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $orgProject $folder) -Recurse -File)) {
        $orgArtifacts.Add([pscustomobject]@{path=$file.FullName.Substring($orgPrefix.Length).Replace('\','/'); bytes=$file.Length; modified_ticks=$file.LastWriteTimeUtc.Ticks})
    }
}
New-Item -ItemType Directory -Path $orgEvidence -Force | Out-Null
$orgPlan = [pscustomobject]@{workspace=$orgWorkspace; moves=@($orgMoves | Select-Object old,new,sha256); stable_code_sha256=$orgFixedHashes; artifact_inventory=@($orgArtifacts)}
$orgManifest = Join-Path $orgEvidence 'ORGANIZATION_MANIFEST.json'
if (Test-Path -LiteralPath $orgManifest) { throw 'Organization manifest already exists; inspect the previous operation first.' }
$orgPlan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $orgManifest -Encoding UTF8

foreach ($move in $orgMoves) {
    $parent = Assert-WorkspacePath ([IO.Path]::GetDirectoryName($move.destination))
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Move-Item -LiteralPath $move.source -Destination $move.destination
}

$orgReference = Join-Path $orgProject 'docs\reference\Person1_Codex_Implementation_Plan.md'
Assert-WorkspacePath $orgReference | Out-Null
if (Test-Path -LiteralPath $orgReference) { throw 'Reference copy already exists.' }
New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($orgReference)) -Force | Out-Null
Copy-Item -LiteralPath 'C:\Users\jahnv\Downloads\Person1_Codex_Implementation_Plan.md' -Destination $orgReference
Write-Output "Moved $($orgMoves.Count) files. Stable code, notebooks and run artifacts stay at their recorded paths."
